"""Additive 84-schedule validation; reuses the existing two-GPU execution engine."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import numpy as np
import torch
from leaf.reproduction.experiments import salt_formulas as engine
from leaf.reproduction.experiments import salt_fit as cpu
from leaf.reproduction.experiments.gmac_planning import curves,sha,hash_file
from leaf.evaluation.runner import load_json,save,config_hash
from leaf.reproduction.runners.campaign_runner import CampaignRunner
from leaf.evaluation.data import mcnemar,holm
from leaf.backbones import SPECS
from leaf.planner.configs import ForecastConfig
from leaf.reproduction.ablation_models.merge_operator_planner import NormLat,NormPlanner
from leaf.planner.lagrange import plan
from leaf.latency.salt import FORMULAS,SaltCost,plan_with_salt
from leaf.planner.gmac import config_cost

ROOT=Path('results/salt_validation')
REPORT=Path('results/reports/t-proxy-validation-2026-10-17.md')
PLAN=Path('results/notes/plan/t-proxy-validation-plan-2026-10-17.md')
MODELS=('current_refit','station_rich','split_tile32')
SOURCE=cpu.SOURCE
PRIOR=cpu.ROOT/'gpu'
BINDING=load_json(SOURCE/'design.json')
CHOICES={}
for bb in SPECS:
    standard=[v for v in BINDING['backbones'][bb]['budgets'] if not v['id'].startswith('extra')]
    CHOICES[bb]=[standard[i]['id'] for i in (0,len(standard)//2,len(standard)-1)]


def runner(bb,out=False):
    r=CampaignRunner(bb,ROOT if out else SOURCE,BINDING);r.spec=SPECS[bb]
    if out:
        r.teacher_cache=SOURCE.resolve()/'teachers'
        r.reuse_roots=[PRIOR.resolve()/bb,cpu.PILOT.resolve()/bb,SOURCE.resolve()/bb]+r.reuse_roots
    d=ForecastConfig(Path(BINDING['backbones'][bb]['root'])/'groups.json',G=16,rank=32)
    return r,d


def frozen_proxy(bb):
    existing=cpu.PILOT/bb/'proxy.json'
    if existing.exists():return load_json(existing)
    base=Path(BINDING['backbones'][bb]['root']);model=load_json(base/'E1/latmodel.json')
    jobs=load_json(SOURCE/bb/'frozen.json')['jobs'];pairs=[];seen=set()
    from leaf.reproduction.experiments.salt_m3_pilot import merge_count
    for bid in [v['id'] for v in BINDING['backbones'][bb]['budgets']]:
        a,b='M0/gmac/'+bid,'swap_M3/gmac/'+bid
        cfg=jobs[a]['config'];key=config_hash(cfg);count=merge_count(jobs[b]['config'])
        if key in seen or not count:continue
        seen.add(key);values=[]
        for rnd in range(3):
            rows=[load_json(SOURCE/bb/'latency'/f"{n.replace('/','|')}#{rnd}.json") for n in (a,b)]
            values.append((rows[1]['measurements']['32']['ms_mean']-rows[0]['measurements']['32']['ms_mean'])/count)
        pairs.append(dict(budget=bid,n_merge=count,per_station_deltas_ms=values))
    assert pairs
    return dict(model=model,extra_m3_ms=max(0.,float(np.median([np.mean(v['per_station_deltas_ms']) for v in pairs]))),
        calibration=pairs,rule='original E1 coefficients frozen; same-config historical operator overhead median')


def model_path(bb,name):
    path=cpu.ROOT/'workload_cv'/bb/f'{name}.json'
    return path if path.exists() else cpu.ROOT/'cv'/bb/f'{name}.json'


def setup():
    if (ROOT/'design.json').exists():validate();return
    inputs={};proxies={}
    for bb in CHOICES:
        proxies[bb]=frozen_proxy(bb)
        paths=[SOURCE/bb/'frozen.json',Path(BINDING['backbones'][bb]['root'])/'groups.json',
            Path(BINDING['backbones'][bb]['root'])/'E1/latmodel.json']+[model_path(bb,n) for n in MODELS]
        for p in paths:inputs[str(p.resolve())]=hash_file(p)
    old={str(p.resolve()):hash_file(p) for folder in ('planner','models','evaluation') for p in Path(folder).rglob('*.py')}
    for p in ('leaf/reproduction/experiments/salt_formulas.py','leaf/reproduction/experiments/salt_fit.py','leaf/reproduction/experiments/gmac_planning.py','leaf/reproduction/experiments/salt_m3_pilot.py'):
        old[str(Path(p).resolve())]=hash_file(p)
    engine.immutable(ROOT/'design.json',dict(script_sha=hash_file(__file__),inputs=inputs,old_sources=old,
        choices=CHOICES,models=['frozen_t',*MODELS],frozen_proxies=proxies,
        protocol='84 M3 schedules + 42 read-only accuracy anchors; all configs fixed before confirm; two GPU workers, config dedup, ascending GMAC; confirm -> freeze -> val -> V2 -> bs32 timing last',
        statistics='Holm all 189 contrasts per split; paired same-budget comparisons with latency eligibility reported separately',
        disclosure='Historical calibration, confirm, valid and V2 already viewed; exploratory expansion. Rank32/G16 fixed. Same budget label is not iso-latency. Original coefficients frozen; refit model weights locked from earlier historical timing.'))


def validate():
    d=load_json(ROOT/'design.json');assert d['script_sha']==hash_file(__file__)
    for p,h in {**d['inputs'],**d['old_sources']}.items():assert hash_file(p)==h,p


def planning():
    setup()
    for bb,bids in CHOICES.items():
        dest=ROOT/bb/'plans.json'
        if dest.exists():continue
        r,d=runner(bb);cs=curves(r,d,'M3');jobs={};proxy=load_json(ROOT/'design.json')['frozen_proxies'][bb]
        lat=NormLat(proxy['model'],r.spec,proxy['extra_m3_ms']);old=NormPlanner(None,r.spec,lat,cs,d,'M3')
        history=load_json(SOURCE/bb/'frozen.json')['jobs']
        for name in ('frozen_t',*MODELS):
            cost=None if name=='frozen_t' else SaltCost(load_json(model_path(bb,name))['full_model'],r.spec,d)
            for bid in bids:
                budget=next(v for v in BINDING['backbones'][bb]['budgets'] if v['id']==bid);start=time.monotonic()
                if name=='frozen_t':
                    result=plan(old,budget['latency_ms']);assert result['state'] is not None
                    cfg=old.cfg(result['state']);pred=lat(cfg['prune_keep'],cfg['merge_keep'],ks=cfg['k'])
                else:
                    result=plan_with_salt(cs,r.spec,d,cost,budget['latency_ms']);assert not result.get('infeasible'),(bb,name,bid)
                    cfg=result['config'];pred=result['predicted_ms']
                assert 0<pred<=budget['latency_ms']-.1+1e-8
                jobs[name+'/'+bid]=dict(config=cfg,state=result['state'],predicted_ms=float(pred),budget=budget,
                    gmac=config_cost(cfg,r.spec),search_seconds=time.monotonic()-start,result=result)
                save(ROOT/bb/'planning_progress.json',dict(jobs=jobs))
                print('PLANNED',bb,name,bid,float(pred),flush=True)
        for bid in bids:
            jobs['gmac_M3/'+bid]=history['M3/gmac/'+bid]
            jobs['paper_M0/'+bid]=history['paper/latency/'+bid]
        engine.immutable(dest,dict(jobs=jobs,sha=sha(jobs)))


def report(final=False):
    records=[];tests=[]
    for bb in CHOICES:
        if not (ROOT/bb/'plans.json').exists():continue
        jobs=load_json(ROOT/bb/'plans.json')['jobs']
        metrics={s:load_json(ROOT/bb/f'{s}_metrics.json') if (ROOT/bb/f'{s}_metrics.json').exists() else {} for s in ('confirm','val','v2')}
        for name,job in jobs.items():
            values=[]
            for rnd in range(3):
                p=ROOT/bb/'latency'/f"{name.replace('/','|')}#{rnd}.json"
                if p.exists():values.append(load_json(p)['measurement']['ms_mean'])
            records.append(dict(backbone=bb,name=name,formula=name.split('/')[0],budget=name.split('/')[1],
                gmac=config_cost(job['config'],SPECS[bb]),predicted_ms=job.get('predicted_ms'),
                budget_ms=job.get('budget',{}).get('latency_ms'),confirm_kl=metrics['confirm'].get(name,{}).get('kl'),
                valid=metrics['val'].get(name,{}).get('top1'),v2=metrics['v2'].get(name,{}).get('top1'),
                latency_ms=float(np.mean(values)) if len(values)==3 else None,latency_sd=float(np.std(values,ddof=1)) if len(values)==3 else None))
        if final:
            r,_=runner(bb,True)
            for model in MODELS:
                for bid in CHOICES[bb]:
                    name=model+'/'+bid
                    for ref in ('frozen_t/'+bid,'gmac_M3/'+bid,'paper_M0/'+bid):
                        for split in ('val','v2'):
                            a=r.cached(jobs[name]['config'],split);b=r.cached(jobs[ref]['config'],split)
                            p,w,l=mcnemar(a['correct'],b['correct'])
                            tests.append(dict(backbone=bb,name=name,reference=ref,split=split,p=p,wins=w,losses=l,
                                delta_pp=100*float(a['correct'].mean()-b['correct'].mean())))
    if final:
        assert len(tests)==378
        for split in ('val','v2'):holm([v for v in tests if v['split']==split])
    save(ROOT/'results.json',dict(complete=final,records=records,paired_tests=tests,time=time.time()))
    summary=[]
    if final:
        for model in ('frozen_t',*MODELS):
            rows=[v for v in records if v['formula']==model]
            summary.append(dict(model=model,n=len(rows),mape=float(np.mean([100*abs(v['predicted_ms']/v['latency_ms']-1) for v in rows])),
                budget_pass=sum(v['latency_ms']<=v['budget_ms'] for v in rows),
                worst_overshoot_ms=max(max(0.,v['latency_ms']-v['budget_ms']) for v in rows)))
        save(ROOT/'summary.json',summary)
    lines=['# Validation of t on 7 backbones — 2026-10-17','',
        'Status: '+('main pass and audit complete; not independently reviewed.' if final else 'running; results are saved after each configuration.'),'',
        '84 M3 schedules = 7 backbones × 3 budgets (low/mid/high) × 4 proxies (frozen_t, current_refit, station_rich, split_tile32). Plus 42 control arms: paper M0 and GMAC-M3; duplicate configs are counted once. The quality probe and the DP are used unchanged, rank32/G16, BF16 batch32. All coefficients and budgets are locked before confirm; nothing is selected by accuracy. The AugReg-B model uses the workload-grouped CV version.','',
        'Two GPU workers take tasks dynamically by GMAC; confirm runs to completion, then the schedule is frozen, then valid/V2; latency at bs32 in three interleaved rounds at the end, with the other GPU idle. Accuracy is reused only for the exact same config/batch/n/checkpoint; timing is re-measured in the same session. All old code/results are kept unchanged.','',
        '| Backbone | Proxy / budget | GMAC | t ms | Confirm KL | Valid top1 | V2 top1 | Latency ms |','|---|---|---:|---:|---:|---:|---:|---:|']
    fmt=lambda v:'—' if v is None else f'{v:.4f}'
    for row in records:lines.append('| '+' | '.join([row['backbone'],row['name']]+[fmt(row[k]) for k in ('gmac','predicted_ms','confirm_kl','valid','v2','latency_ms')])+' |')
    if final:
        lines+=['','| Proxy | N | MAPE % | Mean latency ≤ budget | Largest overshoot ms |','|---|---:|---:|---:|---:|']
        for v in summary:lines.append(f"| {v['model']} | {v['n']} | {v['mape']:.3f} | {v['budget_pass']}/{v['n']} | {v['worst_overshoot_ms']:.3f} |")
    lines+=['', 'Limitations: one GPU type and rank32; the historical image data have been seen before, and calibration/confirm share two images per the earlier audit. The same predicted budget is not automatically iso-latency. Holm over all 189 contrasts per split; for latency closeness and extrapolation, read the final analysis table.','',
        f'Raw: `{ROOT.resolve()}`. `design.json`, `plans.json`, `tasks.json`, `results.json`, per-image arrays and per-GPU logs. Previous results: [formula comparison](t-proxy-formulas-2026-10-17.md).']
    REPORT.write_text('\n'.join(lines)+'\n')
    tax=REPORT.parent/'REPORT_TAXONOMY.md';text=tax.read_text();prefix=f'| [{REPORT.name}]'
    row=f'| [{REPORT.name}]({REPORT.name}) | 84 M3 schedules/7 backbones/3 budgets, frozen t and 3 new models; accuracy on two GPUs, final timing at bs32, historical data/rank32. | '+('Main pass and audit complete; not independently reviewed' if final else 'Running')+' |'
    if prefix in text:text='\n'.join(row if l.startswith(prefix) else l for l in text.splitlines())+'\n'
    else:
        at=text.index('|---|---|---|',text.index('## D.'))+len('|---|---|---|');text=text[:at]+'\n'+row+text[at:]
    plan_prefix=f'| [{PLAN.name}]'
    plan_row=f'| [{PLAN.name}](../plan/{PLAN.name}) | 84 main M3 schedules, near-GMAC ranking check, stress and per-session stability; adds code, keeps old code. | '+('Main pass complete; extra checks next' if final else 'Main pass running')+f' | [{REPORT.name}]({REPORT.name}) |'
    if plan_prefix in text:text='\n'.join(plan_row if l.startswith(plan_prefix) else l for l in text.splitlines())+'\n'
    else:
        at=text.index('|---|---|---|---|',text.index('### G1.'))+len('|---|---|---|---|');text=text[:at]+'\n'+plan_row+text[at:]
    tax.write_text(text)


# Bind the old execution engine to the new campaign without editing its source.
engine.ROOT=ROOT;engine.CHOICES=CHOICES;engine.FORMULAS={k:FORMULAS[k] for k in MODELS}
engine.__file__=str(Path(__file__).resolve());engine.runner=runner;engine.setup=setup
engine.validate=validate;engine.planning=planning;engine.report=report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['plan','launch','worker','timing','report']);parser.add_argument('--split',choices=['confirm','val','v2']);args=parser.parse_args()
    torch.set_num_threads(4);torch.manual_seed(151)
    ROOT.mkdir(parents=True,exist_ok=True)
    if args.action=='launch':engine.launch()
    elif args.action=='worker':engine.worker(args.split)
    elif args.action=='timing':engine.timing()
    elif args.action=='plan':planning()
    else:report()
