"""Additive pilot of the existing Lat/NormLat proxy with M3. Old artifacts read-only."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import torch
import leaf.evaluation.determinism
from leaf.evaluation.runner import load_json, save, config_hash, require_idle_gpu
from leaf.reproduction.runners.campaign_runner import CampaignRunner, build
from leaf.latency.timing import measure_latency
from leaf.evaluation.data import mcnemar, holm
from leaf.planner.configs import ForecastConfig
from leaf.reproduction.ablation_models.merge_operator_planner import NormLat, NormPlanner
from leaf.planner.lagrange import plan, transition_costs
from leaf.planner.schedule_caps import canonical
from leaf.planner.gmac import config_cost
from leaf.backbones import SPECS
from leaf.reproduction.experiments.gmac_planning import curves, sha, hash_file

SOURCE = Path('results/gmac_planning')
ROOT = Path('results/salt_m3_pilot')
CHOICES = {'deit_s': ['DR2.3', 'DR2.7', 'DR3.1'],
           'augreg_s': ['dense0.5', 'dense0.68', 'dense0.86'],
           'mae_b': ['DR8.7', 'DR10.4', 'DR11.5']}
REPORT = 't-m3-pilot-2026-10-17.md'
MAIN = Path(__file__).resolve().parents[3]   # this repository (the campaign exported to the research repository too)


def immutable(path, obj):
    if path.exists():
        assert load_json(path) == obj, f'changed immutable artifact {path}'
    else:
        save(path, obj)


def runner(bb, out=False):
    design = load_json(SOURCE / 'design.json')
    r = CampaignRunner(bb, ROOT if out else SOURCE, design)
    r.spec = SPECS[bb]
    if out:
        r.teacher_cache = SOURCE.resolve() / 'teachers'
        r.reuse_roots.insert(0, SOURCE.resolve() / bb)
    d = ForecastConfig(Path(design['backbones'][bb]['root']) / 'groups.json', G=16, rank=32)
    return r, d


def merge_count(cfg):
    n, count = 197, 0
    for p, m in zip(cfg['prune_keep'], cfg['merge_keep']):
        p = min(n, p); m = min(p, m)
        count += m < p
        n = m
    return count


def setup():
    if (ROOT / 'design.json').exists():
        assert load_json(ROOT / 'design.json')['script_sha'] == hash_file(__file__)
        return
    design = load_json(SOURCE / 'design.json')
    inputs = {str(SOURCE.resolve() / 'design.json'): hash_file(SOURCE / 'design.json')}
    for bb in CHOICES:
        base = Path(design['backbones'][bb]['root'])
        for path in [base / 'E1/latmodel.json', base / 'groups.json', SOURCE / bb / 'frozen.json']:
            inputs[str(path.resolve())] = hash_file(path)
        for path in (SOURCE / bb / 'latency').glob('*.json'):
            if path.name.startswith(('M0|gmac|','swap_M3|gmac|')):
                inputs[str(path.resolve())] = hash_file(path)
    immutable(ROOT / 'design.json', dict(script_sha=hash_file(__file__), choices=CHOICES,
        inputs=inputs, batch=32, rank=32, groups=16, selection='existing separable quality surrogate only',
        proxy='original selected bs32 Lat + nonnegative median same-config M3 station overhead from historical timing',
        protocol='CPU plan -> confirm -> freeze -> val/V2 on two GPUs -> latency bs32 last, three interleaved rounds',
        disclosure='Historical splits already viewed; exploratory pilot. No new quality model fit. D rank fixed, not searched. Same-budget contrasts are not iso-cost claims.',
        old_sources={str(p): hash_file(p) for folder in ('planner', 'evaluation', 'models') for p in Path(folder).rglob('*.py')}))


def planning(bb):
    path = ROOT / bb / 'plans.json'
    if path.exists(): return
    print('PLAN', bb, flush=True)
    r, d = runner(bb)
    design = load_json(SOURCE / 'design.json')
    old = load_json(SOURCE / bb / 'frozen.json')['jobs']
    models = load_json(Path(design['backbones'][bb]['root']) / 'E1/latmodel.json')
    assert models['fit']['32']['chosen'] == 'gmac', 'pilot expects the existing GMAC-based proxy'
    pairs = []; seen_pairs=set()
    for budget in design['backbones'][bb]['budgets']:
        bid = budget['id']; a, b = f'M0/gmac/{bid}', f'swap_M3/gmac/{bid}'
        if b not in old:
            candidates = [name for name in old if 'swap' in name.lower() and name.endswith('/' + bid)]
            if not candidates: continue
            b = candidates[0]
        m = merge_count(old[b]['config'])
        if not m: continue
        ca, cb = old[a]['config'], old[b]['config']
        if config_hash(ca) in seen_pairs: continue
        seen_pairs.add(config_hash(ca))
        assert {k:v for k,v in cb.items() if k not in ('kind','merge')} == {k:v for k,v in ca.items() if k != 'kind'}
        deltas = []
        for rnd in range(3):
            rows = [load_json(SOURCE / bb / 'latency' / f"{name.replace('/', '|')}#{rnd}.json") for name in (a,b)]
            deltas.append((rows[1]['measurements']['32']['ms_mean'] - rows[0]['measurements']['32']['ms_mean']) / m)
        pairs.append(dict(budget=bid, n_merge=m, per_station_deltas_ms=deltas))
    assert pairs, 'missing matched operator calibration'
    extra = max(0., float(np.median([np.mean(p['per_station_deltas_ms']) for p in pairs])))
    immutable(ROOT / bb / 'proxy.json', dict(model=models, extra_m3_ms=extra, calibration=pairs,
        rule='median across unique same-schedule budget pairs, averaged across 3 rounds; clipped at zero'))
    lat = NormLat(models, r.spec, extra)
    p = NormPlanner(None, r.spec, lat, curves(r, d, 'M3'), d, 'M3')
    jobs = {}; diagnostics = []
    for bid in CHOICES[bb]:
        budget = next(b for b in design['backbones'][bb]['budgets'] if b['id'] == bid)
        start = time.monotonic(); result = plan(p, budget['latency_ms'])
        assert result['state'] is not None, (bb, bid, result)
        cfg = p.cfg(result['state'])
        predicted = float(lat(cfg['prune_keep'], cfg['merge_keep'], ks=cfg['k']))
        assert predicted <= budget['latency_ms'] - .1 + 1e-8
        # Verify DP edge costs agree with the full proxy on the unprojected grid path.
        raw = result['grid']; times, _ = transition_costs(p); i = 0; edge_sum = lat.c0
        grid = {int(n): j for j,n in enumerate(p.jt.counts)}; cuts = dict(zip(raw['sites'],raw['counts']))
        for l in range(r.spec.depth):
            j = grid[cuts[l]] if l in cuts else i
            edge_sum += times[l,i,j]; i = j
        rawcfg = p.cfg(p.state(raw['sites'],raw['counts']))
        full = lat(rawcfg['prune_keep'],rawcfg['merge_keep'],ks=rawcfg['k'])
        assert abs(edge_sum-full) < 1e-7, (edge_sum,full)
        name = 't_M3/' + bid
        jobs[name] = dict(config=cfg, state=result['state'], budget=budget, predicted_ms=predicted,
            gmac=config_cost(cfg,r.spec), search_seconds=time.monotonic()-start, result=result)
        for ref in (f'M3/gmac/{bid}', f'paper/latency/{bid}'):
            if ref not in old:
                matches = [n for n in old if n.startswith('paper') and n.endswith('/latency/' + bid)]
                assert len(matches)==1, (ref,matches,list(old))
                ref = matches[0]
            jobs[ref] = dict(old[ref], budget=budget)
        diagnostics.append(dict(name=name, sites=result['state']['sites'], counts=result['state']['counts'],
            predicted_ms=predicted, budget_ms=budget['latency_ms'], gmac=config_cost(cfg,r.spec), seconds=jobs[name]['search_seconds']))
        save(ROOT / bb / 'planning_progress.json', diagnostics)
        print('PLANNED', bb, diagnostics[-1], flush=True)
    immutable(path, dict(jobs=jobs, sha=sha(jobs), proxy_sha=hash_file(ROOT / bb / 'proxy.json')))


def validate():
    design = load_json(ROOT / 'design.json')
    assert design['script_sha'] == hash_file(__file__)
    for path, digest in {**design['inputs'], **design['old_sources']}.items():
        assert hash_file(path) == digest, path
    for bb in CHOICES:
        doc = load_json(ROOT / bb / 'plans.json'); assert sha(doc['jobs']) == doc['sha']


def worker(backbones):
    validate(); require_idle_gpu()
    for bb in backbones:
        if (ROOT / bb / 'accuracy_done.json').exists(): continue
        r, _ = runner(bb, True); _ = r.vit
        jobs = load_json(ROOT / bb / 'plans.json')['jobs']
        profile_path = ROOT / bb / 'profile.json'
        if not profile_path.exists():
            # Exact task/batch feasibility and short throughput profile; no schedule selection.
            cfg = next(v['config'] for k,v in jobs.items() if k.startswith('t_M3/'))
            model = r.inference_model(cfg); rows=[]
            for batch in (16,32):
                x=torch.randn(batch,3,224,224,device='cuda'); torch.cuda.reset_peak_memory_stats()
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                    for _ in range(2): y=model(x)
                    torch.cuda.synchronize(); start=time.monotonic()
                    for _ in range(5): y=model(x)
                    torch.cuda.synchronize()
                rows.append(dict(batch=batch,images_per_s=5*batch/(time.monotonic()-start),peak_bytes=torch.cuda.max_memory_allocated(),finite=bool(torch.isfinite(y).all())))
                del x,y
            assert all(v['finite'] for v in rows)
            assert rows[-1]['peak_bytes'] < torch.cuda.get_device_properties(0).total_memory - 2*2**30
            save(profile_path, dict(rows=rows,chosen=32,reason='paired BF16 accuracy with historical bs32 teachers and controls'))
            del model; gc.collect(); torch.cuda.empty_cache()
        for split in ('confirm','val','v2'):
            if split != 'confirm': assert (ROOT / bb / 'freeze.json').exists()
            metrics = {}
            for name, job in jobs.items():
                require_idle_gpu()
                save(ROOT / bb / 'status.json', dict(phase=split,name=name,time=time.time()))
                arr = r.run(job['config'],split,meta=dict(pilot='t_m3',name=name))
                assert arr is not None
                metrics[name] = dict(top1=float(arr['correct'].mean()*100),kl=float(arr['kl'].mean()),n=len(arr['kl']),config_sha=config_hash(job['config']))
                save(ROOT / bb / f'{split}_metrics.json',metrics)
            if split == 'confirm':
                immutable(ROOT / bb / 'freeze.json', dict(plans_sha=load_json(ROOT/bb/'plans.json')['sha'],confirm_sha=hash_file(ROOT/bb/'confirm_metrics.json'),selection='all three preplanned schedules; no confirm-based selection'))
        save(ROOT / bb / 'accuracy_done.json', dict(time=time.time()))
        del r; gc.collect(); torch.cuda.empty_cache()


def timing():
    validate()
    assert all((ROOT / b / 'accuracy_done.json').exists() for b in CHOICES)
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
    assert not any(s.strip() and int(s.strip())!=os.getpid() for s in apps.splitlines()), apps
    for bb in CHOICES:
        if (ROOT / bb / 'latency_done.json').exists(): continue
        r,_=runner(bb,True); _=r.vit; jobs=load_json(ROOT/bb/'plans.json')['jobs']; seen={}
        # A restart starts a fresh interleaved session; preserve partial old measurements.
        directory=ROOT/bb/'latency'
        if directory.exists(): directory.rename(ROOT/bb/f'latency_interrupted_{time.time_ns()}')
        for rnd in range(3):
            names=sorted(jobs); names=names if rnd%2==0 else names[::-1]
            for name in names:
                cfg=jobs[name]['config']; key=(rnd,config_hash(cfg)); require_idle_gpu()
                if key in seen: row=dict(seen[key],alias=True)
                else:
                    model=build(r.vit,r.spec,cfg); row=measure_latency(model,32,iters=50,warmup=10)
                    seen[key]=row; del model; gc.collect(); torch.cuda.empty_cache()
                save(directory/f"{name.replace('/','|')}#{rnd}.json",dict(name=name,round=rnd,config_sha=key[1],measurement=row))
                print('LATENCY',bb,rnd,name,row['ms_mean'],flush=True)
        save(ROOT/bb/'latency_done.json',dict(time=time.time(),rounds=3,batch=32))
        del r; gc.collect(); torch.cuda.empty_cache()


def report():
    records=[]; tests=[]; complete=True
    lines=['# Pilot t + M3 — 2026-10-17','',
        'Uses Lat/NormLat and the paper DP unchanged; t = c0 + a·GMAC + c_st·K + c_D·N_D + δ_M3·K_merge. Base coefficients are per backbone; δ_M3 is the median M3/M0 difference on the same schedule, from historical timing. No GPU timing during search; no fitting on valid/top-1. Rank32/G16 fixed. The historical data have been seen before; this is an exploratory pilot.', '',
        'Three budget levels per backbone; all schedules found are kept, with no selection by confirm or valid. Controls with the same budget label do not necessarily have the same actual latency/GMAC. Latency at bs32 is measured at the end, in three interleaved rounds within the same session.','',
        '| Backbone | Arm | Predicted t ms | GMAC | Confirm KL | Valid top1 | V2 top1 | Latency ms |','|---|---|---:|---:|---:|---:|---:|---:|']
    for bb in CHOICES:
        if not (ROOT/bb/'plans.json').exists(): complete=False; continue
        r,_=runner(bb,True); jobs=load_json(ROOT/bb/'plans.json')['jobs']
        metrics={s:load_json(ROOT/bb/f'{s}_metrics.json') if (ROOT/bb/f'{s}_metrics.json').exists() else {} for s in ('confirm','val','v2')}
        proxy=load_json(ROOT/bb/'proxy.json'); lat=NormLat(proxy['model'],r.spec,proxy['extra_m3_ms'])
        for name,job in jobs.items():
            cfg=job['config']; values=[]
            for rnd in range(3):
                path=ROOT/bb/'latency'/f"{name.replace('/','|')}#{rnd}.json"
                if path.exists(): values.append(load_json(path)['measurement']['ms_mean'])
            # Only M3 proxy predictions are in-domain; paper M0 does not pay M3 overhead.
            pred=lat(cfg['prune_keep'],cfg['merge_keep'],ks=cfg['k']) if cfg['kind']=='pdm' else None
            row=dict(backbone=bb,arm=name,predicted_ms=pred,gmac=config_cost(cfg,r.spec),
                confirm_kl=metrics['confirm'].get(name,{}).get('kl'),valid=metrics['val'].get(name,{}).get('top1'),
                v2=metrics['v2'].get(name,{}).get('top1'),latency_ms=float(np.mean(values)) if len(values)==3 else None)
            complete &= all(row[k] is not None for k in ('confirm_kl','valid','v2','latency_ms'))
            records.append(row)
            fmt=lambda v:'—' if v is None else f'{v:.4f}'
            lines.append('| '+' | '.join([bb,name]+[fmt(row[k]) for k in ('predicted_ms','gmac','confirm_kl','valid','v2','latency_ms')])+' |')
        for bid in CHOICES[bb]:
            arm='t_M3/'+bid
            for ref in jobs:
                if ref==arm or not ref.endswith('/'+bid): continue
                for split in ('val','v2'):
                    a=r.cached(jobs[arm]['config'],split); b=r.cached(jobs[ref]['config'],split)
                    if a is None or b is None: continue
                    pv,wins,losses=mcnemar(a['correct'],b['correct'])
                    tests.append(dict(backbone=bb,arm=arm,reference=ref,split=split,p=pv,wins=wins,losses=losses,
                        delta_pp=100*float(a['correct'].mean()-b['correct'].mean())))
    for split in ('val','v2'):
        subset=[v for v in tests if v['split']==split]
        if subset:
            holm(subset)
    save(ROOT/'results.json',dict(complete=bool(complete),records=records,paired_tests=tests,time=time.time()))
    if complete:
        validate();checked=0
        for bb in CHOICES:
            jobs=load_json(ROOT/bb/'plans.json')['jobs']
            for split,n in [('confirm',20018),('val',50000),('v2',10000)]:
                for job in jobs.values():
                    path=ROOT/bb/split/'runs'/config_hash(job['config'])
                    meta=load_json(path.with_suffix('.json')); arr=dict(np.load(path.with_suffix('.npz')))
                    assert meta['config']==job['config'] and meta['batch_size']==32 and meta['n']==n
                    assert all(arr[k].shape==(n,) for k in ('kl','pred','correct')) and np.isfinite(arr['kl']).all()
                    checked+=1
        save(ROOT/'audit.json',dict(complete=True,old_sources_unchanged=True,plans_and_inputs_unchanged=True,
            named_arrays_checked=checked,time=time.time()))
    lines+=['','Status: '+('execution complete; not independently reviewed.' if complete else 'running; the table is updated after each arm.'),
        '',f'Raw: `{ROOT.resolve()}`. Machine-readable results and McNemar/Holm per split: `results.json`. Contrasts are at the same budget; check the latency measured in the same session before concluding iso-latency.']
    for repo in {Path.cwd(), MAIN}:
        directory=repo/'results/reports'; directory.mkdir(parents=True,exist_ok=True)
        (directory/REPORT).write_text('\n'.join(lines)+'\n')
        taxonomy=directory/'REPORT_TAXONOMY.md'
        text=taxonomy.read_text(); prefix=f'| [{REPORT}]'
        row=f'| [{REPORT}]({REPORT}) | Pilot Lat/NormLat + M3 on DeiT-S/AugReg-S/MAE-B, 9 schedules; confirm/val/V2 and final timing at bs32. Rank32 fixed; historical data; a same-budget comparison is not yet iso-latency. | '+('Execution complete; not independently reviewed' if complete else 'Running')+' |'
        if prefix in text:text='\n'.join(row if line.startswith(prefix) else line for line in text.splitlines())+'\n'
        else:
            marker='## D. Station placement, token counts and frontier'; at=text.index('|---|---|---|',text.index(marker))+len('|---|---|---|')
            text=text[:at]+'\n'+row+text[at:]
        taxonomy.write_text(text)
    print('REPORT', 'completed' if complete else 'running',len(records),'arms',flush=True)


def launch():
    import fcntl
    ROOT.mkdir(parents=True,exist_ok=True)
    lock=(ROOT/'scheduler.lock').open('w'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    setup()
    for bb in CHOICES: planning(bb)
    report()
    save(ROOT/'status.json',dict(phase='accuracy',pid=os.getpid(),time=time.time()))
    procs=[]
    for gpu,bbs in [(0,['deit_s','augreg_s']),(1,['mae_b'])]:
        log=(ROOT/f'accuracy_gpu{gpu}.log').open('a')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONUNBUFFERED='1')
        process=subprocess.Popen([sys.executable,__file__,'worker','--backbones',*bbs],env=env,stdout=log,stderr=subprocess.STDOUT)
        procs.append(process)
    while any(p.poll() is None for p in procs):
        time.sleep(25);report()
    assert all(p.returncode==0 for p in procs), 'accuracy worker failed; inspect logs and restart'
    save(ROOT/'status.json',dict(phase='latency',pid=os.getpid(),time=time.time()))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHONUNBUFFERED='1')
    with (ROOT/'latency.log').open('a') as log:
        subprocess.run([sys.executable,__file__,'timing'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    report();save(ROOT/'status.json',dict(phase='completed',pid=os.getpid(),time=time.time()))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['plan','worker','timing','report','launch'])
    parser.add_argument('--backbones',nargs='+',choices=list(CHOICES),default=list(CHOICES));args=parser.parse_args()
    torch.set_num_threads(4);torch.manual_seed(151)
    if args.action=='plan':
        setup()
        for bb in args.backbones:planning(bb)
    elif args.action=='worker':worker(args.backbones)
    elif args.action=='timing':timing()
    elif args.action=='report':report()
    else:launch()
