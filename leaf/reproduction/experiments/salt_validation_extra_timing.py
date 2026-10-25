"""Predeclared timing-only pairs/stress and two extra sessions after 84-arm accuracy."""
import argparse
import bisect
import gc
import json
import os
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import numpy as np
import torch
from leaf.reproduction.experiments import salt_validation as main
from leaf.evaluation.runner import load_json,save,config_hash
from leaf.latency.timing import measure_latency
from leaf.reproduction.runners.campaign_runner import build
from leaf.reproduction.ablation_models.merge_operator_planner import norm_config,NormLat
from leaf.latency.salt import SaltCost
from leaf.planner.gmac import config_cost
from leaf.planner.schedule_caps import canonical
from leaf.reproduction.experiments.gmac_planning import hash_file,sha

ROOT=main.ROOT/'extra'


def signature(cfg):
    g=canonical(dict(p=cfg['prune_keep'],m=cfg['merge_keep']),197)
    return sha(dict(p=g['p'],m=g['m'],active=sorted(int(k) for k,v in cfg['k'].items() if v>0),rank=cfg['rank']))


def models(bb,d):
    proxy=load_json(main.ROOT/'design.json')['frozen_proxies'][bb]
    lat=NormLat(proxy['model'],main.SPECS[bb],proxy['extra_m3_ms'])
    result={'frozen_t':lambda cfg:float(lat(cfg['prune_keep'],cfg['merge_keep'],ks=cfg['k']))}
    for name in main.MODELS:
        result[name]=SaltCost(load_json(main.model_path(bb,name))['full_model'],main.SPECS[bb],d)
    return result


def planning():
    ROOT.mkdir(parents=True,exist_ok=True)
    if not (main.ROOT/'design.json').exists():raise RuntimeError('main design must exist first')
    for index,bb in enumerate(main.CHOICES):
        path=ROOT/bb/'design.json'
        if path.exists():continue
        r,d=main.runner(bb);sp=r.spec;rng=np.random.default_rng(1771+index);pool={}
        kvals=sorted({1,2,3,5,8,sp.depth-1})
        for _ in range(6000):
            if len(pool)>=3000:break
            k=int(rng.choice(kvals));end=int(rng.choice([3,32,64,128]))
            sites=sorted(int(v) for v in rng.choice(np.arange(1,sp.depth),k,replace=False))
            counts=sorted([int(v) for v in rng.choice(np.arange(end+1,sp.n_tokens),k-1,replace=False)],reverse=True)+[end]
            st=dict(sites=sites,counts=counts,thetas=[float(v) for v in rng.choice([0.,.5,1.],k)],ks=[float(v) for v in rng.choice([0.,1.,3.],k)])
            cfg=norm_config(st,sp,d,'M3');sig=signature(cfg)
            if sig in pool:continue
            pool[sig]=dict(config=cfg,state=st,gmac=config_cost(cfg,sp),signature=sig)
        candidates=sorted(pool.values(),key=lambda v:v['gmac']);costs=[v['gmac'] for v in candidates]
        used=set();pairs=[];jobs={};predictors=models(bb,d)
        def add(name,row):
            jobs[name]=dict(row,predictions={n:float(fn(row['config'])) for n,fn in predictors.items()})
        # Sample all GMAC ranges before looking at any measured latency.
        order=list(rng.permutation(len(candidates)))
        for i in order:
            a=candidates[i]
            if a['signature'] in used:continue
            lo=bisect.bisect_left(costs,a['gmac']*.9901);hi=bisect.bisect_right(costs,a['gmac']*1.01)
            possible=[b for b in candidates[lo:hi] if b['signature'] not in used and b['signature']!=a['signature'] and len(b['state']['sites'])!=len(a['state']['sites']) and abs(a['gmac']-b['gmac'])<=.01*max(a['gmac'],b['gmac'])]
            if not possible:continue
            b=min(possible,key=lambda b:abs(a['gmac']-b['gmac']));p=len(pairs)
            na,nb=f'pair{p:02d}a',f'pair{p:02d}b';add(na,a);add(nb,b)
            pairs.append(dict(a=na,b=nb,relative_gmac_gap=abs(a['gmac']-b['gmac'])/max(a['gmac'],b['gmac'])))
            used.update([a['signature'],b['signature']])
            if len(pairs)==20:break
        assert len(pairs)==20,(bb,len(pairs))
        stress=[]
        for k in (1,2):
            for end in (3,32,64,128):
                sites=[sp.depth//2] if k==1 else [1,sp.depth//2]
                counts=[end] if k==1 else [round((197+end)/2),end]
                st=dict(sites=sites,counts=counts,thetas=[.5]*k,ks=[1.]*k)
                cfg=norm_config(st,sp,d,'M3');name=f'stress_K{k}_N{end}'
                add(name,dict(config=cfg,state=st,gmac=config_cost(cfg,sp),signature=signature(cfg)));stress.append(name)
        main.engine.immutable(path,dict(backbone=bb,jobs=jobs,pairs=pairs,stress=stress,seed=1771+index,
            source_sha=hash_file(__file__),parent_design_sha=hash_file(main.ROOT/'design.json')))
        print('EXTRA PLANNED',bb,len(jobs),'fixtures',flush=True)
    save(ROOT/'status.json',dict(phase='waiting_for_main',time=time.time()))


def representative(bb):
    jobs=load_json(main.ROOT/bb/'plans.json')['jobs'];bid=main.CHOICES[bb][1]
    return {n:row for n,row in jobs.items() if n.endswith('/'+bid)}


def measure_session(session,bb):
    done=ROOT/bb/f'session{session}_done.json'
    if done.exists():return
    main.validate();main.engine.idle_both();r,_=main.runner(bb,True);_=r.vit
    design=load_json(ROOT/bb/'design.json');assert design['source_sha']==hash_file(__file__)
    rows={f'rep/{n}':v for n,v in representative(bb).items()}
    if session==1:rows.update(design['jobs'])
    directory=ROOT/bb/f'session{session}'
    if directory.exists():directory.rename(ROOT/bb/f'interrupted_session{session}_{time.time_ns()}')
    seen={}
    for rnd in range(3):
        names=sorted(rows);names=names if rnd%2==0 else names[::-1]
        for name in names:
            cfg=rows[name]['config'];key=(rnd,config_hash(cfg));main.engine.idle_both()
            if key in seen:measurement=dict(seen[key],alias=True)
            else:
                model=build(r.vit,r.spec,cfg);measurement=measure_latency(model,32,iters=50,warmup=10)
                assert measurement['peak_vram_gb']<14,'bs32 lacks reserve'
                seen[key]=measurement;del model;gc.collect();torch.cuda.empty_cache()
            save(directory/f"{name.replace('/','|')}#{rnd}.json",dict(name=name,session=session,round=rnd,config_sha=key[1],measurement=measurement))
            save(ROOT/'status.json',dict(phase='timing',session=session,backbone=bb,name=name,round=rnd,time=time.time()))
            print('EXTRA LATENCY',session,bb,rnd,name,measurement['ms_mean'],flush=True)
    save(done,dict(n=len(rows),rounds=3,batch=32,time=time.time()))
    del r;gc.collect();torch.cuda.empty_cache()


def samples(bb,name,session):
    if session==0:base=main.ROOT/bb/'latency';name=name.removeprefix('rep/')
    else:base=ROOT/bb/f'session{session}'
    paths=[base/f"{name.replace('/','|')}#{rnd}.json" for rnd in range(3)]
    return np.array([load_json(p)['measurement']['ms_mean'] for p in paths]) if all(p.exists() for p in paths) else None


def report(final=False):
    rankings=[];stress=[];stability=[]
    for bb in main.CHOICES:
        path=ROOT/bb/'design.json'
        if not path.exists():continue
        design=load_json(path)
        for pair in design['pairs']:
            a,b=(samples(bb,pair[n],1) for n in ('a','b'))
            if a is None or b is None:continue
            tau=max(.05,2*float(np.sqrt((a.var(ddof=1)+b.var(ddof=1))/2)))
            for model in ('frozen_t',*main.MODELS):
                pa,pb=(design['jobs'][pair[n]]['predictions'][model] for n in ('a','b'))
                rankings.append(dict(backbone=bb,model=model,**pair,actual_delta_ms=float(a.mean()-b.mean()),predicted_delta_ms=pa-pb,
                    eligible=bool(abs(a.mean()-b.mean())>tau),correct=bool((pa-pb)*(a.mean()-b.mean())>0)))
        for name in design['stress']:
            actual=samples(bb,name,1)
            if actual is None:continue
            for model,pred in design['jobs'][name]['predictions'].items():
                stress.append(dict(backbone=bb,name=name,model=model,predicted_ms=pred,actual_ms=float(actual.mean()),error_pct=100*abs(pred/actual.mean()-1)))
        if not (main.ROOT/bb/'plans.json').exists():continue
        for name,row in representative(bb).items():
            means=[]
            for session in range(3):
                arr=samples(bb,'rep/'+name,session)
                if arr is not None:means.append(float(arr.mean()))
            budget=row.get('budget',{}).get('latency_ms')
            stability.append(dict(backbone=bb,name=name,sessions=len(means),session_means_ms=means,
                range_ms=max(means)-min(means) if means else None,
                budget_pass_all=bool(len(means)==3 and budget is not None and all(v<=budget for v in means))))
    save(ROOT/'results.json',dict(complete=final,rankings=rankings,stress=stress,stability=stability,time=time.time()))
    lines=['<!-- validation-extra-start -->','## Additional checks: ranking, stress and timing sessions','',
        'Status: '+('complete; not independently reviewed.' if final else 'running; saved after each measurement.'),'',
        '20 pairs/backbone with GMAC difference ≤1%, different station counts, and no workload reused within the pair set. 8 stress cases/backbone: K1/K2 × final tokens 3/32/64/128. The dataset was chosen before measurement, without using accuracy. The mid-budget schedules of the 4 proxies and 2 controls were timed in 3 sessions in total (session 0 is the main pass); 3 rounds per session. All at bs32, on one GPU, after all accuracy runs.','',
        '| Proxy | Pairs with distinguishable latency | Correctly ordered | Stress MAPE % |','|---|---:|---:|---:|']
    for model in ('frozen_t',*main.MODELS):
        a=[v for v in rankings if v['model']==model and v['eligible']];s=[v['error_pct'] for v in stress if v['model']==model]
        acc=f"{sum(v['correct'] for v in a)}/{len(a)}" if a else '—';err=f'{np.mean(s):.3f}' if s else '—'
        lines.append(f'| {model} | {len(a)} | {acc} | {err} |')
    lines+=['',f'Raw: `{ROOT.resolve()}`. `results.json` keeps all pairs, the negative cases and session means; latency differences smaller than noise are not counted as correct or incorrect. Three sessions on the same GPU type do not demonstrate transfer to other hardware.','<!-- validation-extra-end -->']
    if main.REPORT.exists():
        text=main.REPORT.read_text().split('<!-- validation-extra-start -->')[0].rstrip()
        main.REPORT.write_text(text+'\n\n'+'\n'.join(lines)+'\n')
    taxonomy=main.REPORT.parent/'REPORT_TAXONOMY.md';text=taxonomy.read_text()
    prefix=f'| [{main.REPORT.name}]';plan_prefix=f'| [{main.PLAN.name}]'
    status='Main pass and extra checks complete; not independently reviewed' if final else 'Main pass done; extra checks running'
    row=f'| [{main.REPORT.name}]({main.REPORT.name}) | 84 M3 schedules/7 backbones; extra checks on 140 near-GMAC pairs, 56 stress cases and 3 sessions of representative schedules. Rank32/bs32, historical data; old code unchanged. | {status} |'
    planrow=f'| [{main.PLAN.name}](../plan/{main.PLAN.name}) | 84 main schedules, near-GMAC pairs, stress and 3 sessions; adds code and saves after each step. | {status} | [{main.REPORT.name}]({main.REPORT.name}) |'
    taxonomy.write_text('\n'.join(row if l.startswith(prefix) else planrow if l.startswith(plan_prefix) else l for l in text.splitlines())+'\n')


def launch():
    import fcntl
    ROOT.mkdir(parents=True,exist_ok=True);lock=(ROOT/'scheduler.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    planning()
    while True:
        p=main.ROOT/'status.json'
        if p.exists() and load_json(p)['phase']=='completed':break
        pid=int((main.ROOT/'scheduler.pid').read_text())
        if not Path(f'/proc/{pid}').exists():
            save(ROOT/'status.json',dict(phase='main_stopped_before_completion',time=time.time()))
            raise RuntimeError('Main scheduler stopped before completion; saved progress is retained')
        time.sleep(10)
    for session in (1,2):
        for bb in main.CHOICES:
            report();measure_session(session,bb);report()
    assert len([v for v in load_json(ROOT/'results.json')['stability'] if v['sessions']==3])==42
    main.validate();report(True);save(ROOT/'status.json',dict(phase='completed',time=time.time()))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['launch','plan','report']);args=parser.parse_args()
    torch.set_num_threads(4);torch.manual_seed(151)
    if args.action=='launch':launch()
    elif args.action=='plan':planning()
    else:report()
