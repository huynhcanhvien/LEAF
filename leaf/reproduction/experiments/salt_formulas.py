"""Run all nine new proxy formulas through M3 DP, accuracy on two GPUs, timing last."""
import argparse
import csv
import fcntl
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import numpy as np
import torch
import leaf.evaluation.determinism
from leaf.evaluation.runner import load_json,save,config_hash,require_idle_gpu
from leaf.reproduction.runners.campaign_runner import CampaignRunner,build
from leaf.latency.timing import measure_latency
from leaf.evaluation.data import mcnemar,holm
from leaf.planner.configs import ForecastConfig
from leaf.planner.gmac import config_cost
from leaf.latency.salt import FORMULAS,SaltCost,plan_with_salt,features
from leaf.backbones import SPECS
from leaf.reproduction.experiments.gmac_planning import curves,sha,hash_file
from leaf.reproduction.experiments import salt_fit as compare

ROOT=compare.ROOT/'gpu';SOURCE=compare.SOURCE;PILOT=compare.PILOT
CHOICES={'deit_s':['DR2.3','DR2.7'],'augreg_s':['dense0.5','dense0.68'],'mae_b':['DR8.7','DR10.4']}


def immutable(path,value):
    value=json.loads(json.dumps(value))
    if path.exists():assert load_json(path)==value,path
    else:save(path,value)


def runner(bb,out=False):
    design=load_json(SOURCE/'design.json');r=CampaignRunner(bb,ROOT if out else SOURCE,design);r.spec=SPECS[bb]
    if out:
        r.teacher_cache=SOURCE.resolve()/'teachers'
        r.reuse_roots=[PILOT.resolve()/bb,SOURCE.resolve()/bb]+r.reuse_roots
    d=ForecastConfig(Path(design['backbones'][bb]['root'])/'groups.json',G=16,rank=32)
    return r,d


def setup():
    assert (compare.ROOT/'cpu_done.json').exists()
    inputs={str((compare.ROOT/'dataset.json').resolve()):hash_file(compare.ROOT/'dataset.json')}
    for bb in CHOICES:
        for formula in FORMULAS:
            p=compare.ROOT/'cv'/bb/f'{formula}.json';inputs[str(p.resolve())]=hash_file(p)
        for p in [PILOT/bb/'plans.json',SOURCE/bb/'frozen.json']:
            inputs[str(p.resolve())]=hash_file(p)
    immutable(ROOT/'design.json',dict(formulas=list(FORMULAS),choices=CHOICES,inputs=inputs,
        script_sha=hash_file(__file__),formula_sha=hash_file('leaf/latency/salt.py'),
        old_sources={str(p.resolve()):hash_file(p) for folder in ('planner','evaluation','models') for p in Path(folder).rglob('*.py') if p.name!='t_proxy_formulas.py'},
        rules='All 9 formulas, no performance gate; fixed M3, rank32/G16. CPU search -> confirm all -> freeze all -> val all -> V2 all -> bs32 timing last. GPU tasks ascending GMAC, one process per GPU, dynamic claims per config.',
        disclosure='Exploratory historical data, all formulas evaluated. Same-budget contrasts are not automatically iso-latency. No val/V2-based selection.'))


def validate():
    d=load_json(ROOT/'design.json');assert hash_file(__file__)==d['script_sha']
    assert hash_file('leaf/latency/salt.py')==d['formula_sha']
    for path,h in {**d['inputs'],**d['old_sources']}.items():assert hash_file(path)==h,path


def planning():
    setup();design=load_json(SOURCE/'design.json')
    for bb,bids in CHOICES.items():
        path=ROOT/bb/'plans.json'
        if path.exists():continue
        r,d=runner(bb);cs=curves(r,d,'M3');jobs={};history=load_json(SOURCE/bb/'frozen.json')['jobs'];pilot=load_json(PILOT/bb/'plans.json')['jobs']
        for formula in FORMULAS:
            model=load_json(compare.ROOT/'cv'/bb/f'{formula}.json')['full_model'];cost=SaltCost(model,r.spec,d)
            for bid in bids:
                budget=next(v for v in design['backbones'][bb]['budgets'] if v['id']==bid)
                start=time.monotonic();result=plan_with_salt(cs,r.spec,d,cost,budget['latency_ms'])
                name=f'{formula}/{bid}'
                if result.get('infeasible'):
                    jobs[name]=dict(infeasible=True,budget=budget,result=result)
                    print('INFEASIBLE',bb,name,flush=True)
                else:
                    assert 0<result['predicted_ms']<=budget['latency_ms']-.1+1e-8
                    jobs[name]=dict(result,budget=budget,gmac=config_cost(result['config'],r.spec),search_seconds=time.monotonic()-start)
                    print('PLANNED',bb,name,result['predicted_ms'],jobs[name]['gmac'],flush=True)
                save(ROOT/bb/'planning_progress.json',dict(jobs=jobs))
        for bid in bids:
            jobs['frozen_t/'+bid]=dict(pilot['t_M3/'+bid])
            jobs['gmac_M3/'+bid]=dict(history['M3/gmac/'+bid])
            jobs['paper_M0/'+bid]=dict(history['paper/latency/'+bid])
        immutable(path,dict(jobs=jobs,sha=sha(jobs)))
        del r;gc.collect()


def all_tasks():
    tasks={}
    for bb in CHOICES:
        doc=load_json(ROOT/bb/'plans.json');assert sha(doc['jobs'])==doc['sha']
        for name,row in doc['jobs'].items():
            if row.get('infeasible'):continue
            key=f"{bb}-{config_hash(row['config'])}"
            if key not in tasks:tasks[key]=dict(id=key,backbone=bb,config=row['config'],names=[],gmac=config_cost(row['config'],SPECS[bb]))
            tasks[key]['names'].append(name)
    return sorted(tasks.values(),key=lambda t:(t['gmac'],t['id']))


def profile(r,cfg,gpu):
    path=r.root/f'profile_gpu{gpu}.json'
    if path.exists():return
    model=r.inference_model(cfg);rows=[]
    for batch in (16,32):
        x=torch.randn(batch,3,224,224,device='cuda');torch.cuda.reset_peak_memory_stats()
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            for _ in range(2):y=model(x)
            torch.cuda.synchronize();start=time.monotonic()
            for _ in range(3):y=model(x)
            torch.cuda.synchronize()
        rows.append(dict(batch=batch,images_per_s=3*batch/(time.monotonic()-start),peak_bytes=torch.cuda.max_memory_allocated(),finite=bool(torch.isfinite(y).all())))
        del x,y
    assert all(v['finite'] for v in rows)
    assert rows[-1]['peak_bytes']<torch.cuda.get_device_properties(0).total_memory-2*2**30
    save(path,dict(rows=rows,chosen=32,reason='paired BF16 evaluation and historical bs32 teachers; sufficient memory reserve'))
    del model;gc.collect();torch.cuda.empty_cache()


def worker(split):
    validate();require_idle_gpu();gpu=os.environ['CUDA_VISIBLE_DEVICES'];runners={}
    tasks=load_json(ROOT/'tasks.json');claims=ROOT/'claims'/split;claims.mkdir(parents=True,exist_ok=True)
    for task in tasks:
        dest=ROOT/'completed'/split/f"{task['id']}.json"
        if dest.exists():continue
        claim=claims/task['id']
        try:fd=os.open(claim,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
        except FileExistsError:continue
        with os.fdopen(fd,'w') as f:f.write(str(os.getpid()))
        bb=task['backbone'];cfg=task['config']
        if bb not in runners:runners[bb]=runner(bb,True)[0]
        r=runners[bb]
        # Let the existing runner reuse exact-config arrays before any GPU work.
        candidates=[r.root]+r.reuse_roots
        has_cache=any((base/split/'runs'/f'{config_hash(cfg)}.json').exists() for base in candidates)
        require_idle_gpu()
        if not has_cache:profile(r,cfg,gpu)
        save(ROOT/f'worker_gpu{gpu}_status.json',dict(split=split,task=task['id'],names=task['names'],time=time.time()))
        arr=r.run(cfg,split,meta=dict(experiment='t_proxy_formulas',aliases=task['names']))
        assert arr is not None
        expected={'confirm':20018,'val':50000,'v2':10000}[split]
        assert all(arr[k].shape==(expected,) for k in ('kl','pred','correct')) and np.isfinite(arr['kl']).all()
        save(dest,dict(task=task['id'],backbone=bb,names=task['names'],config_sha=config_hash(cfg),gpu=gpu,
            n=expected,top1=100*float(arr['correct'].mean()),kl=float(arr['kl'].mean()),time=time.time()))
        print('DONE',split,task['id'],task['names'],flush=True)


def merge_metrics(split):
    rows={bb:{} for bb in CHOICES}
    for path in (ROOT/'completed'/split).glob('*.json'):
        row=load_json(path)
        for name in row['names']:rows[row['backbone']][name]=row
    for bb,metrics in rows.items():save(ROOT/bb/f'{split}_metrics.json',metrics)


def idle_both():
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
    assert not any(v.strip() and int(v.strip())!=os.getpid() for v in apps.splitlines()),apps


def timing():
    validate();assert (ROOT/'accuracy_done.json').exists();idle_both()
    for bb in CHOICES:
        if (ROOT/bb/'latency_done.json').exists():continue
        r,_=runner(bb,True);_=r.vit;directory=ROOT/bb/'latency'
        if directory.exists():directory.rename(ROOT/bb/f'latency_interrupted_{time.time_ns()}')
        jobs={k:v for k,v in load_json(ROOT/bb/'plans.json')['jobs'].items() if not v.get('infeasible')};seen={}
        for rnd in range(3):
            names=sorted(jobs);names=names if rnd%2==0 else names[::-1]
            for name in names:
                idle_both();cfg=jobs[name]['config'];key=(rnd,config_hash(cfg))
                if key in seen:measurement=dict(seen[key],alias=True)
                else:
                    model=build(r.vit,r.spec,cfg);measurement=measure_latency(model,32,iters=50,warmup=10)
                    seen[key]=measurement;del model;gc.collect();torch.cuda.empty_cache()
                save(directory/f"{name.replace('/','|')}#{rnd}.json",dict(name=name,round=rnd,config_sha=key[1],measurement=measurement))
                print('LATENCY',bb,rnd,name,measurement['ms_mean'],flush=True)
        save(ROOT/bb/'latency_done.json',dict(rounds=3,batch=32,time=time.time()))
        del r;gc.collect();torch.cuda.empty_cache()


def report(final=False):
    records=[];tests=[]
    for bb in CHOICES:
        if not (ROOT/bb/'plans.json').exists():continue
        jobs=load_json(ROOT/bb/'plans.json')['jobs']
        metrics={s:load_json(ROOT/bb/f'{s}_metrics.json') if (ROOT/bb/f'{s}_metrics.json').exists() else {} for s in ('confirm','val','v2')}
        for name,row in jobs.items():
            if row.get('infeasible'):continue
            samples=[]
            for rnd in range(3):
                path=ROOT/bb/'latency'/f"{name.replace('/','|')}#{rnd}.json"
                if path.exists():samples.append(load_json(path)['measurement']['ms_mean'])
            records.append(dict(backbone=bb,name=name,formula=name.split('/')[0],budget=name.split('/')[1],
                gmac=config_cost(row['config'],SPECS[bb]),predicted_ms=row.get('predicted_ms'),
                confirm_kl=metrics['confirm'].get(name,{}).get('kl'),valid=metrics['val'].get(name,{}).get('top1'),
                v2=metrics['v2'].get(name,{}).get('top1'),latency_ms=float(np.mean(samples)) if len(samples)==3 else None,
                latency_sd=float(np.std(samples,ddof=1)) if len(samples)==3 else None))
        if final:
            r,_=runner(bb,True)
            for name,row in jobs.items():
                if row.get('infeasible') or name.split('/')[0] not in FORMULAS:continue
                bid=name.split('/')[1]
                for ref in ('frozen_t/'+bid,'gmac_M3/'+bid,'paper_M0/'+bid):
                    for split in ('val','v2'):
                        a=r.cached(row['config'],split);b=r.cached(jobs[ref]['config'],split)
                        pv,wins,losses=mcnemar(a['correct'],b['correct'])
                        tests.append(dict(backbone=bb,name=name,reference=ref,split=split,p=pv,wins=wins,losses=losses,
                            delta_pp=100*float(a['correct'].mean()-b['correct'].mean())))
    if final:
        for split in ('val','v2'):holm([v for v in tests if v['split']==split])
    save(ROOT/'results.json',dict(complete=final,records=records,paired_tests=tests,time=time.time()))
    lines=['<!-- gpu-results-start -->','## Scheduling experiments with each formula','',
        'GPU status: '+('execution and audit complete; not independently reviewed.' if final else 'running; results are updated per configuration.')+' All formulas are kept; none is selected by confirm/valid. The same predicted budget does not necessarily mean the same actual latency.',
        '', '| Backbone | Formula / budget | GMAC | t ms | Confirm KL | Valid top1 | V2 top1 | Latency ms |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    fmt=lambda v:'—' if v is None else f'{v:.4f}'
    for row in records:lines.append('| '+' | '.join([row['backbone'],row['name']]+[fmt(row[k]) for k in ('gmac','predicted_ms','confirm_kl','valid','v2','latency_ms')])+' |')
    lines+=['',f'Raw GPU: `{ROOT.resolve()}`; `results.json`, per-image data for each run, and a separate log per GPU. Per-config claims keep the two GPUs from overwriting each other; aliases share the same owner. Confirm runs to completion, then the SHA is locked before valid/V2; latency at bs32 in three interleaved rounds, last of all.','<!-- gpu-results-end -->']
    path=compare.REPORT;text=path.read_text().split('<!-- gpu-results-start -->')[0].rstrip()
    if final:text=text.replace('CPU done; next, run M3 schedules with the formulas.','CPU and GPU execution complete; not independently reviewed.')
    path.write_text(text+'\n\n'+'\n'.join(lines)+'\n')
    taxonomy=path.parent/'REPORT_TAXONOMY.md';text=taxonomy.read_text();prefix=f'| [{path.name}]'
    replacement=f'| [{path.name}]({path.name}) | 9 t formulas, 248 configs/7 backbones; nested group CV and near-GMAC ranking. Experiments on 54 M3 schedules/3 backbones, confirm/valid/V2 on two GPUs, final timing at bs32; rank32 and historical data. | '+('Execution/audit complete; not independently reviewed' if final else 'CPU done; GPU running')+' |'
    taxonomy.write_text('\n'.join(replacement if line.startswith(prefix) else line for line in text.splitlines())+'\n')


def audit():
    validate();checks=0
    for task in all_tasks():
        for split,n in [('confirm',20018),('val',50000),('v2',10000)]:
            path=ROOT/task['backbone']/split/'runs'/config_hash(task['config']);meta=load_json(path.with_suffix('.json'));arr=dict(np.load(path.with_suffix('.npz')))
            assert meta['config']==task['config'] and meta['batch_size']==32 and meta['n']==n
            assert all(arr[k].shape==(n,) for k in ('kl','pred','correct')) and np.isfinite(arr['kl']).all();checks+=1
    save(ROOT/'audit.json',dict(complete=True,unique_arrays_checked=checks,sources_and_inputs_unchanged=True,time=time.time()))


def launch():
    ROOT.mkdir(parents=True,exist_ok=True);lock=(ROOT/'scheduler.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    planning();validate();tasks=all_tasks();immutable(ROOT/'tasks.json',tasks);report()
    for split in ('confirm','val','v2'):
        if (ROOT/f'{split}_done.json').exists():continue
        # No old workers may remain on restart; never steal claims from live workers.
        claims=ROOT/'claims'/split
        for claim in claims.glob('*'):
            pid=int(claim.read_text());assert not Path(f'/proc/{pid}').exists(),f'worker still active {pid}'
            claim.unlink()
        if split!='confirm':assert (ROOT/'freeze.json').exists()
        save(ROOT/'status.json',dict(phase=split,pid=os.getpid(),tasks=len(tasks),time=time.time()))
        workers=[]
        for gpu in (0,1):
            log=(ROOT/f'{split}_gpu{gpu}.log').open('a');env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONUNBUFFERED='1')
            workers.append(subprocess.Popen([sys.executable,__file__,'worker','--split',split],env=env,stdout=log,stderr=subprocess.STDOUT))
        while any(p.poll() is None for p in workers):
            time.sleep(10);merge_metrics(split);report()
        assert all(p.returncode==0 for p in workers),f'{split} worker failed'
        merge_metrics(split);assert len(list((ROOT/'completed'/split).glob('*.json')))==len(tasks)
        immutable(ROOT/f'{split}_done.json',dict(tasks=len(tasks)))
        if split=='confirm':immutable(ROOT/'freeze.json',dict(plans={b:load_json(ROOT/b/'plans.json')['sha'] for b in CHOICES},selection='all preplanned formulas and budgets; no accuracy-based selection'))
    immutable(ROOT/'accuracy_done.json',dict(tasks=len(tasks),splits=['confirm','val','v2']))
    save(ROOT/'status.json',dict(phase='latency',pid=os.getpid(),time=time.time()))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHONUNBUFFERED='1')
    with (ROOT/'latency.log').open('a') as log:
        proc=subprocess.Popen([sys.executable,__file__,'timing'],env=env,stdout=log,stderr=subprocess.STDOUT)
        while proc.poll() is None:time.sleep(10);report()
        assert proc.returncode==0,'timing failed'
    audit();report(True)
    rows=load_json(ROOT/'results.json')['records']
    with (ROOT/'all_arms.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    save(ROOT/'status.json',dict(phase='completed',time=time.time(),tasks=len(tasks)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['plan','launch','worker','timing','report']);parser.add_argument('--split',choices=['confirm','val','v2']);args=parser.parse_args()
    torch.set_num_threads(4);torch.manual_seed(151)
    if args.action=='plan':planning()
    elif args.action=='worker':worker(args.split)
    elif args.action=='timing':timing()
    elif args.action=='report':report()
    else:launch()
