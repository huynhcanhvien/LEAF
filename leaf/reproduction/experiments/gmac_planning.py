"""Additive seven-backbone GMAC-planned experiment; all timing is last."""
import argparse
from copy import deepcopy
from datetime import datetime
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import numpy as np
import torch
import leaf.evaluation.determinism
from leaf.evaluation.runner import load_json,save,config_hash,require_idle_gpu,per_image_kl
from leaf.evaluation.data import load_split,iterate,holm,mcnemar
from leaf.reproduction.runners.campaign_runner import CampaignRunner,build
from leaf.backbones import SPECS
from leaf.planner.configs import ForecastConfig,config_from_state
from leaf.planner.probes import probe_jobs,probe_state
from leaf.planner.surrogate import ProbeCurve
from leaf.planner.gmac import GmacPlanner,config_cost,dense_cost,make_budgets

ROOT=Path('results/gmac_planning')
ORIGINAL=Path('results/leaf_m0')
MERGE=Path('results/merge_variants/stage2/attempt2')
BACKBONES=('deit_b','mae_b','deit_t','deit_s','augreg_s','augreg_b','mae_l')
NEW_FILES=('baselines/tome_m3.py','leaf/reproduction/runners/campaign_runner.py','leaf/planner/gmac.py','leaf/reproduction/experiments/gmac_planning.py')

def sha(obj): return hashlib.sha256(json.dumps(obj,sort_keys=True).encode()).hexdigest()
def hash_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(2**20),b''): h.update(chunk)
    return h.hexdigest()
def seal(path,obj):
    row=dict(obj,sha256=sha(obj))
    if path.exists(): assert load_json(path)==row,f'refusing to overwrite seal {path}'
    else: save(path,row)
    return row
def sealed(path):
    row=load_json(path); digest=row.pop('sha256'); assert sha(row)==digest,path
    return dict(row,sha256=digest)
def release(): gc.collect(); torch.cuda.empty_cache()
def design(root): return sealed(root/'design.json')

def setup(root):
    if (root/'design.json').exists(): return design(root)
    rows={}; input_hashes={}
    for bb in BACKBONES:
        files=list((ORIGINAL/bb).glob('**/budgets.json')); assert len(files)==1,(bb,files)
        base=files[0].parent; meta=load_json(ORIGINAL/bb/'design.json')
        manifest=load_json(base/'paper_final/frozen.json')
        rows[bb]=dict(root=str(base),timm_id=meta['timm_id'],budgets=load_json(files[0])['budgets'],jobs=manifest['jobs'])
        for p in (files[0],base/'groups.json',base/'paper_final/frozen.json',ORIGINAL/bb/'design.json'):
            input_hashes[str(p)]=hash_file(p)
    cache=Path('data/cache').resolve(); cache_info={}
    for name in ('calib_bal','calib','confirm','val','v2'):
        for suffix in ('224_u8','labels'):
            p=cache/f'{name}_{suffix}.npy'; arr=np.load(p,mmap_mode='r')
            cache_info[str(p)]=dict(shape=list(arr.shape),dtype=str(arr.dtype),sha256=hash_file(p))
    originals=load_json(root/'original_sources.json')
    assert all(hash_file(p)==digest for p,digest in originals.items()),'old code changed'
    return seal(root/'design.json',dict(version=1,backbones=rows,original_root=str(ORIGINAL),merge_root=str(MERGE),data_cache=str(cache),data=cache_info,input_hashes=input_hashes,
        original_source_hashes=originals,new_source_hashes={p:hash_file(p) for p in NEW_FILES},
        rules=dict(planner='GMAC only; pure matrix/conv MAC + grouped-D MAC; scalar/normalization/sort costs separate',
            constraints='48 GMAC budgets across seven backbones; new DP-M0 and DP-M3 + same-schedule M3; retain all historical arms',
            accuracy_batch=32,latency_batches=[1,32,128,512],latency_rounds=3,latency_last=True,
            matching_gmac='nearest absolute cost within max(0.001 GMAC, 0.005 * arm GMAC); ties faster/lower GMAC',
            matching_latency='nearest mean bs32 latency within max(.05ms, 2*pooled sample SD); ties faster; never accuracy-based',
            holm='all eligible primary GMAC and direct operator contrasts across seven backbones, separately val/v2; latency secondary separate family',
            disclosure='Historical calibration/confirm/val/V2 and operator selection already viewed. Two duplicate calibration images remain in paper confirm (20018) to preserve historical pairing. Not pristine holdouts.',
            no_screening=True,old_code='additive only; hash every original source file at start and end')))

def validate(root):
    doc=design(root)
    assert all(hash_file(p)==h for p,h in doc['original_source_hashes'].items()),'old code was modified'
    assert all(hash_file(p)==h for p,h in doc['new_source_hashes'].items()),'locked new implementation changed'
    return doc

def context(root,bb):
    doc=design(root); r=CampaignRunner(bb,root,doc)
    r.spec=SPECS[bb]; d=ForecastConfig(Path(doc['backbones'][bb]['root'])/'groups.json',G=16,rank=32)
    return r,d

def profile(root,bb):
    """Memory/feasibility only: no latency/throughput measurement before the final stage."""
    path=root/bb/'profile.json'
    if path.exists(): return
    require_idle_gpu(); r,d=context(root,bb); _=r.vit
    old=design(root)['backbones'][bb]['jobs']
    configs=[dict(kind='dense')]
    for kind in ('tome','pitome','evit','mctf','ppt'):
        configs.append(next(v['config'] for v in old.values() if v['config']['kind']==kind))
    configs+=[dict(configs[1],kind='tome_m3'),dict(config_from_state(probe_state(1,189,1.,3.),r.spec,d),kind='pdm',merge=dict(mode='M3',kernel='triton_norm'))]
    rows=[]
    for cfg in configs:
        model=r.inference_model(cfg); x=torch.randn(32,3,224,224,device='cuda')
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16): y=model(x); repeat=model(x)
        torch.cuda.synchronize(); peak=torch.cuda.max_memory_allocated()
        rows.append(dict(kind=cfg['kind'],batch=32,peak_bytes=peak,finite=bool(torch.isfinite(y).all()),deterministic=torch.equal(y,repeat),reserve_bytes=torch.cuda.get_device_properties(0).total_memory-peak))
        if peak>torch.cuda.get_device_properties(0).total_memory-2*2**30: raise RuntimeError(f'bs32 insufficient memory reserve: {bb}/{cfg}')
        del model,x,y,repeat; release()
    # Same config, legacy factory unchanged and additive dispatch agrees exactly.
    from leaf.evaluation.runner import build as old_build
    cfg=next(v['config'] for v in old.values() if v['config']['kind']=='pd')
    a=old_build(r.vit,r.spec,cfg); b=build(r.vit,r.spec,cfg); x=torch.randn(2,3,224,224,device='cuda')
    with torch.no_grad(): parity=torch.equal(a(x),b(x))
    save(path,dict(rows=rows,legacy_dispatch_bitwise=parity,passed=parity and all(v['finite'] and v['deterministic'] for v in rows),note='No performance screening; memory feasibility only. Timing deferred until all accuracy completes.'))

def probes(root,bb):
    require_idle_gpu(); profile(root,bb); r,d=context(root,bb); r.teachers()
    jobs=list(probe_jobs(r.spec)); done=0
    for l,n,theta,k in jobs:
        require_idle_gpu(); cfg=config_from_state(probe_state(l,n,theta,k),r.spec,d)
        for mode in (('M0',) if theta==0 else ('M0','M3')):
            c=cfg if mode=='M0' else dict(cfg,kind='pdm',merge=dict(mode='M3',kernel='triton_norm'))
            arr=r.run(c,'half',meta=dict(phase='probe',probe=[l,n,theta,k],mode=mode))
            if arr is None: raise RuntimeError(f'invalid probe {bb}/{mode}/{l}/{n}/{theta}/{k}')
        done+=1; save(r.root/'probe_progress.json',dict(done=done,total=len(jobs),last=[l,n,theta,k]))
    save(r.root/'probes_done.json',dict(done=done))

def curves(r,d,mode):
    rows={}
    for l,n,theta,k in probe_jobs(r.spec):
        cfg=config_from_state(probe_state(l,n,theta,k),r.spec,d)
        if mode=='M3' and theta!=0: cfg=dict(cfg,kind='pdm',merge=dict(mode='M3',kernel='triton_norm'))
        arr=r.cached(cfg,'half')
        if arr is None: raise RuntimeError(f'missing curve: {r.name}/{mode}/{l}/{n}/{theta}/{k}')
        rows.setdefault((theta,k),{}).setdefault(l,{})[r.spec.n_tokens-n]=float(arr['kl'].mean())
    return {key:ProbeCurve(value,r.spec.n_tokens) for key,value in rows.items()}

def prepare(root,bb):
    require_idle_gpu(); r,d=context(root,bb); doc=design(root); source=doc['backbones'][bb]
    path=r.root/'frozen.json'
    if path.exists(): sealed(path); return
    jobs={}
    for name,row in source['jobs'].items():
        prefix='paper/' if name.startswith(('latency/','latency_gmac/')) else 'baseline/'
        jobs[prefix+name]=deepcopy(row)
        cfg=row['config']
        if cfg['kind']=='tome': jobs['tome_m3/'+name.split('/',1)[-1]]=dict(config=dict(cfg,kind='tome_m3'),meta=dict(family='tome_m3'))
    budgets=make_budgets(r.spec,source['budgets'],source['jobs']); save(r.root/'gmac_budgets.json',budgets)
    for mode in ('M0','M3'):
        planner=GmacPlanner(r.spec,d,curves(r,d,mode),mode)
        for budget in budgets:
            outfile=r.root/'search'/mode/f"{budget['id']}.json"
            if outfile.exists(): result=load_json(outfile)
            else: result=planner.plan(budget['gmac']); save(outfile,result)
            if result.get('infeasible'): raise RuntimeError(f'infeasible GMAC budget {bb}/{mode}/{budget}')
            assert result['gmac']<=budget['gmac']+1e-8
            name=f"{mode}/gmac/{budget['id']}"
            jobs[name]=dict(config=result['config'],state=result['state'],meta=dict(budget=budget,mode=mode,planner='gmac',gmac=result['gmac']))
            if mode=='M0': jobs[f"swap_M3/gmac/{budget['id']}"]=dict(config=dict(result['config'],kind='pdm',merge=dict(mode='M3',kernel='triton_norm')),meta=dict(budget=budget,mode='swap_M3'))
    # Retain previously reported M3 schedules as explicit historical references.
    previous=MERGE/bb/'stage2/frozen.json'
    if previous.exists():
        for name,row in load_json(previous)['jobs'].items():
            if name.startswith('M3/'): jobs['historical/'+name]=row
    seal(r.root/'arms.json',dict(jobs=jobs))
    for name,row in sorted(jobs.items()):
        require_idle_gpu()
        if r.run(row['config'],'confirm',meta=dict(arm=name)) is None: raise RuntimeError(f'invalid arm {name}')
        save(r.root/'prepare_progress.json',dict(last_confirm=name,total=len(jobs)))
    seal(path,dict(jobs=jobs,design_sha=doc['sha256'],time=datetime.now().astimezone().isoformat(),gmac_only=True))

def macs(root,bb):
    require_idle_gpu(); r,_=context(root,bb); frozen=sealed(r.root/'frozen.json'); results={}
    path=r.root/'gmacs.json'
    if path.exists(): results=load_json(path)
    for name,row in sorted(frozen['jobs'].items()):
        if name in results: continue
        cfg=row['config']
        if cfg['kind'] in ('pd','pdm'):
            full=__import__('leaf.planner.schedule_caps',fromlist=['full_cost']).full_cost(cfg['prune_keep'],cfg['merge_keep'],cfg.get('k'),r.spec.n_tokens,r.spec.dim,{k:len(set(v)) for k,v in cfg.get('groups',{}).items()},cfg.get('rank'))
            results[name]=dict(gmac=config_cost(cfg,r.spec),scalar_gops_estimate=full['station_gops'],scope='matrix/conv + grouped D; excludes sorting/scalar/norm ops')
        elif cfg['kind']=='dense': results[name]=dict(gmac=dense_cost(r.spec),scope='matrix/conv')
        else:
            from leaf.latency.timing import gmacs
            _=r.vit; model=build(r.vit,r.spec,cfg)
            results[name]=dict(gmac=gmacs(model),scope='torch FlopCounter matrix/conv/SDPA; actual adapter operations counted')
            del model; release()
        save(path,results)

def accuracy(root,bb,split):
    require_idle_gpu(); r,_=context(root,bb); frozen=sealed(r.root/'frozen.json'); completed={}
    path=r.root/f'{split}_metrics.json'
    if path.exists(): completed=load_json(path)
    for name,row in sorted(frozen['jobs'].items()):
        if name in completed: continue
        require_idle_gpu(); arr=r.run(row['config'],split,meta=dict(arm=name,frozen_sha=frozen['sha256']))
        if arr is None: raise RuntimeError(f'nonfinite accuracy {bb}/{split}/{name}')
        completed[name]=dict(top1=100*float(arr['correct'].mean()),kl=float(arr['kl'].mean()),n=len(arr['kl']),config_sha=sha(row['config']))
        save(path,completed)
    seal(r.root/f'{split}_done.json',dict(n_arms=len(completed),frozen_sha=frozen['sha256']))

def idle_both():
    import subprocess
    require_idle_gpu()
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
    if any(int(v)!=os.getpid() for v in apps.splitlines() if v.strip()): raise RuntimeError('latency requires both GPUs otherwise idle')

def latency(root,bb):
    # The scheduler calls this only after every backbone has completed valid/V2.
    for backbone in BACKBONES:
        assert (root/backbone/'val_done.json').exists() and (root/backbone/'v2_done.json').exists(),'all accuracy must finish before timing'
    from leaf.latency.timing import measure_latency
    idle_both(); r,_=context(root,bb); frozen=sealed(r.root/'frozen.json')
    final=r.root/'latency'
    if (r.root/'latency_done.json').exists(): return
    # Restart an interrupted session from the beginning; preserve previous samples.
    if final.exists(): final.rename(r.root/f'latency_interrupted_{time.time_ns()}')
    names=sorted(frozen['jobs']); seen={}
    for round_id in range(3):
        for name in (names if round_id%2==0 else names[::-1]):
            cfg=frozen['jobs'][name]['config']; key=config_hash(cfg); path=final/f"{name.replace('/','|')}#{round_id}.json"
            alias=seen.get((round_id,key))
            if alias:
                row=load_json(alias); save(path,dict(row,name=name,equal_to=str(alias))); continue
            idle_both(); _=r.vit; model=build(r.vit,r.spec,cfg); measurements={}
            for batch in (1,32,128,512):
                try: measurements[str(batch)]=measure_latency(model,batch,iters=50 if batch<=32 else 10,warmup=10 if batch<=32 else 3)
                except torch.cuda.OutOfMemoryError: measurements[str(batch)]='OOM'
                release()
            save(path,dict(name=name,round=round_id,config_sha=sha(cfg),measurements=measurements,frozen_sha=frozen['sha256']))
            seen[round_id,key]=str(path); del model; release()
    seal(r.root/'latency_done.json',dict(n_arms=len(names),rounds=3,batches=[1,32,128,512],frozen_sha=frozen['sha256']))

def match(costs,arm,refs,tolerance):
    candidates=[name for name in refs if abs(costs[name]-costs[arm])<=tolerance+1e-12]
    return min(candidates,key=lambda name:(abs(costs[name]-costs[arm]),costs[name],name)) if candidates else None

def compare(root):
    doc=validate(root); tests={s:{a:[] for a in ('gmac','latency')} for s in ('val','v2')}; frontiers=[]; absolute=[]
    for bb in BACKBONES:
        r,_=context(root,bb); f=sealed(r.root/'frozen.json'); jobs=f['jobs']; costs=load_json(r.root/'gmacs.json')
        lat={}
        for p in (r.root/'latency').glob('*.json'):
            row=load_json(p); v=row['measurements']['32']
            if v!='OOM': lat.setdefault(row['name'],{})[row['round']]=v['ms_mean']
        latcost={k:float(np.mean(list(v.values()))) for k,v in lat.items() if len(v)==3}
        pooled=float(np.sqrt(np.mean([np.var(list(v.values()),ddof=1) for v in lat.values() if len(v)==3])))
        gmac={k:v['gmac'] for k,v in costs.items()}; tau=max(.05,2*pooled)
        def family(name,cfg):
            if name.startswith('M0/gmac/'): return 'M0_gmac'
            if name.startswith('paper/'): return 'paper_M0'
            if name.startswith('historical/'): return 'historical_M3'
            if name.startswith('baseline/DR'): return 'DiffRate'
            return cfg['kind'] if cfg['kind'] in ('tome','tome_m3','evit','pitome','ppt','mctf') else None
        families={}
        for name,row in jobs.items():
            fam=family(name,row['config'])
            if fam: families.setdefault(fam,[]).append(name)
        matches=[]
        for arm in sorted(n for n in jobs if n.startswith('M3/gmac/')):
            for axis,cs,tolerance in [('gmac',gmac,max(.001,.005*gmac[arm])),('latency',latcost,tau)]:
                if arm not in cs: continue
                for fam,refs in families.items():
                    available=[n for n in refs if n in cs]
                    ref=match(cs,arm,available,tolerance)
                    matches.append(dict(backbone=bb,axis=axis,arm=arm,reference=ref,family=fam,eligible=ref is not None,cost=cs[arm],ref_cost=cs.get(ref),tolerance=tolerance))
            # Same-schedule comparison isolates M3 independent of changed GMAC planning.
            m0=arm.replace('M3/','M0/',1); swap=arm.replace('M3/','swap_M3/',1)
            matches.append(dict(backbone=bb,axis='gmac',arm=swap,reference=m0,family='operator_only',eligible=True,cost=gmac[swap],ref_cost=gmac[m0],tolerance=0.))
        seal(r.root/'pairs.json',dict(rows=matches,frozen_sha=f['sha256'],accuracy_not_used_for_matching=True,tau_ms=tau))
        for split in ('val','v2'):
            metrics=load_json(r.root/f'{split}_metrics.json')
            for name,row in jobs.items(): absolute.append(dict(backbone=bb,split=split,name=name,gmac=gmac[name],latency_ms=latcost.get(name),**metrics[name]))
            for pair in matches:
                if not pair['eligible']: continue
                a=r.cached(jobs[pair['arm']]['config'],split); b=r.cached(jobs[pair['reference']]['config'],split)
                p,w,l=mcnemar(a['correct'].astype(bool),b['correct'].astype(bool))
                faster=False
                if pair['arm'] in lat and pair['reference'] in lat:
                    faster=all(lat[pair['arm']][i]<=lat[pair['reference']][i]-tau for i in range(3))
                tests[split][pair['axis']].append(dict(pair,split=split,p=p,wins=w,losses=l,delta=100*float(a['correct'].mean()-b['correct'].mean()),faster_by_tau_all_rounds=faster))
            # Envelope is descriptive, has no per-image McNemar p-value.
            for axis,cs in [('gmac',gmac),('latency',latcost)]:
                for fam,refs in families.items():
                    points={}
                    for n in refs:
                        if n in cs: points[cs[n]]=max(points.get(cs[n],-np.inf),metrics[n]['top1'])
                    xs=sorted(points)
                    if len(xs)<2: continue
                    ys=np.maximum.accumulate([points[x] for x in xs])
                    for arm in sorted(n for n in jobs if n.startswith('M3/gmac/')):
                        if arm not in cs or not xs[0]<=cs[arm]<=xs[-1]: continue
                        frontier=float(np.interp(cs[arm],xs,ys))
                        frontiers.append(dict(backbone=bb,axis=axis,split=split,arm=arm,family=fam,delta=metrics[arm]['top1']-frontier,interpolated_reference=frontier,descriptive_only=True))
    for split in tests:
        for axis in tests[split]: holm(tests[split][axis])
    save(root/'comparisons.json',tests); save(root/'frontiers.json',frontiers); save(root/'absolute.json',absolute)

def audit(root):
    doc=validate(root); checked={}; rows=load_json(root/'comparisons.json')
    for bb in BACKBONES:
        r,_=context(root,bb); f=sealed(r.root/'frozen.json'); count=0
        for split,expected in [('confirm',20018),('val',50000),('v2',10000)]:
            for name,row in f['jobs'].items():
                arr=r.cached(row['config'],split); meta=load_json(r.path(row['config'],split).with_suffix('.json'))
                assert meta['config']==row['config'] and meta['batch_size']==32 and meta['n']==expected
                assert all(arr[k].shape==(expected,) for k in ('kl','pred','correct')) and np.isfinite(arr['kl']).all()
                count+=1
        assert len(list((r.root/'latency').glob('*.json')))==len(f['jobs'])*3
        assert len(load_json(r.root/'gmacs.json'))==len(f['jobs'])
        checked[bb]=dict(arms=len(f['jobs']),arrays=count,profile_passed=load_json(r.root/'profile.json')['passed'])
    # Independently recompute the stored paired tests and the global Holm families.
    for split,families in rows.items():
        for axis,records in families.items():
            fresh=[]
            for row in records:
                r,_=context(root,row['backbone']); jobs=sealed(r.root/'frozen.json')['jobs']
                a=r.cached(jobs[row['arm']]['config'],split); b=r.cached(jobs[row['reference']]['config'],split)
                p,w,l=mcnemar(a['correct'].astype(bool),b['correct'].astype(bool)); assert abs(p-row['p'])<1e-12 and (w,l)==(row['wins'],row['losses'])
                fresh.append(dict(p=p,stored=row['holm']))
            holm(fresh); assert all(abs(v['holm']-v['stored'])<1e-12 for v in fresh)
    save(root/'audit.json',dict(execution_complete=True,checked=checked,old_sources_unchanged=True,new_sources_unchanged=True,statistics_recomputed=True,finished=datetime.now().astimezone().isoformat()))

def main():
    p=argparse.ArgumentParser(); p.add_argument('action',choices=('setup','profile','probes','prepare','macs','accuracy','latency','compare','audit')); p.add_argument('--root',type=Path,default=ROOT); p.add_argument('--backbone',choices=BACKBONES); p.add_argument('--split',choices=('val','v2'),default='val')
    a=p.parse_args(); a.root.mkdir(parents=True,exist_ok=True)
    if a.action=='setup': setup(a.root); return
    validate(a.root)
    if a.action=='compare': compare(a.root)
    elif a.action=='audit': audit(a.root)
    else:
        torch.set_num_threads(4); torch.manual_seed(151)
        if a.action=='accuracy': accuracy(a.root,a.backbone,a.split)
        else: globals()[a.action](a.root,a.backbone)

if __name__=='__main__': main()
