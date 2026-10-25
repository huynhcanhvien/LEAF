"""Follow-up queue, waiting for the predecessor's successful final audit."""
import argparse
from copy import deepcopy
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import numpy as np
import torch
import leaf.evaluation.determinism
from leaf.evaluation.runner import save,load_json,config_hash,require_idle_gpu
from leaf.latency.timing import inference_copy,measure_latency
from leaf.evaluation.data import load_split,mcnemar,holm
from leaf.backbones import SPECS
from leaf.reproduction.ablation_models.ablation_vit import ExtraViT
from leaf.planner.configs import ForecastConfig,config_from_state
from leaf.planner.probes import probe_jobs,probe_state
from leaf.planner.surrogate import ProbeCurve
from leaf.planner.plan import plan_latency,config_details,operator_config
from leaf.reproduction.experiments import low_budget_and_factorial as parent
from leaf.reproduction.experiments.gmac_planning import sha,hash_file

REPO=Path(__file__).resolve().parents[3]
ROOT=REPO/'results/horizon_norm_seed_ablation'
PREVIOUS=parent.ROOT
BBS=('deit_b','mae_b','augreg_s');LEVELS=('low90','low70');SEEDS=(251,351)
VARIANTS=('future1','future3','shuffle','shuffle_size','identity','mean_norm','rms_norm')
REPORT=REPO/'results/reports/latency-d-m3-extra-ablation-2026-10-18.md'
STATE=REPO/'results/notes/state/latency-d-m3-extra-ablation-2026-10-18-progress.md'
SPLITS=dict(confirm=20018,val=50000,v2=10000)
NEW=('leaf/reproduction/experiments/horizon_norm_seed_ablation.py','leaf/reproduction/ablation_models/ablation_vit.py',
     'leaf/reproduction/ablation_models/norm_target_merge_kernel.py','tests/test_horizon_norm_ablation.py','leaf/reproduction/experiments/horizon_norm_seed_analysis.py')


def immutable(path,obj):parent.immutable(path,obj)


def build(vit,spec,cfg):
    return ExtraViT(vit,cfg,spec.pool) if cfg['kind']=='extra_pdm' else parent.build(vit,spec,cfg)


class ExtraRunner(parent.LowRunner):
    def inference_model(self,cfg):
        _=self.vit
        if getattr(self,'_cast',None) is None:self._cast=inference_copy(self.vit)
        model=build(self.vit,self.spec,cfg)
        if cfg['kind']=='dense':return self._cast
        model.vit=self._cast
        for mod in model.modules():
            if mod is not self._cast and hasattr(mod,'prepare_inference'):mod.prepare_inference(torch.bfloat16)
        return model.eval()
    def data(self,split):
        if split!='half' or getattr(self,'seed',None) is None:return super().data(split)
        if split not in self._data:
            _=self.vit
            manifest=load_json(ROOT/'calibration'/f'{self.seed}.json');idx=np.array(manifest['indices'])
            images,labels=load_split('calib_bal',self.data_cache)
            teacher=np.load(self.teacher_cache/f'calib_bal_{self.name}_bs32_logits.npy',mmap_mode='r')
            self._data[split]=images[idx],labels[idx],torch.from_numpy(np.array(teacher[idx],dtype=np.float32))
        return self._data[split]


def context(bb,seed=None):
    design=load_json(ROOT/'design.json');r=ExtraRunner(bb,ROOT,design);r.spec=SPECS[bb]
    r.teacher_cache=parent.SOURCE.resolve()/'teachers';r.seed=seed
    # Only prior configs (same source snapshot, teachers, batch, data) can be reused.
    r.reuse_roots=[PREVIOUS/bb] if seed is None else []
    if seed is not None:r.root=ROOT/bb/f'probe_seed{seed}'
    d=ForecastConfig(Path(design['backbones'][bb]['root'])/'groups.json',G=16,rank=32)
    return r,d


def check():
    d=load_json(ROOT/'design.json')
    assert sha({k:v for k,v in d.items() if k!='sha256'})==d['sha256']
    for p,h in d['source_hashes'].items():assert hash_file(REPO/p)==h,p
    for p,h in d['input_hashes'].items():assert hash_file(p)==h,p
    parent.check_sources()


def setup():
    if (ROOT/'design.json').exists():check();return
    previous=load_json(PREVIOUS/'design.json');parent.check_sources()
    paths=[PREVIOUS/'design.json',PREVIOUS/'plans_sealed.json',PREVIOUS/'audit.json']
    paths += [PREVIOUS/bb/'plans.json' for bb in BBS]
    paths += [PREVIOUS/bb/'checkpoint_fingerprint.json' for bb in BBS]
    paths += [REPO/'results/notes/plan/latency-d-m3-extra-ablation-plan-2026-10-18.md']
    source={str(p.relative_to(REPO)):hash_file(p) for folder in ('models','planner','evaluation','baselines','scripts','tests') for p in (REPO/folder).rglob('*.py')}
    for p,h in source.items():
        dst=ROOT/'source_snapshot'/p;dst.parent.mkdir(parents=True,exist_ok=True)
        if not dst.exists():shutil.copy2(REPO/p,dst)
        assert hash_file(dst)==h
    labels=np.load(REPO/'data/cache/calib_bal_labels.npy');manifests={}
    content={}
    for split in ('calib_bal','confirm','calib'):
        images=np.load(REPO/f'data/cache/{split}_224_u8.npy',mmap_mode='r')
        content[split]=[hashlib.sha256(image.tobytes()).hexdigest() for image in images]
    confirm_set=set(content['confirm']);group_pool=set(content['calib'])
    for seed in SEEDS:
        rng=np.random.default_rng(seed);indices=[]
        for label in range(1000):
            candidates=np.flatnonzero(labels==label);assert len(candidates)>=5
            indices+=list(map(int,rng.choice(candidates,5,replace=False)))
        indices=sorted(indices);assert len(indices)==5000 and len(set(indices))==5000
        m=dict(seed=seed,indices=indices,labels=labels[indices].tolist(),pool='calib_bal',
               pool_sha=previous['data'][str(REPO/'data/cache/calib_bal_224_u8.npy')]['sha256'],
               overlap_original_half=sum(i%2==0 for i in indices),
               confirm_overlap_indices=[i for i in indices if content['calib_bal'][i] in confirm_set],
               grouping_pool_overlap_indices=[i for i in indices if content['calib_bal'][i] in group_pool],
               note='grouping pool is calib10k, not an assertion about the exact grouping1024 subset')
        m['sha256']=sha(m);immutable(ROOT/'calibration'/f'{seed}.json',m);manifests[str(seed)]=m['sha256']
    immutable(ROOT/'calibration/overlap_audit.json',dict(
        seed_pair_overlap=len(set(load_json(ROOT/'calibration/251.json')['indices'])&set(load_json(ROOT/'calibration/351.json')['indices'])),
        source_content_hashes=content,rule='SHA256 of raw uint8 image bytes; row manifests preserve class-balanced selection'))
    data=dict(previous_design_sha=previous['sha256'],backbones=previous['backbones'],
              data_cache=previous['data_cache'],original_root=previous['original_root'],merge_root=previous['merge_root'],
              source_hashes=source,input_hashes={str(p):hash_file(p) for p in paths},calibration_sha=manifests,
              selection=dict(backbones=BBS,levels=LEVELS,seeds=SEEDS,variants=VARIANTS,shuffle_seed=1810),
              rules='54 added named final configs + read-only references; fixed paths except seed replans; two GPU accuracy; timing last GPU0 alone; no accuracy-based selection',
              disclosure='Historical splits viewed; calibration seed draws overlap in fixed train pool; inherited confirm overlap audited, not a pristine holdout')
    data['sha256']=sha(data);immutable(ROOT/'design.json',data)
    immutable(ROOT/'probe_tasks.json',[dict(id=f'{bb}-seed{seed}',backbone=bb,seed=seed) for bb in BBS for seed in SEEDS])


def profile(r,cfg):
    dest=r.root/'profiles'/f'{config_hash(cfg)}.json'
    if dest.exists():assert load_json(dest)['passed'];return
    model=r.inference_model(cfg);rows=[]
    checkpoint=ROOT/r.name/'checkpoint_loaded.json'
    if not checkpoint.exists():
        h=hashlib.sha256()
        for name,value in r.vit.state_dict().items():
            value=value.detach().cpu().contiguous()
            h.update(json.dumps([name,list(value.shape),str(value.dtype)]).encode())
            h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        expected=load_json(PREVIOUS/r.name/'checkpoint_fingerprint.json')['state_dict_sha256']
        assert h.hexdigest()==expected,'loaded checkpoint differs from predecessor'
        immutable(checkpoint,dict(state_dict_sha256=h.hexdigest(),predecessor=str(PREVIOUS/r.name/'checkpoint_fingerprint.json')))
    for batch in (16,32,64):
        try:
            x=torch.randn(batch,3,224,224,device='cuda');torch.cuda.reset_peak_memory_stats()
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                y=model(x);torch.cuda.synchronize();start=time.monotonic()
                for _ in range(3):y=model(x)
                torch.cuda.synchronize();seconds=time.monotonic()-start;repeat=model(x)
            rows.append(dict(batch=batch,images_per_s=3*batch/seconds,peak_bytes=torch.cuda.max_memory_allocated(),
                        reserve_bytes=torch.cuda.get_device_properties(0).total_memory-torch.cuda.max_memory_allocated(),
                        finite=bool(torch.isfinite(y).all()),deterministic=torch.equal(y,repeat)))
            del x,y,repeat
        except torch.cuda.OutOfMemoryError:
            rows.append(dict(batch=batch,status='OOM'));gc.collect();torch.cuda.empty_cache()
            if batch<=32:raise
    row=next(r for r in rows if r['batch']==32)
    assert row['finite'] and row['deterministic'] and row['reserve_bytes']>=2*2**30,row
    save(dest,dict(config=cfg,rows=rows,chosen=32,passed=True,reason='bs32 paired protocol; actual-workload profile with memory reserve'))
    del model;gc.collect();torch.cuda.empty_cache()


def gpu_gate():
    dest=ROOT/f"gpu_gate_{os.environ['CUDA_VISIBLE_DEVICES']}.json"
    if dest.exists():assert load_json(dest)['passed'];return
    require_idle_gpu();sys.path.insert(0,str(REPO/'tests'))
    from test_extra_ablation import gpu
    rows=gpu();save(dest,dict(passed=True,rows=rows,time=time.time(),gpu=os.environ['CUDA_VISIBLE_DEVICES']))


def claim(task,stage):
    path=ROOT/'claims'/stage/task;path.parent.mkdir(parents=True,exist_ok=True)
    f=path.open('a+')
    try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:f.close();return None
    return f


def probe_worker():
    check();require_idle_gpu();gpu_gate()
    for task in load_json(ROOT/'probe_tasks.json'):
        done=ROOT/'completed/probes'/f"{task['id']}.json"
        if done.exists():continue
        handle=claim(task['id'],'probes')
        if handle is None:continue
        try:
            if done.exists():continue
            bb,seed=task['backbone'],task['seed'];r,d=context(bb,seed);jobs=list(probe_jobs(r.spec))
            c=operator_config(config_from_state(probe_state(1,189,1.,3.),r.spec,d),'M3');profile(r,c)
            for i,(l,n,th,k) in enumerate(jobs):
                require_idle_gpu();cfg=config_from_state(probe_state(l,n,th,k),r.spec,d)
                if th!=0:cfg=operator_config(cfg,'M3')
                arr=r.run(cfg,'half',meta=dict(seed=seed,calibration_sha=load_json(ROOT/'design.json')['calibration_sha'][str(seed)]))
                assert arr is not None and len(arr['kl'])==5000
                save(ROOT/f"worker_gpu{os.environ['CUDA_VISIBLE_DEVICES']}.json",dict(stage='probes',task=task,done=i+1,total=len(jobs),pid=os.getpid(),time=time.time()))
            save(done,dict(task=task,probes=len(jobs),time=time.time()));del r;gc.collect();torch.cuda.empty_cache()
        finally:handle.close()


def seed_curves(bb,seed,d):
    r,_=context(bb,seed);rows={};audit=[]
    for l,n,th,k in probe_jobs(r.spec):
        cfg=config_from_state(probe_state(l,n,th,k),r.spec,d)
        if th!=0:cfg=operator_config(cfg,'M3')
        prefix=r.path(cfg,'half');meta=load_json(prefix.with_suffix('.json'))
        assert meta['calibration_sha']==load_json(ROOT/'design.json')['calibration_sha'][str(seed)]
        assert meta['batch_size']==32 and meta['n']==5000 and meta['config']==cfg
        with np.load(prefix.with_suffix('.npz')) as a:value=float(a['kl'].mean());assert np.isfinite(a['kl']).all()
        rows.setdefault((th,k),{}).setdefault(l,{})[197-n]=value
        audit.append(dict(block=l,count=n,theta=th,kappa=k,kl=value,arrays=str(prefix.with_suffix('.npz')),
                          arrays_sha=hash_file(prefix.with_suffix('.npz')),metadata_sha=hash_file(prefix.with_suffix('.json'))))
    immutable(ROOT/bb/f'probe_means_seed{seed}.json',audit)
    return {key:ProbeCurve(row,197) for key,row in rows.items()}


def planning():
    check();jobs_all=[]
    for bb in BBS:
        dest=ROOT/bb/'plans.json'
        if dest.exists():jobs_all.extend(load_json(dest)['jobs'].values());continue
        _,d=context(bb);jobs={};old=load_json(PREVIOUS/bb/'plans.json')['jobs'];coeff=load_json(PREVIOUS/'design.json')['models'][bb]['main']
        curves={seed:seed_curves(bb,seed,d) for seed in SEEDS}
        for level in LEVELS:
            base=old[f'full/latency/{level}'];cfg=base['config'];budget=base['budget']
            jobs[f'reference/{level}/full']=dict(config=cfg,group='reference',level=level,budget=budget)
            for arm,label in (('m3_p','P_only'),('m0_pd','M0_PD')):
                jobs[f'reference/{level}/{label}']=dict(config=old[f'fixed/{level}/{arm}']['config'],group='reference',level=level,budget=budget)
            for variant in VARIANTS:
                c=deepcopy(cfg);c['kind']='extra_pdm';c['ablation']=dict(variant=variant,seed=1810)
                jobs[f'{variant}/{level}']=dict(config=c,group='norm' if variant in ('mean_norm','rms_norm') else 'D',level=level,budget=budget,
                    base_state=base['state'],details=config_details(cfg,SPECS[bb]),fixed_path=True)
            for seed in SEEDS:
                result=plan_latency(curves[seed],SPECS[bb],d,coeff,budget['latency_ms'])
                jobs[f'calibration/{level}/seed{seed}']=dict(result,group='calibration',level=level,budget=budget,seed=seed,
                    calibration_sha=load_json(ROOT/'design.json')['calibration_sha'][str(seed)])
        for name,row in jobs.items():
            if 'config' in row:row['config_sha']=config_hash(row['config'])
        immutable(dest,dict(jobs=jobs,sha=sha(jobs)))
        immutable(REPO/f'schedules/latency_d_m3_extra_ablation/2026-10-18/{bb}.json',dict(backbone=bb,jobs=jobs,sha=sha(jobs),design_sha=load_json(ROOT/'design.json')['sha256']))
    grouped={}
    for bb in BBS:
        for name,row in load_json(ROOT/bb/'plans.json')['jobs'].items():
            if 'config' not in row:continue
            key=bb+'-'+config_hash(row['config'])
            grouped.setdefault(key,dict(id=key,backbone=bb,config=row['config'],names=[]))['names'].append(name)
    immutable(ROOT/'tasks.json',sorted(grouped.values(),key=lambda t:(BBS.index(t['backbone']),t['id'])))
    immutable(ROOT/'plans_sealed.json',dict(tasks_sha=hash_file(ROOT/'tasks.json'),plans={bb:hash_file(ROOT/bb/'plans.json') for bb in BBS}))


def worker(split):
    check();require_idle_gpu();gpu_gate();r=None;bb_last=None
    for task in load_json(ROOT/'tasks.json'):
        done=ROOT/'completed'/split/f"{task['id']}.json"
        if done.exists():continue
        handle=claim(task['id'],split)
        if handle is None:continue
        try:
            if done.exists():continue
            bb,cfg=task['backbone'],task['config']
            if bb!=bb_last:r=None;gc.collect();torch.cuda.empty_cache();r,_=context(bb);bb_last=bb
            require_idle_gpu();profile(r,cfg)
            save(ROOT/f"worker_gpu{os.environ['CUDA_VISIBLE_DEVICES']}.json",dict(stage=split,task=task,pid=os.getpid(),time=time.time()))
            arr=r.run(cfg,split,meta=dict(experiment='extra_ablation',aliases=task['names']))
            assert arr is not None and all(arr[k].shape==(SPLITS[split],) for k in ('kl','pred','correct'))
            assert np.isfinite(arr['kl']).all()
            p=r.path(cfg,split)
            save(done,dict(task=task['id'],config_sha=config_hash(cfg),names=task['names'],status='ok',n=SPLITS[split],
                top1=100*float(arr['correct'].mean()),kl=float(arr['kl'].mean()),arrays=str(p.with_suffix('.npz')),
                arrays_sha=hash_file(p.with_suffix('.npz')),metadata_sha=hash_file(p.with_suffix('.json')),time=time.time()))
        finally:handle.close()


def timing():
    check();assert (ROOT/'accuracy_done.json').exists()
    for bb in BBS:
        if (ROOT/bb/'latency_done.json').exists():continue
        while not parent.idle_gpus():time.sleep(20)
        r,_=context(bb);_=r.vit;dest=ROOT/bb/'latency'
        if dest.exists():dest.rename(ROOT/bb/f'latency_interrupted_{time.time_ns()}')
        tasks=[t for t in load_json(ROOT/'tasks.json') if t['backbone']==bb]
        for rnd in range(3):
            for task in (tasks if rnd%2==0 else tasks[::-1]):
                assert parent.idle_gpus(),'another GPU job during timing'
                model=build(r.vit,r.spec,task['config']);measurement=measure_latency(model,32,iters=50,warmup=10)
                assert parent.idle_gpus(),'another GPU job during replays'
                save(dest/f"{task['id']}#{rnd}.json",dict(task=task,round=rnd,measurement=measurement,time=time.time()))
                del model;gc.collect();torch.cuda.empty_cache()
        save(ROOT/bb/'latency_done.json',dict(rounds=3,n_configs=len(tasks),time=time.time()))
        del r;gc.collect();torch.cuda.empty_cache()


def status(phase,**extra):save(ROOT/'status.json',dict(phase=phase,pid=os.getpid(),time=time.time(),**extra))


def docs():
    st=load_json(ROOT/'status.json');nt=len(load_json(ROOT/'tasks.json')) if (ROOT/'tasks.json').exists() else None
    counts={s:len(list((ROOT/'completed'/s).glob('*.json'))) for s in ('probes',*SPLITS)}
    text=f"# Supplementary D/M3/calibration ablation — 2026-10-18\n\nPhase: {st['phase']}. Unique final configs: {nt}. Completed: {counts}.\n\n54 added named configs + cached references; predecessor must pass audit before GPU work. Raw: `{ROOT}`.\n\nFull configs: `schedules/latency_d_m3_extra_ablation/2026-10-18/`. Per-image predictions, calibration indices, probe means, source snapshot and three-round timing are retained. Historical splits viewed; not a new pristine holdout.\n"
    if st.get('error'):text+='\nError: '+st['error']+'\n'
    REPORT.write_text(text);STATE.write_text(text)
    tax=REPORT.parent/'REPORT_TAXONOMY.md';s=tax.read_text();prefix=f'| [{REPORT.name}]'
    row=f'| [{REPORT.name}]({REPORT.name}) | D horizon/placebo, M3 mean/RMS norm, calibration seeds; 3 backbone×2 budget, 54 added named configs. | {st["phase"]}; '+('audit passed, not yet independently reviewed' if st['phase']=='completed' else 'not yet complete')+' |'
    if prefix in s:s='\n'.join(row if line.startswith(prefix) else line for line in s.splitlines())+'\n'
    else:
        anchor='| New report | Scope | Status |\n|---|---|---|\n'
        if anchor in s:s=s.replace(anchor,anchor+row+'\n',1)
        else:s=s.replace('## G. Plans',anchor+row+'\n\n## G. Plans',1)
    pp='| [latency-d-m3-extra-ablation-plan-2026-10-18.md]'
    pr=f'| [latency-d-m3-extra-ablation-plan-2026-10-18.md](../plan/latency-d-m3-extra-ablation-plan-2026-10-18.md) | D horizon/placebo, M3 norm, calibration seeds; 54 added named final configs. | {st["phase"]} | [{REPORT.name}]({REPORT.name}) |'
    if pp in s:s='\n'.join(pr if line.startswith(pp) else line for line in s.splitlines())+'\n'
    else:s=s.replace('| Plan | Summary | Status | Result report |\n|---|---|---|---|\n','| Plan | Summary | Status | Result report |\n|---|---|---|---|\n'+pr+'\n',1)
    tax.write_text(s)


def child(args,gpu=None,label=None):
    env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1',A6_GPU_COORDINATION_LOCK=str(ROOT/'gpu.lock'))
    if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
    p=ROOT/'logs'/(label or '-'.join(args));p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a') as stream:
        return subprocess.Popen([sys.executable,str(Path(__file__).resolve()),*args],cwd=REPO,env=env,stdout=stream,stderr=subprocess.STDOUT)


def done(stage):
    tasks=load_json(ROOT/('probe_tasks.json' if stage=='probes' else 'tasks.json'))
    return all((ROOT/'completed'/stage/f"{t['id']}.json").exists() for t in tasks)


def run_two(stage):
    if done(stage):return
    for attempt in range(1,4):
        ps={}
        for gpu in (0,1):
            while not parent.gpu_free(gpu):time.sleep(20)
            ps[gpu]=child(['probes'] if stage=='probes' else ['worker','--split',stage],gpu,f'{stage}-gpu{gpu}-attempt{attempt}.log')
        save(ROOT/'children.json',dict(stage=stage,pids={str(g):p.pid for g,p in ps.items()},attempt=attempt))
        while any(p.poll() is None for p in ps.values()):docs();time.sleep(20)
        if done(stage):return
    raise RuntimeError(f'{stage} incomplete after 3 attempts; see logs')


def coordinate():
    ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'coordinator.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            while True:
                st=load_json(PREVIOUS/'status.json')
                audit=load_json(PREVIOUS/'audit.json') if (PREVIOUS/'audit.json').exists() else {}
                if st['phase']=='completed' and audit.get('passed'):break
                status('waiting_dependency_failed' if st['phase']=='failed' else 'waiting_predecessor_audit',predecessor=st)
                docs();time.sleep(30)
            status('setup');docs();setup()
            status('probes');docs();run_two('probes')
            status('planning');docs();planning()
            for split in SPLITS:status(split);docs();run_two(split)
            save(ROOT/'accuracy_done.json',dict(time=time.time()))
            status('latency');docs()
            for attempt in range(1,4):
                while not parent.gpu_free(0) or not parent.gpu_free(1):time.sleep(20)
                p=child(['timing'],0,f'latency-attempt{attempt}.log')
                save(ROOT/'children.json',dict(stage='latency',pids={'0':p.pid},attempt=attempt))
                while p.poll() is None:docs();time.sleep(20)
                if p.returncode==0:break
            if p.returncode:raise RuntimeError('latency failed after 3 attempts')
            status('analysis');docs()
            from leaf.reproduction.experiments.horizon_norm_seed_analysis import analyze,verify
            analyze(ROOT,BBS,LEVELS);verify(ROOT,BBS,LEVELS)
            status('completed');docs()
        except Exception as exc:
            save(ROOT/'failure.json',dict(error=str(exc),traceback=traceback.format_exc(),time=time.time()))
            status('failed',error=str(exc));docs();raise


def launch():
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'scheduler.pid').exists():
        pid=int((ROOT/'scheduler.pid').read_text())
        try:os.kill(pid,0)
        except ProcessLookupError:pass
        else:raise RuntimeError(f'coordinator already alive: {pid}')
    (ROOT/'logs').mkdir(exist_ok=True)
    with (ROOT/'logs/coordinator.log').open('a') as log:
        p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'coordinate'],cwd=REPO,
            env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1'),
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    (ROOT/'scheduler.pid').write_text(str(p.pid));print('LAUNCHED',p.pid,ROOT)


if __name__=='__main__':
    os.chdir(REPO);torch.set_num_threads(4)
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('launch','coordinate','probes','planning','worker','timing','verify','status'));p.add_argument('--split',choices=tuple(SPLITS));a=p.parse_args()
    if a.phase=='worker':worker(a.split)
    elif a.phase=='probes':probe_worker()
    elif a.phase=='verify':
        from leaf.reproduction.experiments.horizon_norm_seed_analysis import verify
        verify(ROOT,BBS,LEVELS)
    elif a.phase=='status':print(json.dumps(load_json(ROOT/'status.json'),indent=2))
    else:globals()[a.phase]()
