"""Append-only low-budget campaign: sealed CPU planning, two-GPU accuracy, timing last.

Run: python leaf/reproduction/experiments/low_budget_and_factorial.py launch
Inspect: python leaf/reproduction/experiments/low_budget_and_factorial.py status
Reproduce: python leaf/reproduction/experiments/low_budget_and_factorial.py verify
The launch command detaches a coordinator; all children and logs remain discoverable.
"""
import argparse
from copy import deepcopy
import csv
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import torch
import leaf.evaluation.determinism
from leaf.evaluation.runner import save, load_json, config_hash, require_idle_gpu
from leaf.reproduction.runners.campaign_runner import CampaignRunner, build as old_build
from leaf.evaluation.data import mcnemar, holm
from leaf.latency.timing import INFERENCE_DTYPE, inference_copy, measure_latency, gmacs
from leaf.backbones import SPECS
from leaf.planner.configs import ForecastConfig, config_from_state
from leaf.planner.surrogate import ProbeCurve
from leaf.planner.probes import probe_jobs, probe_state
from leaf.planner.gmac import GmacPlanner, config_cost
from leaf.planner.plan import OperatorCost, operator_config, plan_latency, config_details
from leaf.latency.salt import fit
from leaf.reproduction.experiments.gmac_planning import hash_file, sha

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / 'results/low_budget_and_factorial'
SOURCE = REPO / 'results/gmac_planning'
VALIDATION = REPO / 'results/salt_validation'
COMPARISON = REPO / 'results/salt_fit'
REPORT = REPO / 'results/reports/latency-d-m3-low-budget-2026-10-18.md'
STATE = REPO / 'results/notes/state/latency-d-m3-low-budget-2026-10-18-progress.md'
SCHEDULES = REPO / 'schedules/latency_d_m3_low_budget/2026-10-18'
BBS = tuple(SPECS)
ABLATION = ('deit_b', 'mae_b', 'augreg_s')
RATIOS = (.9, .8, .7)
SPLITS = dict(confirm=20018, val=50000, v2=10000)
NEW = ('leaf/planner/plan.py', 'leaf/reproduction/experiments/low_budget_and_factorial.py',
       'baselines/low_budget_posthoc.py', 'tests/test_operator_aware_planning.py',
       'leaf/reproduction/experiments/low_budget_analysis.py')


def immutable(path, obj):
    obj = json.loads(json.dumps(obj, default=float))
    if path.exists():
        assert load_json(path) == obj, f'refusing to overwrite {path}'
    else:
        save(path, obj)


def csv_save(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    tmp = path.with_suffix('.tmp')
    with tmp.open('w') as out:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def build(vit, spec, cfg):
    if cfg['kind'] in ('tofu', 'rapid', 'onemore'):
        from baselines.low_budget_posthoc import build_posthoc
        return build_posthoc(vit, spec, cfg)
    return old_build(vit, spec, cfg)


class LowRunner(CampaignRunner):
    def inference_model(self, cfg):
        _ = self.vit
        if getattr(self, '_cast', None) is None:
            self._cast = inference_copy(self.vit)
        model = build(self.vit, self.spec, cfg)
        if cfg['kind'] == 'dense':
            return self._cast
        model.vit = self._cast
        for mod in model.modules():
            if mod is not self._cast and hasattr(mod, 'prepare_inference'):
                mod.prepare_inference(INFERENCE_DTYPE)
        return model.eval()


def context(bb):
    design = load_json(ROOT / 'design.json')
    r = LowRunner(bb, ROOT, design)
    r.spec = SPECS[bb]
    r.teacher_cache = SOURCE.resolve() / 'teachers'
    # Reuse only explicitly bound historical accuracy roots, exact config/batch/n.
    # Probes are read separately. Current new core configs are normally novel.
    r.reuse_roots = [SOURCE.resolve()/bb, VALIDATION.resolve()/bb]
    d = ForecastConfig(Path(design['backbones'][bb]['root'])/'groups.json', G=16, rank=32)
    return r, d


def check_sources():
    old = load_json(ROOT/'original_sources.json')
    for p, h in old.items():
        assert hash_file(REPO/p) == h, f'existing code changed: {p}'
    if (ROOT/'design.json').exists():
        doc = load_json(ROOT/'design.json')
        body = {k:v for k,v in doc.items() if k != 'sha256'}
        assert sha(body) == doc['sha256'], 'design SHA mismatch'
        for p,h in doc['new_source_hashes'].items():
            assert hash_file(REPO/p) == h, f'new sealed source changed: {p}'
        for p,h in doc['input_hashes'].items():
            assert hash_file(p) == h, f'bound input changed: {p}'
    return True


def setup():
    if (ROOT/'design.json').exists():
        check_sources()
        return
    check_sources()
    for p,h in {**load_json(ROOT/'original_sources.json'),**{p:hash_file(REPO/p) for p in NEW}}.items():
        target=ROOT/'source_snapshot'/p
        if not target.exists():
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(REPO/p,target)
        assert hash_file(target)==h, target
    base = load_json(SOURCE/'design.json')
    inputs = [SOURCE/'design.json', SOURCE/'tables/all_48_gmac_budgets.csv',
              VALIDATION/'results.json', COMPARISON/'dataset.json',
              REPO/'results/notes/plan/latency-d-m3-low-budget-plan-2026-10-18.md',
              ROOT/'baseline_adapter_provenance.json', ROOT/'original_sources.json']
    models = {}
    data = load_json(COMPARISON/'dataset.json')
    for bb in BBS:
        path = COMPARISON/'workload_cv'/bb/'station_rich.json'
        if not path.exists(): path = COMPARISON/'cv'/bb/'station_rich.json'
        inputs += [path, Path(base['backbones'][bb]['root'])/'groups.json', SOURCE/bb/'frozen.json']
        model = load_json(path)['full_model']
        # This model was fit jointly to historical M0 and M3 workloads.
        models[bb] = dict(main=model, source=str(path.resolve()))
        rows = [r for r in data['rows'] if r['backbone'] == bb]
        names = ['original_gmac', 'N_D', 'd_token']
        x = np.array([[r['features'].get(k,0) for k in names] for r in rows])
        restricted = fit(x, np.array([r['ms'] for r in rows]), model['alpha'])
        restricted.update(feature_names=names, formula='no_reduction_overhead',
                          rule='same historical rows; same locked ridge alpha; all station features removed')
        models[bb]['restricted'] = restricted
        models[bb]['training_range'] = {k: [min(r['features'].get(k,0) for r in rows),
                                                max(r['features'].get(k,0) for r in rows)]
                                               for k in model['feature_names']}
    # Hash the actual local image/label arrays once, instead of assuming they match history.
    assets = {}
    for split in ('calib_bal','calib','confirm','val','v2'):
        for suffix in ('224_u8','labels'):
            p = REPO/f'data/cache/{split}_{suffix}.npy'
            arr = np.load(p, mmap_mode='r')
            assets[str(p)] = dict(sha256=hash_file(p), shape=list(arr.shape), dtype=str(arr.dtype))
    teachers = {}
    for bb in BBS:
        for split in ('calib_bal',*SPLITS):
            p = SOURCE.resolve()/'teachers'/f'{split}_{bb}_bs32_logits.npy'
            assert p.exists(), p
            teachers[str(p)] = dict(sha256=hash_file(p), bytes=p.stat().st_size)
    doc = dict(version=1, created=time.time(), backbones=base['backbones'],
               data_cache=str(REPO/'data/cache'), original_root=base['original_root'], merge_root=base['merge_root'],
               original_source_hashes=load_json(ROOT/'original_sources.json'),
               new_source_hashes={p:hash_file(REPO/p) for p in NEW},
               input_hashes={str(p.resolve()):hash_file(p) for p in inputs},
               data=assets, teachers=teachers, models=models,
               rules=dict(primary='latency station_rich + P+D G16/r32 + M3',
                          ratios=RATIOS, ablation_backbones=ABLATION, accuracy_batch=32,
                          latency_batch=32, latency_rounds=3, latency_last=True,
                          profile='16/32/64 per exact config, report throughput and memory; retain bs32 for paired predictions',
                          no_accuracy_screening=True, selection='surrogate only; no confirm, val, V2 or new target timing feedback',
                          restricted_fit='same main-model alpha, historical rows only; no CV retuning',
                          disclosure='Historical splits viewed; two calibration/confirm duplicate contents inherited. Not a pristine holdout.',
                          error_policy='retry technical tasks up to 3; nonfinite/OOM terminal with reason; no automatic budget increase'),
               environment=dict(python=sys.version, torch=torch.__version__, platform=platform.platform()))
    doc['sha256'] = sha(doc)
    immutable(ROOT/'design.json', doc)
    budget_inventory()


def budget_inventory():
    entries = {bb:[] for bb in BBS}
    # Preserve exact config hashes and provenance of historical minima.
    for bb in BBS:
        jobs = load_json(SOURCE/bb/'frozen.json')['jobs']
        for name,row in jobs.items():
            if not name.startswith('M3/gmac/'): continue
            cfg = row['config']
            t = [load_json(SOURCE/bb/'latency'/f"{name.replace('/','|')}#{i}.json")['measurements']['32']['ms_mean'] for i in range(3)]
            entries[bb].append(dict(name=name, source=str((SOURCE/bb/'frozen.json').resolve()),
                                   config=cfg, config_sha=config_hash(cfg), gmac=config_cost(cfg,SPECS[bb]), latency_ms=float(np.mean(t))))
    for row in load_json(VALIDATION/'results.json')['records']:
        if row['formula'] not in ('frozen_t','current_refit','station_rich','split_tile32'): continue
        bb,name = row['backbone'],row['name']
        cfg = load_json(VALIDATION/bb/'plans.json')['jobs'][name]['config']
        entries[bb].append(dict(name=name, source=str((VALIDATION/bb/'plans.json').resolve()),config=cfg,
                               config_sha=config_hash(cfg),gmac=config_cost(cfg,SPECS[bb]),latency_ms=row['latency_ms']))
    budgets = {}
    for bb,rows in entries.items():
        gm = min(rows,key=lambda r:(r['gmac'],r['name']))
        tm = min(rows,key=lambda r:(r['latency_ms'],r['name']))
        budgets[bb] = dict(G0=gm, L0=tm, levels=[dict(id=f'low{int(round(100*q))}',ratio=q,
                           gmac=q*gm['gmac'],latency_ms=q*tm['latency_ms']) for q in RATIOS])
    immutable(ROOT/'inventory.json', entries)
    immutable(ROOT/'budgets.json', budgets)


def probe_curves(bb, dcfg, mode):
    """Read-only historical, exact-config probe cache with full provenance export."""
    path = ROOT/bb/f'probe_means_{mode}.json'
    if path.exists():
        records = load_json(path)
    else:
        records = []
        for l,n,th,k in probe_jobs(SPECS[bb]):
            cfg = config_from_state(probe_state(l,n,th,k),SPECS[bb],dcfg)
            if mode == 'M3' and th != 0: cfg = operator_config(cfg,'M3')
            prefix = SOURCE.resolve()/bb/'half/runs'/config_hash(cfg)
            meta = load_json(prefix.with_suffix('.json'))
            assert meta['config'] == cfg and meta['batch_size'] == 32 and meta['n'] == 5000, prefix
            with np.load(prefix.with_suffix('.npz')) as a:
                assert a['kl'].shape == (5000,) and np.isfinite(a['kl']).all()
                value = float(a['kl'].mean())
            records.append(dict(block=l,count=n,theta=th,kappa=k,kl=value,config=cfg,
                                metadata=str(prefix.with_suffix('.json')),metadata_sha=hash_file(prefix.with_suffix('.json')),
                                arrays=str(prefix.with_suffix('.npz')),arrays_sha=hash_file(prefix.with_suffix('.npz'))))
        immutable(path, records)
    rows = {}
    for a in records:
        rows.setdefault((a['theta'],a['kappa']),{}).setdefault(a['block'],{})[SPECS[bb].n_tokens-a['count']]=a['kl']
    cv = {key:ProbeCurve(value,SPECS[bb].n_tokens) for key,value in rows.items()}
    assert all(max(deltas) == 194 for c in cv.values() for deltas,values in c.curves.values())
    return cv


def baseline_jobs(bb):
    out = {'dense':dict(config=dict(kind='dense'), group='reference')}
    old = load_json(SOURCE/bb/'frozen.json')['jobs']
    for name,row in old.items():
        if name.startswith('baseline/') or name.startswith('tome_m3/'):
            out['history/'+name] = dict(config=row['config'], group='baseline', historical=True)
    # Locked structural grid; includes r beyond historical r16, legal native clamping retained.
    for kind in ('tome','tome_m3','tofu','rapid'):
        for reduction in (16,20,24,32,48):
            out[f'baseline/{kind}/r{reduction}'] = dict(config=dict(kind=kind,r=reduction),group='baseline', grid='low_budget_extension')
    for kr in (.15,.25,.35,.45):
        out[f'baseline/evit/kr{kr}'] = dict(config=dict(kind='evit',keep_rate=kr),group='baseline', grid='low_budget_extension')
    return out


def planning(bb):
    setup()
    dest = ROOT/bb/'plans.json'
    if dest.exists():
        doc=load_json(dest); assert sha(doc['jobs'])==doc['sha']; return
    spec = SPECS[bb]; _,d = context(bb)
    curves = {mode:probe_curves(bb,d,mode) for mode in ('M0','M3')}
    models = load_json(ROOT/'design.json')['models'][bb]
    budgets = load_json(ROOT/'budgets.json')[bb]
    jobs = baseline_jobs(bb)
    for tag in ('G0','L0'):
        jobs['anchor/'+tag] = dict(config=budgets[tag]['config'],group='reference',historical_source=budgets[tag]['source'])
    gmac = GmacPlanner(spec,d,curves['M3'],'M3')
    for budget in budgets['levels']:
        bid = budget['id']
        for axis in ('latency','gmac'):
            started=time.monotonic()
            result = (plan_latency(curves['M3'],spec,d,models['main'],budget['latency_ms']) if axis=='latency' else gmac.plan(budget['gmac']))
            name=f'full/{axis}/{bid}'
            jobs[name]=dict(result,group='full',axis=axis,budget=budget,search_seconds=time.monotonic()-started)
            save(ROOT/bb/'planning_progress.json',dict(last=name,jobs=jobs))
        if bb in ABLATION:
            base = jobs[f'full/latency/{bid}']
            for mode,use_d,arm in (('M0',False,'m0_p'),('M0',True,'m0_pd'),('M3',False,'m3_p')):
                result=plan_latency(curves[mode],spec,d,models['main'],budget['latency_ms'],mode,use_d)
                jobs[f'replan/{bid}/{arm}']=dict(result,group='replan',axis='latency',budget=budget,mode=mode,use_d=use_d)
                if 'config' in base:
                    cfg=operator_config(base['config'],mode,use_d)
                    jobs[f'fixed/{bid}/{arm}']=dict(config=cfg,status='planned',group='fixed',axis='fixed_path',budget=budget,
                        mode=mode,use_d=use_d,state=base['state'],gmac=config_cost(cfg,spec),
                        note='same full-method count path; fixed theta/counts, score/operator only')
                else:
                    jobs[f'fixed/{bid}/{arm}']=dict(group='fixed',status='dependency_infeasible',budget=budget)
            result=plan_latency(curves['M3'],spec,d,models['restricted'],budget['latency_ms'])
            jobs[f'pricing/{bid}/no_overhead']=dict(result,group='pricing',axis='latency',budget=budget)
    from leaf.latency.salt import features
    for name,row in jobs.items():
        if 'config' not in row: continue
        cfg=row['config']
        row['config_sha']=config_hash(cfg)
        row['details']=config_details(cfg,spec)
        if cfg['kind'] in ('pd','pdm'):
            f=features(cfg,spec)
            row['feature_extrapolation']={k:dict(value=f.get(k,0),historical_range=rng) for k,rng in models['training_range'].items() if not rng[0]-1e-10<=f.get(k,0)<=rng[1]+1e-10}
            row['prediction_main_ms']=OperatorCost(models['main'],spec,d,'M3' if cfg['kind']=='pdm' else 'M0')(cfg)
    immutable(dest,dict(jobs=jobs,sha=sha(jobs)))
    immutable(SCHEDULES/f'{bb}.json',dict(backbone=bb,timm_id=load_json(ROOT/'design.json')['backbones'][bb]['timm_id'],
                    design_sha=load_json(ROOT/'design.json')['sha256'],jobs=jobs,sha=sha(jobs)))
    print('PLANNED',bb,len(jobs),'slots',flush=True)


def tasks():
    grouped = {}
    for bb in BBS:
        doc=load_json(ROOT/bb/'plans.json'); assert sha(doc['jobs'])==doc['sha']
        for name,row in doc['jobs'].items():
            if 'config' not in row: continue
            cfg=row['config']; key=bb+'-'+config_hash(cfg)
            grouped.setdefault(key,dict(id=key,backbone=bb,config=cfg,names=[]))['names'].append(name)
    return sorted(grouped.values(),key=lambda t:(BBS.index(t['backbone']),
                   0 if any(n.startswith('full/') for n in t['names']) else 1,t['id']))


def profile(r,cfg):
    dest=r.root/'profiles'/f'{config_hash(cfg)}.json'
    if dest.exists(): return load_json(dest)
    model=r.inference_model(cfg); rows=[]; failure=None
    checkpoint=r.root/'checkpoint_fingerprint.json'
    if not checkpoint.exists():
        h=hashlib.sha256(); tensors=[]
        for name,value in r.vit.state_dict().items():
            value=value.detach().cpu().contiguous()
            header=json.dumps([name,list(value.shape),str(value.dtype)]).encode();h.update(header)
            h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
            tensors.append(dict(name=name,shape=list(value.shape),dtype=str(value.dtype)))
        immutable(checkpoint,dict(timm_id=r.binding['timm_id'],state_dict_sha256=h.hexdigest(),tensors=tensors))
    for batch in (16,32,64):
        try:
            x=torch.randn(batch,3,224,224,device='cuda'); torch.cuda.reset_peak_memory_stats()
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                y=model(x); torch.cuda.synchronize();start=time.monotonic()
                for _ in range(3): y=model(x)
                torch.cuda.synchronize(); elapsed=time.monotonic()-start
                repeat=model(x)
            peak=torch.cuda.max_memory_allocated();reserve=torch.cuda.get_device_properties(0).total_memory-peak
            rows.append(dict(batch=batch,images_per_s=3*batch/elapsed,peak_bytes=peak,reserve_bytes=reserve,
                             finite=bool(torch.isfinite(y).all()),deterministic=torch.equal(y,repeat)))
            del x,y,repeat
        except torch.cuda.OutOfMemoryError:
            rows.append(dict(batch=batch,status='OOM'))
            gc.collect();torch.cuda.empty_cache()
            if batch<=32: failure='OOM';break
    bs32=next((v for v in rows if v['batch']==32),{})
    if failure is None and (not bs32.get('finite') or not bs32.get('deterministic')): failure='correctness_gate_failed'
    if failure is None and bs32.get('reserve_bytes',0)<2*2**30: failure='insufficient_bs32_memory_reserve'
    counted=None
    if failure is None and cfg['kind'] not in ('pd','pdm'):
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):counted=gmacs(model)
    row=dict(config=cfg,rows=rows,chosen=32,counted_gmac=counted,best_profile_batch=max((v for v in rows if 'images_per_s' in v),key=lambda v:v['images_per_s'])['batch'] if any('images_per_s' in v for v in rows) else None,
             reason='bs32 retained for batch-matched paired predictions; throughput and memory profile recorded',failure=failure)
    save(dest,row);del model;gc.collect();torch.cuda.empty_cache()
    return row


def worker(split):
    check_sources();require_idle_gpu();gpu_precheck();r=None;current=None
    for task in load_json(ROOT/'tasks.json'):
        target=ROOT/'completed'/split/f"{task['id']}.json"
        if target.exists():continue
        lock=ROOT/'claims'/split/task['id'];lock.parent.mkdir(parents=True,exist_ok=True)
        with lock.open('a+') as stream:
            try: fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError: continue
            if target.exists():continue
            prior=ROOT/'completed'/'confirm'/f"{task['id']}.json"
            if split!='confirm' and load_json(prior)['status']!='ok':
                save(target,dict(task=task['id'],status='dependency_failed',reason=load_json(prior)['status']));continue
            bb=task['backbone'];cfg=task['config']
            if bb!=current:
                r=None;gc.collect();torch.cuda.empty_cache();r,_=context(bb);current=bb
            require_idle_gpu()
            save(ROOT/f"worker_gpu{os.environ['CUDA_VISIBLE_DEVICES']}.json",dict(pid=os.getpid(),split=split,task=task,time=time.time()))
            prof=profile(r,cfg)
            if prof['failure']:
                save(target,dict(task=task['id'],status=prof['failure'],profile=str(r.root/'profiles'/f'{config_hash(cfg)}.json')));continue
            arr=r.run(cfg,split,meta=dict(experiment='latency_d_m3_low_budget',aliases=task['names'],design_sha=load_json(ROOT/'design.json')['sha256']))
            if arr is None:
                save(target,dict(task=task['id'],status='nonfinite'));continue
            assert all(arr[k].shape==(SPLITS[split],) for k in ('kl','pred','correct')) and np.isfinite(arr['kl']).all()
            prefix=r.path(cfg,split)
            row=dict(task=task['id'],status='ok',backbone=bb,names=task['names'],config_sha=config_hash(cfg),
                     n=SPLITS[split],top1=100*float(arr['correct'].mean()),kl=float(arr['kl'].mean()),
                     arrays=str(prefix.with_suffix('.npz')),arrays_sha=hash_file(prefix.with_suffix('.npz')),
                     metadata_sha=hash_file(prefix.with_suffix('.json')),time=time.time())
            save(target,row);print('DONE',split,task['id'],row['top1'],flush=True)


def idle_gpus():
    query=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
    other=[int(s.strip()) for s in query.splitlines() if s.strip() and int(s.strip())!=os.getpid()]
    return not other


def gpu_precheck():
    """Independent reference parity at extreme compression before any accuracy worker."""
    gpu=os.environ['CUDA_VISIBLE_DEVICES'];dest=ROOT/f'gpu_precheck_{gpu}.json'
    if dest.exists():assert load_json(dest)['passed'];return
    require_idle_gpu()
    sys.path.insert(0,str(REPO/'tests'))
    from test_merge_norm import gpu_tests
    rows=gpu_tests()
    save(dest,dict(passed=True,rows=rows,device=torch.cuda.get_device_name(0),gpu=gpu,time=time.time()))


def timing():
    check_sources();assert (ROOT/'accuracy_done.json').exists()
    for bb in BBS:
        if (ROOT/bb/'latency_done.json').exists():continue
        while not idle_gpus():time.sleep(20)
        r,_=context(bb);_=r.vit
        selected=[t for t in load_json(ROOT/'tasks.json') if t['backbone']==bb and load_json(ROOT/'completed'/'confirm'/f"{t['id']}.json")['status']=='ok']
        dest=ROOT/bb/'latency'
        if dest.exists():dest.rename(ROOT/bb/f'latency_interrupted_{time.time_ns()}')
        clean=True
        for rnd in range(3):
            for task in (selected if rnd%2==0 else selected[::-1]):
                if not idle_gpus():
                    save(ROOT/bb/'latency_contaminated.json',dict(time=time.time(),round=rnd));clean=False;break
                cfg=task['config'];model=build(r.vit,r.spec,cfg)
                try:measurement=measure_latency(model,32,iters=50,warmup=10)
                except torch.cuda.OutOfMemoryError:measurement=dict(status='OOM')
                # A process appearing during replays invalidates the session as well.
                if not idle_gpus():clean=False
                save(dest/f"{task['id']}#{rnd}.json",dict(task=task['id'],config=cfg,config_sha=config_hash(cfg),round=rnd,
                    measurement=measurement,clean=clean,device=torch.cuda.get_device_name(0),time=time.time()))
                del model;gc.collect();torch.cuda.empty_cache()
                if not clean:break
            if not clean:break
        del r;gc.collect();torch.cuda.empty_cache()
        if not clean:raise RuntimeError('latency contaminated by another GPU job; preserved session, retry from start')
        save(ROOT/bb/'latency_done.json',dict(rounds=3,batch=32,n_configs=len(selected),time=time.time()))


def phase_complete(split):
    return all((ROOT/'completed'/split/f"{t['id']}.json").exists() for t in load_json(ROOT/'tasks.json'))


def set_status(phase,**extra):
    row=dict(phase=phase,pid=os.getpid(),time=time.time(),**extra)
    save(ROOT/'status.json',row)
    return row


def update_docs(final=False):
    status=load_json(ROOT/'status.json') if (ROOT/'status.json').exists() else dict(phase='setup')
    counts={s:len(list((ROOT/'completed'/s).glob('*.json'))) for s in SPLITS}
    all_tasks=load_json(ROOT/'tasks.json') if (ROOT/'tasks.json').exists() else []
    lines=['# Latency + D + M3 at low budget — 2026-10-18','',
           f"Status: {status['phase']}; {'execution and audit complete, not yet independently reviewed' if final else 'not complete' }.",'',
           f"Unique configs: {len(all_tasks)}; confirm {counts['confirm']}, val {counts['val']}, V2 {counts['v2']}.",
           '', '7 existing backbones; budget set a further 10/20/30% below the historical minimum. Latency-primary station_rich; GMAC control. D×M0/M3 replan and fixed-path on DeiT-B/MAE-B/AugReg-S. The historical data has already been seen; this is not a new blind holdout.',
           '',f'Raw and manifest: `{ROOT}`. Full configuration: `schedules/latency_d_m3_low_budget/2026-10-18/*.json`. All negative points, invalid/OOM/infeasible points and all aliases are kept.',
           '', 'Re-check: `/home/huynhvien/ml/bin/python leaf/reproduction/experiments/low_budget_and_factorial.py verify`; see `REPRODUCE.md` in the raw campaign.',
           '', 'Each prediction cache has KL/pred/correct arrays in the exact image order, plus config/batch/checkpoint metadata and SHA. Each latency config has three per-round files; files from interrupted sessions are kept separately.']
    if status.get('error'):lines+=['', 'Current error: `'+status['error'].replace('\n',' ')+'`.']
    if (ROOT/'all_arms.csv').exists():lines+=['', 'Result table: `all_arms.csv`; statistics: `paired_comparisons.csv`, `factorial_effects.csv`, `coverage.csv`; audit: `audit.json`.']
    REPORT.write_text('\n'.join(lines)+'\n')
    STATE.write_text(f"Phase: {status['phase']}; {len(all_tasks)} unique configs; confirm/val/V2: {counts}.\n\nReport: [latency-d-m3-low-budget-2026-10-18.md](../reports/latency-d-m3-low-budget-2026-10-18.md).\nRaw: `{ROOT}`.\n")
    taxonomy=REPORT.parent/'REPORT_TAXONOMY.md';text=taxonomy.read_text();prefix=f'| [{REPORT.name}]'
    row=f'| [{REPORT.name}]({REPORT.name}) | 7 backbones, low-budget latency/GMAC + D×M3 replan/fixed-path, per-image artifacts and config SHA. | '+('Complete/audited; not yet independently reviewed' if final else 'Running: '+status['phase'])+' |'
    if prefix in text:text='\n'.join(row if l.startswith(prefix) else l for l in text.splitlines())+'\n'
    else:text=text.replace('## D.','## D.',1) if False else text.replace('## G. Plans',row+'\n\n## G. Plans',1)
    # Keep the summary in F rather than adding run logs as research reports.
    if prefix not in taxonomy.read_text():
        text=text.replace(row+'\n\n## G.', '## G.',1)
        marker='## G. Plans'
        pos=text.find(marker)
        # Insert an independent, correctly formed small table in F immediately before G.
        text=text[:pos]+'| New report | Scope | Status |\n|---|---|---|\n'+row+'\n\n'+text[pos:]
    planprefix='| [latency-d-m3-low-budget-plan-2026-10-18.md]'
    replacement='| [latency-d-m3-low-budget-plan-2026-10-18.md](../plan/latency-d-m3-low-budget-plan-2026-10-18.md) | 7 backbones, low budget 10/20/30%; 105 core slots before dedup, plus baselines/anchors. | '+('Execution/audit complete' if final else 'Running: '+status['phase'])+f' | [{REPORT.name}]({REPORT.name}) |'
    text='\n'.join(replacement if l.startswith(planprefix) else l for l in text.splitlines())+'\n'
    taxonomy.write_text(text)


def child(command,gpu=None,logname=None):
    env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1',
             A6_GPU_COORDINATION_LOCK=str(ROOT/'gpu_coordination.lock'))
    if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
    log=ROOT/'logs'/(logname or ('-'.join(command)+'.log'));log.parent.mkdir(parents=True,exist_ok=True)
    stream=log.open('a')
    p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),*command],cwd=REPO,env=env,stdout=stream,stderr=subprocess.STDOUT)
    stream.close();return p


def coordinate():
    ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'coordinator.lock').open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('coordinator already active')
        try:
            set_status('setup');update_docs();setup()
            set_status('planning');update_docs()
            for bb in BBS:
                p=child(['plan','--backbone',bb],logname=f'planning-{bb}.log')
                while p.poll() is None:time.sleep(5)
                if p.returncode:raise RuntimeError(f'planning failed {bb}; see logs/planning-{bb}.log')
                update_docs()
            immutable(ROOT/'tasks.json',tasks())
            immutable(ROOT/'plans_sealed.json',dict(design_sha=load_json(ROOT/'design.json')['sha256'],
                       plan_hashes={bb:hash_file(ROOT/bb/'plans.json') for bb in BBS},tasks_sha=hash_file(ROOT/'tasks.json')))
            save(ROOT/'execution_inventory.json',dict(named_slots=sum(len(load_json(ROOT/bb/'plans.json')['jobs']) for bb in BBS),
                       unique_configs=len(load_json(ROOT/'tasks.json')),core_slots=105,
                       no_config_slots=[dict(backbone=bb,name=n,**{k:v for k,v in r.items() if k in ('status','infeasible','budget')}) for bb in BBS for n,r in load_json(ROOT/bb/'plans.json')['jobs'].items() if 'config' not in r]))
            for split in SPLITS:
                if phase_complete(split):continue
                set_status(split);update_docs()
                for attempt in range(1,4):
                    ps={}
                    for gpu in (0,1):
                        # Each worker refuses a busy GPU again before its first task.
                        while not gpu_free(gpu):time.sleep(15)
                        ps[gpu]=child(['worker','--split',split],gpu,logname=f'{split}-gpu{gpu}-attempt{attempt}.log')
                    save(ROOT/'children.json',dict(phase=split,attempt=attempt,pids={str(g):p.pid for g,p in ps.items()}))
                    while any(p.poll() is None for p in ps.values()):
                        update_docs();time.sleep(20)
                    update_docs()
                    if phase_complete(split):break
                if not phase_complete(split):raise RuntimeError(f'{split}: incomplete after 3 worker attempts; inspect logs')
            save(ROOT/'accuracy_done.json',dict(time=time.time(),tasks=len(tasks())))
            set_status('latency');update_docs()
            for attempt in range(1,4):
                while not gpu_free(0) or not gpu_free(1):time.sleep(20)
                p=child(['timing'],0,logname=f'latency-attempt{attempt}.log')
                save(ROOT/'children.json',dict(phase='latency',pids={'0':p.pid},attempt=attempt))
                while p.poll() is None:update_docs();time.sleep(20)
                if p.returncode==0:break
            if p.returncode:raise RuntimeError('latency failed after 3 attempts')
            set_status('analysis');update_docs();analyze();verify(full=True)
            set_status('completed');update_docs(True)
        except Exception as exc:
            save(ROOT/'failure.json',dict(error=str(exc),traceback=traceback.format_exc(),time=time.time()))
            set_status('failed',error=str(exc));update_docs();raise


def gpu_free(gpu):
    try:
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu))
        subprocess.run([sys.executable,'-c','from leaf.evaluation.runner import require_idle_gpu; require_idle_gpu()'],cwd=REPO,env=env,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:return False


def launch():
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'scheduler.pid').exists():
        pid=int((ROOT/'scheduler.pid').read_text())
        try:os.kill(pid,0)
        except ProcessLookupError:pass
        else:raise RuntimeError(f'coordinator already exists: {pid}')
    (ROOT/'logs').mkdir(exist_ok=True)
    with (ROOT/'logs/coordinator.log').open('a') as stream:
        p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'coordinate'],cwd=REPO,
             env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1'),
             stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    (ROOT/'scheduler.pid').write_text(str(p.pid))
    print(json.dumps(dict(pid=p.pid,root=str(ROOT),log=str(ROOT/'logs/coordinator.log'))))


def analyze():
    from leaf.reproduction.experiments.low_budget_analysis import analyze as implementation
    implementation(ROOT,BBS,ABLATION)


def verify(full=False):
    check_sources()
    if not (ROOT/'tasks.json').exists():return
    seal=load_json(ROOT/'plans_sealed.json')
    assert hash_file(ROOT/'tasks.json')==seal['tasks_sha']
    checks=0
    for bb,h in seal['plan_hashes'].items():assert hash_file(ROOT/bb/'plans.json')==h
    for task in load_json(ROOT/'tasks.json'):
        for split,n in SPLITS.items():
            p=ROOT/'completed'/split/f"{task['id']}.json"
            if not p.exists():
                if full:raise RuntimeError(f'missing {p}')
                continue
            row=load_json(p)
            if row['status']!='ok':continue
            arrpath=Path(row['arrays']);meta=load_json(arrpath.with_suffix('.json'))
            assert hash_file(arrpath)==row['arrays_sha'] and hash_file(arrpath.with_suffix('.json'))==row['metadata_sha']
            assert meta['config']==task['config'] and meta['batch_size']==32 and meta['n']==n
            with np.load(arrpath) as a:
                assert all(a[k].shape==(n,) for k in ('correct','pred','kl'))
                assert np.isfinite(a['kl']).all()
                labels=np.load(REPO/f'data/cache/{split}_labels.npy')
                assert np.array_equal(a['correct'],a['pred']==labels)
                assert abs(row['top1']-100*float(a['correct'].mean()))<1e-9
            checks+=1
    if full:
        for p,info in load_json(ROOT/'design.json')['data'].items():assert hash_file(p)==info['sha256']
        for p,info in load_json(ROOT/'design.json')['teachers'].items():assert hash_file(p)==info['sha256']
        for bb in BBS:
            for mode in ('M0','M3'):
                for probe in load_json(ROOT/bb/f'probe_means_{mode}.json'):
                    assert hash_file(probe['metadata'])==probe['metadata_sha']
                    assert hash_file(probe['arrays'])==probe['arrays_sha']
            assert (ROOT/bb/'latency_done.json').exists()
        from leaf.reproduction.experiments.low_budget_analysis import audit_statistics
        audit_statistics(ROOT)
        save(ROOT/'audit.json',dict(passed=True,prediction_arrays_checked=checks,old_code_unchanged=True,time=time.time(),independent_review=False))
    print('VERIFIED',checks,'prediction arrays; existing code unchanged',flush=True)


if __name__=='__main__':
    os.chdir(REPO)
    torch.set_num_threads(4)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('launch','coordinate','setup','plan','worker','timing','analyze','verify','status'))
    p.add_argument('--backbone',choices=BBS);p.add_argument('--split',choices=tuple(SPLITS));p.add_argument('--full',action='store_true')
    a=p.parse_args()
    if a.phase=='plan':planning(a.backbone)
    elif a.phase=='worker':worker(a.split)
    elif a.phase=='status':
        print(json.dumps(load_json(ROOT/'status.json') if (ROOT/'status.json').exists() else {},indent=2))
    elif a.phase=='verify':verify(a.full)
    else:globals()[a.phase]()
