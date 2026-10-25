"""Section 7.2 scheduler. Two parallel probe/confirm workers, then exclusive latency
sessions and paired val/V2. Restart by running the same launch command. No paper edits.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from leaf.evaluation.runner import Runner, build, config_hash, load_json, save, require_idle_gpu
from leaf.reproduction.runners.sealed_stage_protocol import Stage, digest
from leaf.reproduction.experiments.merge_norm_runner import NormCampaignRunner, cached_original, pair_design, paired_tests, latency_rounds
from leaf.planner.configs import ForecastConfig, config_from_state, diffrate_config
from leaf.planner.probes import probe_jobs, probe_state
from leaf.reproduction.ablation_models.merge_operator_planner import NormLat, NormPlanner, norm_config, curves_from_cache
from leaf.planner.lagrange import plan
from leaf.backbones import SPECS

ROOT='results/merge_variants/stage2/attempt2'
ORIGINAL='results/leaf_m0'
GATE='results/merge_variants/stage15/attempt4'
BACKBONES=('deit_b','mae_b')
MODES=('M3','N4')
BUDGETS={'deit_b':('DR8.7','DR10.0','DR10.4','DR11.5','DR12.5'),'mae_b':('DR8.7','DR10.0','DR10.4','DR11.5')}
DEVIATION='User explicitly waived the cost gate after Triton tests; M3/N4 replanning is authorized, including N4 below the original accuracy threshold.'


def progress(args,status,finished=False):
    state=Path('results/notes/state/merge-variants-progress.md')
    text=state.read_text() if state.exists() else ''
    start='<!-- merge-norm-stage2-start -->';end='<!-- merge-norm-stage2-end -->'
    if start in text:
        text=text.split(end,1)[1].lstrip()
    block=f'''{start}
# §7 — Triton M3/N4 and rescheduling

Status: **{status}**.
New decision: the user waived the cost gate and approved trial scheduling for both M3/N4.
The kernels passed the FP32 ≤1e-5 test and are deterministic; on one DeiT-B/MAE-B batch,
no prediction changed relative to the reference. Maximum CUDA graph overhead: M3 +0.07337 ms/station;
N4 +0.10670 ms/station. The per-backbone increase is added to c_st at stations with merge.
Each of the two methods uses M3 or N4 consistently, never mixed with M0 in one schedule.
Probe θ=0 uses the paper-final cache; θ=1/2,1 are run fresh. The M0 planner passed a test that exactly
reproduces all 18 locked schedules (DeiT-B 10, MAE-B 8).
Root: `{args.root}`. Monitor `status.json`, `prepare_deit_b.log`, `prepare_mae_b.log`.
Restart: `python leaf/reproduction/experiments/merge_norm_evaluation.py launch` (only after the old scheduler has stopped).
V2 runs only after confirm → freeze → latency → pairs. Do not edit the paper.
{end}

'''
    state.write_text(block+text)
    path=Path('results/reports/REPORT_TAXONOMY.md');lines=path.read_text().splitlines()
    prefix='| [merge-variants-plan-2026-10-15.md]'
    report_path='merge-variants-stage2-2026-10-16.md'
    for i,line in enumerate(lines):
        if line.startswith(prefix):
            lines[i]=prefix+'(../plan/merge-variants-plan-2026-10-15.md) | §7: Triton M3/N4, separate rescheduling, DeiT-B/MAE-B; cost gate waived by the user. | '+status+' | ['+report_path+']('+report_path+') |'
    report_row='| ['+report_path+']('+report_path+') | M3/N4 Triton: error/determinism checks, cost gate waived by the user, separate probe/scheduling and bs32 sessions; val selects the operator, V2 is an independent check. | '+status+' |'
    existing=next((i for i,line in enumerate(lines) if line.startswith('| ['+report_path+']')),None)
    if existing is not None:lines[existing]=report_row
    else:
        at=next(i for i,line in enumerate(lines) if line.startswith('| [merge-variants-2026-10-16.md]'))
        lines.insert(at+1,report_row)
    path.write_text('\n'.join(lines)+'\n')
    report=Path('results/reports')/report_path
    if not report.exists():
        report.write_text('# Merge variants — stages 1.5 and 2\n\n'+DEVIATION+'\n\nStatus: '+status+'. No phase 2 conclusion yet.\n\n'+
                          'The Triton kernels pass the FP32 ≤1e-5 test, are deterministic, and give the same predictions on the DeiT-B/MAE-B check batch. The cost still exceeds the old gate; the user approved scheduling with the measured cost.\n')


def immutable(path,obj):
    path=Path(path)
    if path.exists():
        if load_json(path)!=obj:raise RuntimeError(f'refusing to alter locked artifact {path}')
    else:save(path,obj)
    return obj


def sources():
    files=['leaf/model/kernels/mlerp_merge.py','leaf/model/mlerp_vit.py','leaf/model/token_ops.py','leaf/model/reduced_vit.py',
           'leaf/forecast/attention_forecast.py','leaf/forecast/kernels/kappa_score.py','leaf/model/kernels/mean_merge.py',
           'leaf/reproduction/ablation_models/merge_operator_planner.py','leaf/planner/lagrange.py','leaf/planner/schedule_caps.py','leaf/planner/probes.py',
           'leaf/planner/surrogate.py','leaf/latency/profile_model.py','leaf/planner/configs.py','leaf/planner/schedule_caps.py',
           'leaf/reproduction/experiments/merge_norm_runner.py','leaf/evaluation/runner.py','leaf/latency/timing.py',
           'leaf/evaluation/data.py','leaf/reproduction/runners/sealed_stage_protocol.py','leaf/reproduction/experiments/merge_norm_evaluation.py']
    return {p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files}


def setup(args):
    gate=load_json(Path(args.gate_root)/'gate.json')
    for p,expected in gate['source_hashes'].items():
        if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=expected:raise RuntimeError(f'norm gate/test source changed: {p}')
    for p in ('kernel_tests.json','end2end.json','microbench.json'):
        if not (Path(args.gate_root)/p).exists():raise RuntimeError('kernel correctness evidence missing')
    doc=dict(version=1,source_hashes=sources(),modes=list(MODES),backbones=list(BACKBONES),budgets={b:list(v) for b,v in BUDGETS.items()},
             original_root=str(Path(args.baseline_root).resolve()),data_cache=str(Path(args.data_cache).resolve()),
             gate_root=str(Path(args.gate_root).resolve()),deviation=DEVIATION,batch=32,
             baseline_reuse='theta=0 probes and all historical predictions/KL read-only',
             protocol='half probes -> Fhat-only plan -> confirm 20018 -> freeze SHA -> single bs32 session/backbone -> pairs -> val -> V2',
             disclosure='Merge operators were selected on ImageNet val; V2 is the independent check.',
             gate=gate)
    for b in BACKBONES:
        base=Path(args.baseline_root)/b/'bs32'
        for p in ('groups.json','E1/latmodel.json','budgets.json','paper_final/frozen.json'):
            path=base/p;doc.setdefault('input_hashes',{})[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
    return immutable(Path(args.root)/'design.json',doc)


def validate(args):
    doc=load_json(Path(args.root)/'design.json')
    for p,expected in doc['source_hashes'].items():
        if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=expected:raise RuntimeError(f'implementation changed after design lock: {p}')
    for p,expected in doc['input_hashes'].items():
        if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=expected:raise RuntimeError(f'input changed: {p}')
    return doc


def context(args,b):
    base=Path(args.baseline_root)/b/'bs32';r=NormCampaignRunner(b,Path(args.root)/b,base,args.data_cache,Path(args.baseline_root)/'teachers')
    r.spec=SPECS[b];d=ForecastConfig(base/'groups.json',G=16,rank=32)
    return r,d,base


def overhead(args,b,mode):
    costs=load_json(Path(args.gate_root)/'microbench.json')['rows']
    delta=max(row['extra_ms'] for row in costs if row['backbone']==b and row['mode']==mode)
    return 0. if delta<=.01 else max(0.,delta)


def probe_metadata(mode,l,n,theta,k):
    # Runner owns n (= number of evaluated images), id/config and metric fields.
    return dict(phase='probe',mode=mode,block=l,tokens_after=n,theta=theta,kappa=k)


def profile(args,r,d):
    import torch
    print(f'PROFILE {r.name} norm probing',flush=True)
    rows=[]
    for mode in MODES:
        # Largest post-cut token count and widest D horizon cover the probe task.
        config=norm_config(probe_state(1,189,1.,3.),r.spec,d,mode)
        model=r.inference_model(config)
        for batch in (16,32,64,128):
            image=torch.randn(batch,3,224,224,device='cuda');torch.cuda.reset_peak_memory_stats()
            try:
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                    for _ in range(2):model(image)
                    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(3):model(image)
                    end.record();end.synchronize()
                ms=start.elapsed_time(end)/3;peak=torch.cuda.max_memory_allocated()/2**30
                rows.append(dict(mode=mode,batch=batch,images_per_s=batch*1000/ms,peak_gib=peak))
            except torch.cuda.OutOfMemoryError:rows.append(dict(mode=mode,batch=batch,oom=True))
            del image;gc.collect();torch.cuda.empty_cache()
        del model
    save(r.root/'profile.json',dict(rows=rows,chosen=32,reason='same batch as original BF16 probes/teachers and paired baselines',
         fastest={m:max((row for row in rows if row['mode']==m and not row.get('oom')),key=lambda row:row['images_per_s'])['batch'] for m in MODES}))
    if any(row.get('oom') or row['peak_gib']>14 for row in rows if row['batch']==32):raise RuntimeError('bs32 lacks VRAM headroom')


def m0_regression(args,r,d,base):
    """Reproduce each locked M0 plan before using the same DP for a new operator."""
    path=r.root/'m0_planner_regression.json'
    if path.exists():return
    print(f'M0 PLANNER REGRESSION {r.name}',flush=True)
    original=Runner(r.name,root=base,batch=32)
    curves=curves_from_cache(original,r,r.spec,d,'M0')
    p=NormPlanner(None,r.spec,NormLat(load_json(base/'E1/latmodel.json'),r.spec),curves,d,'M0')
    locked=load_json(Path('schedules/kappa_form/leaf_m0')/f'{r.name}.json')['schedules']
    rows=[]
    for budget in load_json(base/'budgets.json')['budgets']:
        if budget['id'] not in BUDGETS[r.name]:continue
        for method,cap in (('latency',None),('latency_gmac',budget['gmac'])):
            result=plan(p,budget['latency_ms'],cap)
            config=p.cfg(result['state']);expected=locked[f"{method}/{budget['id']}"]['config']
            if config!=expected:raise RuntimeError(f'M0 planner differs from locked schedule {r.name}/{method}/{budget["id"]}')
            rows.append(dict(method=method,budget=budget['id'],key=config_hash(config),exact=True))
    save(path,dict(rows=rows))


def prepare(args,b):
    import leaf.evaluation.determinism
    import torch
    require_idle_gpu();r,d,base=context(args,b)
    m0_regression(args,r,d,base)
    profile(args,r,d)
    original=Runner(b,root=base,batch=32)
    for mode in MODES:
        print(f'PROBE {b}/{mode}',flush=True)
        counts=dict(reused_prune=0,new_merge=0,cached_merge=0)
        for l,n,theta,k in probe_jobs(r.spec):
            state=probe_state(l,n,theta,k)
            if theta==0:
                old=config_from_state(state,r.spec,d)
                cached_original(base,old,'half',5000);counts['reused_prune']+=1
                continue
            config=norm_config(state,r.spec,d,mode)
            existed=r.cached(config,'half') is not None
            if existed:counts['cached_merge']+=1
            else:
                if r.run(config,'half',meta=probe_metadata(mode,l,n,theta,k)) is None:
                    raise RuntimeError('nonfinite probe')
                counts['new_merge']+=1
            save(r.root/f'probe_progress_{mode}.json',dict(**counts,last=[l,n,theta,k]))
        curves=curves_from_cache(original,r,r.spec,d,mode)
        lat=NormLat(load_json(base/'E1/latmodel.json'),r.spec,overhead(args,b,mode))
        planner=NormPlanner(r,r.spec,lat,curves,d,mode)
        for budget in load_json(base/'budgets.json')['budgets']:
            if budget['id'] not in BUDGETS[b]:continue
            for method,cap in (('latency',None),('latency_gmac',budget['gmac'])):
                path=r.root/'search'/mode/f"{method}_{budget['id']}.json"
                if path.exists():continue
                result=plan(planner,budget['latency_ms'],cap)
                if result['state'] is None:raise RuntimeError('infeasible norm schedule')
                config=planner.cfg(result['state']);arr=r.run(config,'half')
                save(path,dict(result,config=config,method=method,mode=mode,budget=budget,
                               calib_kl=float(arr['kl'].mean()),merge_overhead_ms=lat.norm_extra))
    jobs={'dense':dict(config=dict(kind='dense'),state=None,meta={})}
    locked=load_json(Path('schedules/kappa_form/leaf_m0')/f'{b}.json')['schedules']
    for bid in BUDGETS[b]:
        jobs[bid]=dict(config=diffrate_config(r.spec,bid[2:]),state=None,meta=dict(reference='DiffRate'))
        for method in ('latency','latency_gmac'):
            name=f'{method}/{bid}';jobs['M0/'+name]=dict(config=locked[name]['config'],state=None,meta=dict(reference='LEAF-M0'))
            for mode in MODES:
                doc=load_json(r.root/'search'/mode/f'{method}_{bid}.json')
                jobs[f'{mode}/{name}']=dict(config=doc['config'],state=doc['state'],meta=dict(budget=doc['budget'],planner=mode,method=method))
    stage=Stage(r,'stage2');immutable(stage.dir/'arms.json',jobs)
    stage.confirm()
    for job in jobs.values():
        if r.cached(job['config'],'confirm') is None:raise RuntimeError('missing confirm before freeze')
        if job['config']['kind']=='pdm' and not (stage.dir/'frozen.json').exists():
            if r.cached(job['config'],'val') is not None or r.cached(job['config'],'v2') is not None:raise RuntimeError('early heldout result')
    stage.freeze()
    save(r.root/'prepared.json',dict(sha256=stage.sealed()['sha256']))
    del r;gc.collect();torch.cuda.empty_cache()


def idle_both():
    import subprocess
    require_idle_gpu()
    rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
    if any(p.strip() and int(p.strip())!=os.getpid() for p in rows.splitlines()):raise RuntimeError('session requires both GPUs otherwise idle')


def latency(args,b):
    import leaf.evaluation.determinism
    idle_both();r,_,_=context(args,b);stage=Stage(r,'stage2')
    # Restart resumes only a complete session; interrupted timing is retained and a new session directory is used.
    if (stage.dir/'pairs.json').exists():return
    if (stage.dir/'lat').exists() and list((stage.dir/'lat').glob('*.json')):
        (stage.dir/'lat').rename(stage.dir/f'lat_interrupted_{time.time_ns()}')
    stage.latency(rounds=3,batches=(32,),repeats=1,guard=idle_both)
    pairs,noise=pair_design(stage.sealed()['jobs'],stage.lat_table())
    save(stage.dir/'noise.json',noise)
    stage.lock_pairs(pairs,'latency-only pairs; fastest reference >= arm - tau; Holm within backbone/method/split over all reference families and budgets')


def heldout(args,b):
    import leaf.evaluation.determinism
    require_idle_gpu();r,_,_=context(args,b);stage=Stage(r,'stage2')
    pairs,noise=pair_design(stage.sealed()['jobs'],stage.lat_table())
    if load_json(stage.dir/'pairs.json')['pairs']!=pairs or load_json(stage.dir/'noise.json')!=noise:raise RuntimeError('pair/session changed')
    for split in ('val','v2'):stage.valid(split)


def report(args):
    output=['# Merge variants — stages 1.5 and 2', '', DEVIATION, '',
        'M3 is the MLERP of [ToFu](https://arxiv.org/abs/2312.01026); N4 keeps the weighted-mean norm and the channel mean.',
        '', 'The operator is selected on ImageNet val. V2 is an independent check, run only after freeze, latency and pairs.', '',
        '| Backbone | Split | Arm | Reference | Δtop1 (% points) | Holm | Eligible |', '|---|---|---|---|---:|---:|---|']
    kernel_tests=load_json(Path(args.gate_root)/'kernel_tests.json')['rows']
    end2end=load_json(Path(args.gate_root)/'end2end.json')['rows']
    output[7:7]=['', '§7.1: '+str(len(kernel_tests))+' kernel test configurations, max FP32 error '+str(max(row['max_abs_error'] for row in kernel_tests))+
        '; every run deterministic. Changed predictions on the end-to-end batch: '+str(sum(row['changed_predictions'] for row in end2end))+'.',
        'Benchmark M0/M3/N4 with the same bs32, BF16 autocast, FP32 residual, CUDA graph; at most +0.07337 ms/station for M3 and +0.10670 ms/station for N4.',
        'The old gate failed; gate waived by the user. Probe θ=0 reuses exactly the pd keys; every planner component stays paper-final. Confirm uses exactly the paper-final set of 20,018 images.', '']
    summaries={}
    for b in BACKBONES:
        r,_,_=context(args,b);stage=Stage(r,'stage2');jobs=stage.sealed()['jobs'];rows=stage.lat_table();pairs=load_json(stage.dir/'pairs.json')['pairs']
        out={split:paired_tests(r,jobs,rows,pairs,split) for split in ('val','v2')}
        save(r.root/'comparisons.json',out)
        for split,tests in out.items():
            for row in tests:output.append(f"| {b} | {split} | {row['arm']} | {row['ref']} | {row['delta']:+.3f} | {row['holm']:.5g} | {row['eligible']} |")
        claims={}
        for mode in MODES:
            v=[row for row in out['val'] if row['method']==mode and row['reference']=='LEAF-M0']
            claims[mode]=bool(all(row['eligible'] and row['same_or_faster'] for row in v) and any(row['delta']>0 and row['holm']<.05 for row in v) and not any(row['delta']<0 and row['holm']<.05 for row in v))
        summaries[b]=claims
        export={};seen={}
        from leaf.planner.schedule_caps import station_view
        for name,job in jobs.items():
            if not name.startswith(('M3/','N4/')):continue
            config=job['config'];v=r.cached(config,'val');v2=r.cached(config,'v2');key=config_hash(config)
            sites,counts,thetas=station_view(dict(p=config['prune_keep'],m=config['merge_keep']),197)
            export[name]=dict(config=config,stations=[dict(block=l,tokens_after=n,theta=theta,kappa=config['k'].get(str(l),0.)) for l,n,theta in zip(sites,counts,thetas)],
                method=job['meta']['method'],budget=job['meta']['budget'],planner=job['meta']['planner'],
                measured=dict(top1_val=100*float(v['correct'].mean()),top1_v2=100*float(v2['correct'].mean()),kl_val=float(v['kl'].mean()),latency_ms={'32':float(latency_rounds(rows,name).mean())}),
                equal_to=seen.get(key))
            seen.setdefault(key,name)
        path=Path('schedules/kappa_form/merge_variants')/f'{b}.json'
        save(path,dict(backbone=b,timm_id=SPECS[b].timm_id,forecast=dict(groups=16,rank=32),
            source=dict(manifest_sha256=stage.sealed()['sha256'],deviation=DEVIATION),schedules=export))
    output+=['', 'Claim rule on val: '+json.dumps(summaries), '', 'V2 is reported separately; this task does not expand to 7 backbones or edit the paper.']
    path=Path('results/reports/merge-variants-stage2-2026-10-16.md');path.write_text('\n'.join(output)+'\n')
    save(Path(args.root)/'decision.json',dict(claims=summaries,report=str(path)))
    progress(args,'§7.2 complete; see the val/V2 comparison and claim limits in the report',finished=True)


def coordinate(args):
    import fcntl
    root=Path(args.root);lock=(root/'scheduler.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    validate(args)
    save(root/'status.json',dict(state='probing_planning_confirm',pid=os.getpid()))
    progress(args,'Running probe → scheduling → confirm for M3/N4')
    common=['--root',args.root,'--baseline-root',args.baseline_root,'--data-cache',args.data_cache,'--gate-root',args.gate_root]
    def parallel(action):
        children=[]
        for gpu,b in enumerate(BACKBONES):
            log=(root/f'{action}_{b}.log').open('a')
            child=subprocess.Popen([sys.executable,'-u',__file__,action,'--backbone',b]+common,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=log,stderr=subprocess.STDOUT)
            children.append((child,log))
        codes=[child.wait() for child,_ in children]
        for _,log in children:log.close()
        if any(codes):raise RuntimeError(f'{action} workers failed: {codes}')
    try:
        parallel('prepare')
        save(root/'status.json',dict(state='latency',pid=os.getpid()))
        progress(args,'Probe/confirm done; measuring two bs32 latency sessions')
        for b in BACKBONES:
            subprocess.run([sys.executable,'-u',__file__,'latency','--backbone',b]+common,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0'),check=True)
        save(root/'status.json',dict(state='val_v2',pid=os.getpid()))
        progress(args,'Manifest/pairs locked; evaluating val, then V2')
        parallel('heldout')
        report(args)
    except Exception as exc:
        save(root/'status.json',dict(state='failed',pid=os.getpid(),reason=str(exc)));raise
    save(root/'status.json',dict(state='complete',pid=os.getpid()))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['setup','launch','scheduler','prepare','latency','heldout','report'])
    p.add_argument('--root',default=ROOT);p.add_argument('--baseline-root',default=ORIGINAL);p.add_argument('--gate-root',default=GATE)
    p.add_argument('--data-cache',default=str(Path('data/cache').resolve()));p.add_argument('--backbone',choices=BACKBONES)
    args=p.parse_args();Path(args.root).mkdir(parents=True,exist_ok=True)
    if args.action=='setup':setup(args);return
    validate(args)
    if args.action=='launch':
        log=(Path(args.root)/'scheduler.log').open('a')
        cmd=[sys.executable,'-u',__file__,'scheduler','--root',args.root,'--baseline-root',args.baseline_root,'--data-cache',args.data_cache,'--gate-root',args.gate_root]
        child=subprocess.Popen(cmd,start_new_session=True,stdout=log,stderr=subprocess.STDOUT);log.close();print(f'setsid stage2 pid={child.pid}')
    elif args.action=='scheduler':coordinate(args)
    elif args.action=='prepare':prepare(args,args.backbone)
    elif args.action=='latency':latency(args,args.backbone)
    elif args.action=='heldout':heldout(args,args.backbone)
    elif args.action=='report':report(args)


if __name__=='__main__':main()
