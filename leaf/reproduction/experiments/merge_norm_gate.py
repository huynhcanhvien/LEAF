"""Section 7.1: detached/restartable GPU scheduler; correctness, end-to-end and fair cost gate.

python leaf/reproduction/experiments/merge_norm_gate.py launch
Results never replace stage-1 artifacts. GPU tests and timing run sequentially under
this single scheduler, with both cards otherwise idle. No val/V2 runs in this gate.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from leaf.evaluation.runner import build, config_hash, load_json, save
from leaf.reproduction.runners.merge_variant_runner import design, bind_baselines, MergeVariantRunner

ROOT = Path('results/merge_variants/stage15')
BASELINE = 'results/leaf_m0'


def idle():
    subprocess.run(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used', '--format=csv'], check=True)
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
    other = [p.strip() for p in apps.splitlines() if p.strip() and int(p.strip()) != os.getpid()]
    if other: raise RuntimeError(f'gate requires both GPUs otherwise idle: {other}')


def cfg(base, mode):
    return dict(base, kind='pdm', merge=dict(mode=mode, kernel='triton_norm'))


def timed(fn, repeats=5, iters=30):
    import torch
    values=[]
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for _ in range(5): fn()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = fn()
        for _ in range(5): graph.replay()
        for _ in range(repeats):
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters): graph.replay()
            end.record(); end.synchronize(); values.append(start.elapsed_time(end)/iters)
    return values


def run(args):
    import fcntl
    root=Path(args.root); root.mkdir(parents=True, exist_ok=True)
    lock=(root/'scheduler.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    import leaf.evaluation.determinism
    import numpy as np
    import torch
    from leaf.model.mlerp_vit import compress_norm
    from leaf.evaluation.data import iterate
    from leaf.evaluation.runner import per_image_kl
    sys.path.insert(0, str(Path('tests').resolve()))
    from test_merge_norm import gpu_tests
    idle()
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in map(Path,
        ['leaf/model/kernels/mlerp_merge.py','leaf/model/mlerp_vit.py','leaf/reproduction/experiments/merge_norm_gate.py','tests/test_merge_norm.py','leaf/evaluation/runner.py'])}
    save(root/'status.json', dict(state='running',pid=os.getpid(),source_hashes=hashes))
    try:
        # Short task-specific microbatch profile before the test/benchmark workloads.
        x=torch.randn(32,197,768,device='cuda'); s=torch.ones(32,197,1,device='cuda'); score=torch.randn(32,197,device='cuda')
        profile=[]
        for mode in ('M0','M3','N4'):
            for bs in (1,8,16,32):
                torch.cuda.reset_peak_memory_stats()
                times=timed(lambda:compress_norm(x[:bs],s[:bs],score[:bs],197,97,mode),repeats=2,iters=5)
                peak=torch.cuda.max_memory_allocated()/2**30
                profile.append(dict(mode=mode,batch=bs,ms=float(np.median(times)),images_per_s=1000*bs/float(np.median(times)),peak_gib=peak))
                if peak>14: raise RuntimeError('bs32 norm workload lacks headroom')
        save(root/'profile.json',dict(rows=profile,chosen_microbench=32,reason='protocol-fixed bs32; task profile and headroom checked',
             fastest={mode:max((r for r in profile if r['mode']==mode),key=lambda r:r['images_per_s'])['batch'] for mode in ('M0','M3','N4')}))
        del x,s,score
        print('GPU correctness tests',flush=True)
        save(root/'kernel_tests.json',dict(rows=gpu_tests(),fp32_tolerance=1e-5))
        rule=bind_baselines(design(exclude_n2=True),args.baseline_root)
        end2end=[]
        for backbone,name in (('deit_b','latency/DR11.5'),('mae_b','latency/DR11.5')):
            arm=next(a for a in rule['arms'] if a['backbone']==backbone and a['name']==name)
            r=MergeVariantRunner(backbone,root=root/backbone,batch=arm['evaluation_batch'],data_cache=args.data_cache,
                                 teacher_cache=Path(args.baseline_root)/'teachers',timm_id=arm['timm_id'])
            r.spec=__import__('leaf.backbones',fromlist=['SPECS']).SPECS[backbone]
            im,lb,teacher=r.data('half')
            image,_=next(iterate(im,lb,r.batch,'cuda',r.spec.mean,r.spec.std))
            # Profile the real end-to-end path on the inherited comparison batch.
            for mode in ('M0','M3','N4'):
                model=r.inference_model(cfg(arm['baseline'],mode))
                torch.cuda.reset_peak_memory_stats()
                times=timed(lambda:model(image),repeats=2,iters=3)
                peak=torch.cuda.max_memory_allocated()/2**30
                if peak>14: raise RuntimeError('end-to-end batch lacks headroom')
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                    out=model(image).float(); repeat=model(image).float()
                    ref_cfg=arm['baseline'] if mode=='M0' else dict(arm['baseline'],kind='pdm',merge=dict(mode=mode))
                    ref=r.inference_model(ref_cfg)(image).float()
                if not torch.equal(out,repeat): raise RuntimeError('end-to-end nondeterministic')
                changed=out.argmax(-1)!=ref.argmax(-1)
                margin=ref.topk(2,-1).values.diff(dim=-1).abs().squeeze(-1)
                error=(out-ref).abs().amax(-1)
                if mode=='M0' and (not torch.equal(out,ref) or not torch.equal(per_image_kl(out,teacher[:len(out)].cuda()),per_image_kl(ref,teacher[:len(out)].cuda()))):
                    raise RuntimeError('M0 bitwise regression')
                # Changed predictions require an actual near tie relative to measured logit perturbation.
                if bool((changed & (margin>2*error+1e-6)).any()): raise RuntimeError('prediction differs without near tie')
                end2end.append(dict(backbone=backbone,mode=mode,batch=r.batch,changed_predictions=int(changed.sum()),
                    max_logit_error=float(error.max()),changed_margins=margin[changed].tolist(),repeat=True,peak_gib=peak,ms=float(np.median(times))))
                del model,ref,out,repeat
            del r,image,im,lb,teacher
            import gc
            gc.collect();torch.cuda.empty_cache()
        save(root/'end2end.json',dict(rows=end2end))
        costs=[]
        for arm in rule['arms']:
            n=197;c=__import__('leaf.backbones',fromlist=['SPECS']).SPECS[arm['backbone']].dim
            for block,(p,m) in enumerate(zip(arm['baseline']['prune_keep'],arm['baseline']['merge_keep'])):
                p,m=min(n,p),min(n,p,m)
                if p>m:
                    idle()
                    torch.manual_seed(151+block)
                    x=torch.randn(32,n,c,device='cuda');s=torch.rand(32,n,1,device='cuda')+.5;score=torch.randn(32,n,device='cuda')
                    times={mode:timed(lambda mode=mode:compress_norm(x,s,score,p,m,mode)) for mode in ('M0','M3','N4')}
                    for mode in ('M3','N4'):
                        costs.append(dict(backbone=arm['backbone'],name=arm['name'],block=block,mode=mode,
                            n=n,p=p,m=m,c=c,batch=32,ms=float(np.median(times[mode])),m0_ms=float(np.median(times['M0'])),
                            extra_ms=float(np.median(times[mode])-np.median(times['M0'])),samples=times[mode],m0_samples=times['M0']))
                    del x,s,score
                n=m
        save(root/'microbench.json',dict(rows=costs,precision='FP32 residual under BF16 autocast; all Triton',both_cards_idle=True,
             timing='CUDA graph replay for M0/M3/N4; identical full compression scope'))
        gate={mode:dict(max_extra_ms=max(r['extra_ms'] for r in costs if r['mode']==mode),
                       passed=max(r['extra_ms'] for r in costs if r['mode']==mode)<=.05) for mode in ('M3','N4')}
        save(root/'gate.json',dict(modes=gate,threshold_ms=.05,source_hashes=hashes,stage2_allowed=all(g['passed'] for g in gate.values())))
        save(root/'status.json',dict(state='complete',pid=os.getpid(),gate=gate))
        print(json.dumps(gate),flush=True)
    except Exception as exc:
        save(root/'status.json',dict(state='failed',pid=os.getpid(),reason=str(exc)));raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['launch','run'])
    p.add_argument('--root',default=str(ROOT));p.add_argument('--baseline-root',default=BASELINE);p.add_argument('--data-cache',default=str(Path('data/cache').resolve()))
    args=p.parse_args();Path(args.root).mkdir(parents=True,exist_ok=True)
    if args.action=='launch':
        log=(Path(args.root)/'scheduler.log').open('a')
        command=[sys.executable,'-u',__file__,'run','--root',args.root,'--baseline-root',args.baseline_root,'--data-cache',args.data_cache]
        child=subprocess.Popen(command,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0'),start_new_session=True,stdout=log,stderr=subprocess.STDOUT)
        log.close();print(f'setsid stage1.5 scheduler pid={child.pid}')
    else:run(args)


if __name__=='__main__':main()
