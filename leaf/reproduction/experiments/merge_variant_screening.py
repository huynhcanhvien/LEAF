"""Stage 1 tooling. Default `design` is CPU-only. GPU jobs require explicit commands.

  python leaf/reproduction/experiments/merge_variant_screening.py design
  python leaf/reproduction/experiments/merge_variant_screening.py seal
  python leaf/reproduction/experiments/merge_variant_screening.py launch --baseline-root <historical-cache-root>

launch uses setsid and supports restart (validated cached arms are skipped). workers
run one process per GPU. Microbenchmarks follow both workers sequentially with both
cards otherwise idle. No V2 and no stage-2 action is implemented here.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from leaf.reproduction.runners.merge_variant_runner import (ROOT, MergeVariantRunner, assert_ready, cached_arrays, design, paired_row, read_rule,
                                      seal, summarize)
from leaf.evaluation.runner import Runner, build, require_idle_gpu, config_hash, load_json, per_image_kl, save
from leaf.backbones import SPECS


def idle():
    subprocess.run(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used', '--format=csv'], check=True)
    require_idle_gpu(idle=True)


def all_idle():
    idle()
    rows = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
    if any(v.strip() and v.strip() != str(os.getpid()) for v in rows.splitlines()):
        raise RuntimeError('microbenchmark requires both GPUs otherwise idle')


def baseline(args, arm, split):
    n = args.rule['splits'][split]
    ref = arm['baseline_cache'][split]
    return cached_arrays(ref['root'], ref['name'], arm['baseline'], split, n, ref['batch'])


def profile_and_gate(args, r, arm):
    """Profile each actual mode, gate M0 equivalence and bitwise repeats, before val."""
    import torch
    from leaf.evaluation.data import iterate
    print(f"PROFILE {r.name}/{arm['name']} batch={r.batch}", flush=True)
    images, labels, teacher = r.data('half')
    cfg = arm['baseline']
    configs = dict(arm['variants'], M0=dict(cfg, kind='pdm', merge={'mode': 'M0'}))
    rows = []
    gate_x, _ = next(iterate(images, labels, r.batch, 'cuda', r.spec.mean, r.spec.std))
    old = r.inference_model(cfg)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        expected = old(gate_x).float()
    historical = baseline(args, arm, 'half')
    import numpy as np
    kl = per_image_kl(expected, teacher[:len(expected)].cuda()).cpu().numpy()
    if (not np.array_equal(expected.argmax(-1).cpu().numpy(), historical['pred'][:len(expected)]) or
            not np.array_equal(kl, historical['kl'][:len(expected)])):
        raise RuntimeError('current PD does not reproduce locked M0 calibration predictions/KL')
    del old
    for mode, config in configs.items():
        print(f"GATE/PROFILE {r.name}/{arm['name']}/{mode}", flush=True)
        model = r.inference_model(config)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            a, b = model(gate_x).float(), model(gate_x).float()
        if not torch.equal(a, b) or not torch.isfinite(a).all():
            raise RuntimeError(f'{mode}: nondeterministic/nonfinite GPU gate')
        if mode == 'M0' and (not torch.equal(a, expected) or
                              not torch.equal(per_image_kl(a, teacher[:len(a)].cuda()),
                                              per_image_kl(expected, teacher[:len(a)].cuda()))):
            raise RuntimeError('M0 fails bitwise logits/KL equivalence')
        for batch in sorted({16, 32, 64, 128, 256, r.batch}):
            x, _ = next(iterate(images, labels, batch, 'cuda', r.spec.mean, r.spec.std))
            torch.cuda.reset_peak_memory_stats()
            try:
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                    for _ in range(2): model(x)
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(4): model(x)
                    end.record(); end.synchronize()
                ms = start.elapsed_time(end) / 4
                rows.append(dict(mode=mode, batch=batch, ms=ms, images_per_s=1000 * batch / ms,
                                 peak_gib=torch.cuda.max_memory_allocated() / 2**30))
            except torch.cuda.OutOfMemoryError:
                rows.append(dict(mode=mode, batch=batch, oom=True))
            del x
            gc.collect(); torch.cuda.empty_cache()
        del model
    path = Path(args.root) / r.name / 'profiles' / (config_hash(cfg) + '.json')
    save(path, dict(rule_sha=args.sha, gate='passed', rows=rows, chosen_batch=r.batch,
                    reason='paired historical BF16 M0 cache requires same batch; reject if insufficient headroom',
                    fastest={mode: max((v for v in rows if v['mode'] == mode and not v.get('oom')),
                                      key=lambda v: v['images_per_s'])['batch'] for mode in configs}))
    if any(v.get('oom') or v['peak_gib'] > 14 for v in rows if v['batch'] == r.batch):
        raise RuntimeError('historical batch lacks 2 GiB headroom; cannot compare to M0 cache')
    del gate_x, a, b, expected
    gc.collect(); torch.cuda.empty_cache()


def worker(args):
    import leaf.evaluation.determinism  # set cuBLAS determinism before model initialization
    idle()
    for backbone in args.backbones:
        arms = [a for a in args.rule['arms'] if a['backbone'] == backbone]
        r = MergeVariantRunner(backbone, root=Path(args.root) / backbone,
                               batch=arms[0]['evaluation_batch'], data_cache=args.data_cache, teacher_cache=args.teacher_cache,
                               timm_id=arms[0]['timm_id'])
        r.spec = SPECS[backbone]
        for arm in arms:
            # M0 is read-only, including for prune-only schedules.
            for split in ('half', 'val'): baseline(args, arm, split)
            if not arm['has_merge']: continue
            profile_and_gate(args, r, arm)
            for mode, cfg in arm['variants'].items():
                for split in ('half', 'val'):
                    cached = r.path(cfg, split).with_suffix('.json')
                    if cached.exists():
                        cached_arrays(args.root, backbone, cfg, split, args.rule['splits'][split], batch=r.batch, rule_sha=args.sha)
                        continue
                    arr = r.run(cfg, split, meta=dict(stage1_rule_sha=args.sha, mode=mode))
                    if arr is None: raise RuntimeError(f'infeasible variant {backbone}/{mode}')
        del r
        gc.collect()
        import torch
        torch.cuda.empty_cache()


def benchmark(args):
    import leaf.evaluation.determinism
    import torch
    from leaf.model.merge_variants import MODES, compress_variant
    all_idle()
    torch.manual_seed(151)
    for arm in args.rule['arms']:
        if not arm['has_merge']: continue
        n = 197; c = SPECS[arm['backbone']].dim
        for block, (p, m) in enumerate(zip(arm['baseline']['prune_keep'], arm['baseline']['merge_keep'])):
            p, m = min(n, p), min(n, p, m)
            if p > m:
                # Residuals/size are FP32 in the actual BF16-autocast ViT.
                x = torch.randn(32, n, c, device='cuda')
                size = torch.rand(32, n, 1, device='cuda') + .5
                score = torch.randn(32, n, device='cuda')
                keys = torch.randn(32, n, c // SPECS[arm['backbone']].heads, device='cuda')
                forecast = None
                if str(block) in arm['baseline'].get('k', {}):
                    forecast = (torch.randn(32, c, device='cuda') / c**.5, 1e-6)
                times = {}; memory = {}
                for mode in ['M0'] + args.rule['modes']:
                    samples = []
                    all_idle()
                    for repeat in range(5):
                        torch.cuda.reset_peak_memory_stats()
                        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                            for _ in range(3): compress_variant(x, size, score, p, m, mode, keys=keys, forecast=forecast)
                            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                            start.record()
                            for _ in range(20): compress_variant(x, size, score, p, m, mode, keys=keys, forecast=forecast)
                            end.record(); end.synchronize()
                        samples.append(start.elapsed_time(end) / 20)
                        memory[mode] = torch.cuda.max_memory_allocated() / 2**30
                    times[mode] = samples
                    if memory[mode] > 14: raise RuntimeError('microbenchmark bs32 lacks headroom')
                import numpy as np
                rows = [dict(backbone=arm['backbone'], name=arm['name'], block=block, mode=mode,
                             ms=float(np.median(times[mode])), extra_ms=float(np.median(times[mode]) - np.median(times['M0'])),
                             samples=times[mode], m0_samples=times['M0'], batch=32, peak_gib=memory[mode],
                             precision='FP32 residual and size under BF16 autocast',
                             scope='merge only; excludes key extraction and shared D score/vector construction')
                        for mode in args.rule['modes']]
                filename = f"{arm['backbone']}_{config_hash(arm['baseline'])}_{block}.json"
                save(Path(args.root) / 'microbench' / filename, dict(rule_sha=args.sha, rows=rows))
                del x, size, score, keys, forecast
            n = m


def report(args):
    rows, costs = [], []
    for path in (Path(args.root) / 'microbench').glob('*.json'):
        doc = load_json(path)
        if doc['rule_sha'] != args.sha: raise RuntimeError('microbenchmark SHA mismatch')
        costs.extend(doc['rows'])
    for arm in args.rule['arms']:
        base, base_half = baseline(args, arm, 'val'), baseline(args, arm, 'half')
        for mode, cfg in arm['variants'].items():
            if arm['has_merge']:
                arr = cached_arrays(args.root, arm['backbone'], cfg, 'val', 50000, batch=arm['evaluation_batch'], rule_sha=args.sha)
                half = cached_arrays(args.root, arm['backbone'], cfg, 'half', 5000, batch=arm['evaluation_batch'], rule_sha=args.sha)
            else:
                arr, half = base, base_half
            rows.append(paired_row(arm, mode, base, arr, half))
    result = summarize(args.rule, rows, costs)
    save(Path(args.root) / 'stage1_decision.json', dict(rule_sha=args.sha, **result))
    lines = ['# Merge variants — stage 1', '', f"Rule SHA256: `{args.sha}`.", '',
             args.rule['disclosure'], '', 'Operator conventions:']
    lines += [f'- {k}: {v}' for k, v in args.rule['conventions'].items()]
    lines += ['', '| Group | Backbone/schedule | Mode | Δval (% points) | KL half | KL val | Holm |',
              '|---|---|---|---:|---:|---:|---:|']
    for row in result['rows']:
        delta = '=' if row['equal'] else f"{row['delta']:+.3f}"
        lines.append(f"| {row['group']} | {row['backbone']}/{row['name']} | {row['mode']} | {delta} | {row['calib_kl']:.5f} | {row['val_kl']:.5f} | {row['holm']:.5g} |")
    lines += ['', '| Mode | Δweak | Δstrong | Max Δms/station | Passed |', '|---|---:|---:|---:|---|']
    for row in result['variants']:
        lines.append(f"| {row['mode']} | {row['weak']:+.3f} | {row['strong']:+.3f} | {row['max_extra_ms']:.4f} | {row['passed']} |")
    lines += ['', f"Decision: {result['decision']}; selected: {', '.join(result['selected']) or 'none passed'}.", '',
              'The cost measures only the merge operation; it excludes key extraction and the shared construction of vector D. ',
              'Stage 2 runs only after the user reviews the report and approves.']
    path = Path('results/reports/merge-variants-2026-10-16.md')
    path.write_text('\n'.join(lines) + '\n')
    taxonomy_status('Stage 1 complete; ' + result['decision'], report=True)


def taxonomy_status(status, report=False):
    path = Path('results/reports/REPORT_TAXONOMY.md')
    prefix = '| [merge-variants-plan-2026-10-15.md]'
    link = '[merge-variants-2026-10-16.md](merge-variants-2026-10-16.md)' if report else '[Progress](../state/merge-variants-progress.md)'
    row = prefix + '(../plan/merge-variants-plan-2026-10-15.md) | New merge on locked schedules; selected on val, V2 reserved for the final pipeline. | ' + status + ' | ' + link + ' |'
    lines = path.read_text().splitlines()
    existing = next((i for i, line in enumerate(lines) if line.startswith(prefix)), None)
    if existing is None:
        at = next(i for i, line in enumerate(lines) if line.startswith('### G2.'))
        while not lines[at - 1].startswith('|'): at -= 1
        lines.insert(at, row)
    else: lines[existing] = row
    path.write_text('\n'.join(lines) + '\n')


def scheduler(args):
    # One scheduler per root, including across restarts.
    import fcntl
    lock = (Path(args.root) / 'scheduler.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    save(Path(args.root) / 'status.json', dict(state='running', pid=os.getpid(), rule_sha=args.sha))
    names = sorted({a['backbone'] for a in args.rule['arms']})
    children = []
    base = [sys.executable, '-u', __file__]
    common = ['--root', args.root, '--baseline-root', args.baseline_root,
              '--data-cache', args.data_cache, '--teacher-cache', args.teacher_cache]
    for gpu in (0, 1):
        log = (Path(args.root) / f'worker_gpu{gpu}.log').open('a')
        child = subprocess.Popen(base + ['worker'] + common + ['--backbones'] + names[gpu::2],
                                 env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)), stdout=log, stderr=subprocess.STDOUT)
        children.append((child, log))
    try:
        codes = [child.wait() for child, _ in children]
    finally:
        for child, log in children:
            if child.poll() is None: child.terminate()
            log.close()
    try:
        if any(codes): raise RuntimeError(f'worker failed: {codes}; restart launch after fixing cause')
        save(Path(args.root) / 'status.json', dict(state='microbenchmark', pid=os.getpid(), rule_sha=args.sha))
        subprocess.run(base + ['benchmark'] + common, env=dict(os.environ, CUDA_VISIBLE_DEVICES='0'), check=True)
        report(args)
    except Exception as exc:
        save(Path(args.root) / 'status.json', dict(state='failed', pid=os.getpid(), rule_sha=args.sha, reason=str(exc)))
        raise
    save(Path(args.root) / 'status.json', dict(state='complete', pid=os.getpid(), rule_sha=args.sha))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['design', 'seal', 'launch', 'scheduler', 'worker', 'benchmark', 'report'])
    parser.add_argument('--root', default=str(ROOT))
    parser.add_argument('--baseline-root', default='results/leaf_m0')
    parser.add_argument('--data-cache', default='data/cache')
    parser.add_argument('--teacher-cache', default=None)
    parser.add_argument('--include-mae-l', action='store_true')
    parser.add_argument('--exclude-n2', action='store_true', help='defer N2 and its combinations while formula is pending')
    parser.add_argument('--backbones', nargs='+', default=[])
    args = parser.parse_args()
    args.teacher_cache = args.teacher_cache or str(Path(args.baseline_root) / 'teachers')
    if args.action == 'design':
        doc = design(include_mae_l=args.include_mae_l, exclude_n2=args.exclude_n2)
        print(json.dumps(doc, indent=1)); return
    if args.action == 'seal':
        print(seal(args.root, args.include_mae_l, args.baseline_root, args.exclude_n2)['sha256']); return
    sealed = read_rule(args.root); args.rule, args.sha = sealed['rule'], sealed['sha256']
    assert_ready(args.rule)
    if args.action == 'launch':
        log = (Path(args.root) / 'scheduler.log').open('a')
        command = [sys.executable, '-u', __file__, 'scheduler', '--root', args.root, '--baseline-root', args.baseline_root,
                   '--data-cache', args.data_cache, '--teacher-cache', args.teacher_cache]
        p = subprocess.Popen(command, start_new_session=True, stdout=log, stderr=subprocess.STDOUT)
        log.close(); print(f'setsid scheduler pid={p.pid}; restart with the same launch command'); return
    if args.action == 'scheduler': scheduler(args)
    elif args.action == 'worker': worker(args)
    elif args.action == 'benchmark': benchmark(args)
    elif args.action == 'report': report(args)


if __name__ == '__main__':
    main()
