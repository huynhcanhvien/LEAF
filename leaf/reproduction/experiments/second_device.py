"""Second accelerator: the latency-pricing test (latency_pricing.py) repeated on another GPU, e.g. an NVIDIA T4.

Same backbones (DeiT-S/B, ViT-B MAE), the same workload set, the same historical M3 probes and the same DP as the
pricing study; only the device changes and SALT is refitted on its timings. GMAC-planned schedules are device-free
and identical to those of the pricing study, so their accuracy is reused. Regimes: batch 4 and 32 on the device.
Turing GPUs (T4, sm_75) have no BF16: run the device stages with LEAF_DTYPE=float16 (latency only; accuracy is always
evaluated in BF16 on the home GPUs).

Stages (results/second_device/):
  kit             (home, CPU)  copy the workload set and runner binding of the pricing study -> kit/ (to upload)
  check           (device)     smoke test: every workload kind builds and runs in LEAF_DTYPE, Triton kernels included
  time_workloads  (device)     three interleaved rounds of every workload at batch 4 and 32 -> <bb>/timing_gpu/
  plan            (home, CPU)  per regime: grouped-CV fit of the SALT formulas (latency_pricing.analyze_regime), then
                               SALT-planned schedules at 0.97x the device latency of every DiffRate anchor, and the
                               GMAC-planned grid of the pricing study -> <bb>/plans.json (sealed)
  worker          (home, GPU)  accuracy (confirm, val, V2) of every planned arm; one process per card, cached reuse
  time_final      (device)     one session: every planned arm and the dense model, batch 4 and 32, three rounds
  compare         (home, CPU)  SALT vs GMAC planning at equal measured latency on the device, McNemar + Holm
  pack / unpack   (both)       move the campaign's JSON files between the device and home as one .tar.gz

Device run (Kaggle, GPU T4 x1):  bash scripts/run_second_device.sh device-1   ... later  device-2
Home run:                         bash scripts/run_second_device.sh home-1     ... later  home-2

Transfer mode (one device session, about an hour on a T4; no re-planning):
  time_transfer   (device)     the sealed arms of the pricing study (SALT-planned on the home GPU and CPU, the
                               GMAC-planned grid, the DiffRate anchors, dense), batch 4 and 32, three rounds
                               -> <bb>/transfer_cuda/ ; accuracy of every arm already exists at home
  compare_transfer (home, CPU) SALT-planned (home device) vs GMAC-planned at equal measured device latency,
                               McNemar + Holm on the pricing study's per-image results -> compare_transfer.json
Device: bash scripts/run_second_device.sh device-transfer ; home: bash scripts/run_second_device.sh home-transfer <archive>
"""
import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
import numpy as np

import leaf.reproduction.experiments.latency_pricing as lp

ROOT = REPO / 'results/second_device'
PRICING = REPO / 'results/latency_pricing'
BBS = lp.BBS
REGIMES = {'dev_bs4': ('cuda', 4), 'dev_bs32': ('cuda', 32)}


def card_idle():
    """latency_pricing.card_idle, except that processes whose PID is not visible here are ignored: inside a container
    (Kaggle) nvidia-smi may list this very process under its host PID. Outside containers every PID is visible, so the
    check is unchanged."""
    index = os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0]
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,pci.bus_id', '--format=csv,noheader'], text=True)
    bus = dict(r.split(', ') for r in rows.strip().splitlines())[index]
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_bus_id', '--format=csv,noheader'], text=True)
    pids = [int(r.split(', ')[0]) for r in apps.strip().splitlines() if r and r.split(', ')[1] == bus]
    return not [p for p in pids if p != os.getpid() and os.path.exists(f'/proc/{p}')]


def bind():
    """Point the pricing campaign's functions at this campaign's root and regimes."""
    lp.ROOT, lp.REGIMES, lp.card_idle = ROOT, REGIMES, card_idle
    base = lp.runner

    def runner(bb):                     # accuracy of arms shared with the pricing study is reused from its cache
        r = base(bb); r.reuse_roots = list(getattr(r, 'reuse_roots', [])) + [PRICING.resolve() / bb]
        return r
    lp.runner = runner


def kit():
    for bb in BBS:
        shutil.copy2(PRICING / bb / 'workload.json', (ROOT / bb).mkdir(parents=True, exist_ok=True) or ROOT / bb / 'workload.json')
        shutil.copy2(PRICING / bb / 'plans.json', ROOT / bb / 'pricing_plans.json')
    shutil.copy2(PRICING / 'design.json', ROOT / 'design.json')
    print('kit ready in', ROOT)


def device_runner(bb):
    """Model building only (no data): the pricing study's runner binding, weights fetched by timm."""
    from leaf.reproduction.experiments.low_budget_and_factorial import LowRunner
    return LowRunner(bb, ROOT, lp.load(ROOT / 'design.json'))


def device_info():
    import torch
    from leaf.latency.timing import INFERENCE_DTYPE
    return dict(device=torch.cuda.get_device_name(0), capability=list(torch.cuda.get_device_capability(0)),
                dtype=str(INFERENCE_DTYPE), torch=torch.__version__, cuda=torch.version.cuda,
                driver=subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                                      capture_output=True, text=True).stdout.strip())


def check():
    """Every workload kind builds and runs a CUDA-graph forward in LEAF_DTYPE; D kernels against the eager forecast."""
    import torch
    from leaf.latency.timing import measure_latency
    out = dict(info=device_info(), kinds={})
    for bb in BBS:
        r = device_runner(bb); jobs = lp.inventory(bb)['jobs']
        for job in jobs:
            kind = job['config']['kind'] + ('+D' if job['config'].get('k') else '')
            if kind == 'dense': kind = f'dense/{bb}'       # every backbone once: its weights are fetched here
            if kind in out['kinds']: continue
            try:
                m = r.inference_model(job['config']); meas = measure_latency(m, 4, iters=3, warmup=2)
                out['kinds'][kind] = dict(ok=True, backbone=bb, ms=meas['ms_mean'])
            except Exception as e:
                out['kinds'][kind] = dict(ok=False, backbone=bb, error=repr(e)[:500])
            print(kind, out['kinds'][kind], flush=True); gc.collect(); torch.cuda.empty_cache()
    # D kernel (Triton) against the PyTorch forecast on one station, in the device dtype.
    try:
        from leaf.forecast.kernels.alpha_score import zscored_alpha
        bb = 'deit_s'; r = device_runner(bb); job = next(j for j in lp.inventory(bb)['jobs'] if j['config'].get('k'))
        m = r.inference_model(job['config']); key = next(iter(m.future)); fut = m.future[key]
        x = torch.randn(4, 197, r.spec.dim, device='cuda'); size = torch.ones(4, 197, 1, device='cuda')
        with torch.no_grad():
            a = zscored_alpha(fut, x, size, None, 1.); fut.triton = False; b = zscored_alpha(fut, x, size, None, 1.)
        rho = float(torch.corrcoef(torch.stack([a[:, 1:].flatten(), b[:, 1:].flatten()]))[0, 1])
        out['d_kernel'] = dict(pearson=rho, ok=rho > .99)
    except Exception as e:
        out['d_kernel'] = dict(ok=False, error=repr(e)[:500])
    print('D kernel', out['d_kernel'], flush=True)
    lp.save(ROOT / 'device_check.json', out)
    bad = [k for k, v in out['kinds'].items() if not v['ok']] + ([] if out['d_kernel'].get('ok') else ['D kernel'])
    if bad: raise SystemExit(f'device check failed for {bad}; see {ROOT / "device_check.json"}')


def time_workloads():
    """latency_pricing.gpu_time on the device: dense, our family (for the fit) and the DiffRate anchors (budgets)."""
    import torch
    from leaf.reproduction.runners.exclusive_timing import cast_exclusive
    for bb in BBS:
        done = ROOT / bb / 'gpu_done.json'
        if done.exists(): continue
        doc = lp.inventory(bb); r = device_runner(bb)
        jobs = [j for j in doc['jobs'] if j['config']['kind'] in ('pd', 'pdm', 'dense') or any(n.startswith('baseline/DR') for n in j['names'])]
        batches = [b for _, b in REGIMES.values()]; dest = ROOT / bb / 'timing_gpu'
        if dest.exists(): dest.rename(ROOT / bb / f'timing_gpu_interrupted_{time.time_ns()}')
        for rnd in range(lp.ROUNDS):
            for job in (jobs if rnd % 2 == 0 else jobs[::-1]):
                if not lp.card_idle(): raise RuntimeError('another process appeared on the timed card; session discarded')
                try:
                    m = r.inference_model(job['config']); meas = {str(b): cast_exclusive(m, b, iters=50, warmup=10) for b in batches}
                except torch.cuda.OutOfMemoryError:
                    meas = dict(status='OOM')
                except Exception as e:
                    meas = dict(status='error', error=repr(e))
                lp.save(dest / f"{job['id']}#{rnd}.json", dict(id=job['id'], round=rnd, measurements=meas, clean=lp.card_idle(),
                                                               time=time.time(), **device_info()))
                m = None; gc.collect(); torch.cuda.empty_cache()
            print(bb, 'round', rnd, 'done', flush=True)
        lp.save(done, dict(rounds=lp.ROUNDS, batches=batches, n=len(jobs), time=time.time(), **device_info()))


def plan():
    summary = {}
    for bb in BBS:
        for regime in REGIMES:
            res = lp.analyze_regime(bb, regime); lp.save(ROOT / bb / f'analysis_{regime}.json', res)
            summary[f'{bb}/{regime}'] = res['gate'] | dict(n=res['n'], dense_ms=res['dense_ms'])
            g = res['gate']
            print(f"{bb} {regime} n={res['n']} dense={res['dense_ms']:.2f}ms GMAC MAPE {g['gmac_mape']:.2f}% "
                  f"best {g['best_formula']} {g['best_mape']:.2f}%", flush=True)
    lp.save(ROOT / 'analysis_summary.json', summary)
    for bb in BBS: lp.plan(bb, sorted(REGIMES))       # every regime is planned: the device test is pre-declared


def time_final():
    for bb in BBS: lp.final_time(bb, sorted(REGIMES), devices=('cuda',)); print('final timing done', bb, flush=True)
    lp.save(ROOT / 'final_device.json', device_info())


def transfer_tasks(bb):
    """{task id: names} of the sealed pricing arms timed in transfer mode (the other baselines are left out)."""
    out = {}
    for name, row in lp.load(ROOT / bb / 'pricing_plans.json')['jobs'].items():
        if 'config' not in row or not (row.get('group') in ('salt', 'gmac') or name in ('ref/dense',) or name.startswith('ref/baseline/DR')):
            continue
        out.setdefault(bb + '-' + row['config_sha'], dict(config=row['config'], names=[]))['names'].append(name)
    return out


def time_transfer():
    import torch
    from leaf.reproduction.runners.exclusive_timing import cast_exclusive
    batches = [b for _, b in REGIMES.values()]
    for bb in BBS:
        done = ROOT / bb / 'transfer_done.json'
        if done.exists(): continue
        ts = list(transfer_tasks(bb).items()); r = device_runner(bb); dest = ROOT / bb / 'transfer_cuda'
        if dest.exists(): dest.rename(ROOT / bb / f'transfer_cuda_interrupted_{time.time_ns()}')
        started = time.time()
        for rnd in range(lp.ROUNDS):
            for tid, t in (ts if rnd % 2 == 0 else ts[::-1]):
                if not lp.card_idle(): raise RuntimeError('another process appeared on the timed card; session discarded')
                try:
                    m = r.inference_model(t['config']); meas = {str(b): cast_exclusive(m, b, iters=50, warmup=10) for b in batches}
                except Exception as e:
                    meas = dict(status='error', error=repr(e))
                lp.save(dest / f'{tid}#{rnd}.json', dict(id=tid, names=t['names'], round=rnd, measurements=meas,
                                                       clean=lp.card_idle(), time=time.time()))
                m = None; gc.collect(); torch.cuda.empty_cache()
            print(bb, 'round', rnd, 'done', f'{(time.time() - started) / 60:.1f} min', flush=True)
        lp.save(done, dict(rounds=lp.ROUNDS, batches=batches, n=len(ts), time=time.time(), **device_info()))


def transfer_ms(bb, bs):
    rows = {}
    for p in sorted((ROOT / bb / 'transfer_cuda').glob('*.json')):
        rec = lp.load(p); m = rec['measurements'].get(str(bs)) if isinstance(rec['measurements'], dict) else None
        if isinstance(m, dict) and 'ms_mean' in m: rows.setdefault(rec['id'], []).append(m['ms_mean'])
    return {k: (float(np.mean(v)), float(np.std(v, ddof=1))) for k, v in rows.items() if len(v) == lp.ROUNDS}


def compare_transfer():
    """Pricing pairing rule on the device: each SALT arm against the nearest GMAC arm within tau."""
    from leaf.evaluation.data import mcnemar, holm
    tests = []; dense = {}
    for bb in BBS:
        jobs = lp.load(ROOT / bb / 'pricing_plans.json')['jobs']
        for regime, (_, bs) in REGIMES.items():
            ms = transfer_ms(bb, bs); sid = lambda j: bb + '-' + j['config_sha']
            dense[f'{bb}/{regime}'] = ms.get(sid(jobs['ref/dense']), (None,))[0]
            gm = [(n, j) for n, j in jobs.items() if j.get('group') == 'gmac' and 'config' in j and sid(j) in ms]
            for n, j in jobs.items():
                if j.get('group') != 'salt' or 'config' not in j or sid(j) not in ms: continue
                a = ms[sid(j)]; best = None
                for gn, g in gm:
                    b = ms[sid(g)]; tau = max(.0025 * a[0], 2 * np.sqrt((a[1] ** 2 + b[1] ** 2) / 2)); gap = abs(a[0] - b[0])
                    if gap <= tau and (best is None or gap < best[0]): best = (gap, gn, tau)
                row = dict(backbone=bb, regime=regime, salt=n, planned_for=j['regime'], salt_ms=a[0], matched=best is not None)
                if best:
                    g = jobs[best[1]]; row.update(gmac=best[1], gmac_ms=ms[sid(g)][0], tau=best[2], same_config=j['config_sha'] == g['config_sha'])
                    for split in ('val', 'v2'):
                        ca = np.load(PRICING / bb / split / 'runs' / f"{j['config_sha']}.npz")['correct']
                        cb = np.load(PRICING / bb / split / 'runs' / f"{g['config_sha']}.npz")['correct']
                        p, *_ = mcnemar(ca, cb); row[split] = dict(delta=float(100 * (ca.mean() - cb.mean())), p=p)
                tests.append(row)
    for split in ('val', 'v2'): holm([t[split] for t in tests if t['matched'] and not t['same_config']])
    info = lp.load(ROOT / BBS[0] / 'transfer_done.json')
    lp.save(ROOT / 'compare_transfer.json', dict(device=info, dense_ms=dense, tests=tests, time=time.time()))
    m = [t for t in tests if t['matched'] and not t['same_config']]
    print(info['device'], info['dtype'], '| matched', len(m), 'of', len(tests), 'identical', sum(t.get('same_config', False) for t in tests))
    for t in m:
        print(t['backbone'], t['regime'], t['salt'], 'vs', t['gmac'], f"{t['salt_ms']:.2f}/{t['gmac_ms']:.2f} ms",
              f"val {t['val']['delta']:+.2f} (holm {t['val']['holm']:.3g})  v2 {t['v2']['delta']:+.2f} (holm {t['v2']['holm']:.3g})")


def pack(name):
    """Every JSON of the campaign except per-config caches and claims -> <name>.tar.gz in the repository root."""
    dest = REPO / f'{name}.tar.gz'
    with tarfile.open(dest, 'w:gz') as tar:
        for p in sorted(ROOT.rglob('*.json')):
            if not {'claims', 'runs', 'completed'} & set(p.relative_to(ROOT).parts): tar.add(p, arcname=str(p.relative_to(REPO)))
    print('wrote', dest, f'{dest.stat().st_size / 1e6:.1f} MB')


def unpack(path):
    """Extract a tarball written by pack(); files already here are kept unless identical (sealed files never change)."""
    with tarfile.open(path) as tar:
        for m in tar.getmembers():
            if not m.name.startswith('results/second_device/') or '..' in Path(m.name).parts: raise SystemExit(f'unexpected member {m.name}')
            dest = REPO / m.name; data = tar.extractfile(m).read()
            if dest.exists() and dest.read_bytes() != data and dest.name in ('plans.json', 'pricing_plans.json', 'design.json', 'workload.json'):
                raise SystemExit(f'{dest} differs from the archive; refusing to overwrite a sealed file')
            dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(data)
    print('unpacked', path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=['kit', 'check', 'time_workloads', 'plan', 'worker', 'time_final', 'compare', 'time_transfer', 'compare_transfer', 'pack', 'unpack'])
    ap.add_argument('--name', default='second_device', help='pack: archive name (without .tar.gz)')
    ap.add_argument('--archive', help='unpack: archive path')
    a = ap.parse_args(); bind()
    if a.stage == 'kit': kit()
    elif a.stage == 'check': check()
    elif a.stage == 'time_workloads': time_workloads()
    elif a.stage == 'plan': plan()
    elif a.stage == 'worker': lp.worker()
    elif a.stage == 'time_final': time_final()
    elif a.stage == 'compare': lp.compare(sorted(REGIMES))
    elif a.stage == 'time_transfer': time_transfer()
    elif a.stage == 'compare_transfer': compare_transfer()
    elif a.stage == 'pack': pack(a.name)
    else: unpack(a.archive)


if __name__ == '__main__':
    main()
