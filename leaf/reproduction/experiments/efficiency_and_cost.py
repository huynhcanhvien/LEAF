"""Efficiency and search-cost measurements for the paper (Tab. efficiency, Tab. search cost, Tab. latency pricing).

Stages (results/efficiency_and_cost/):
  probe_rate (GPU)   wall-clock of single-station probes: PROBE_SAMPLE probes spread over the paper's grid of every
                     backbone, run in a fresh cache (forward passes over the 5,000 probe images, batch 32); extrapolated
                     to the full grid. One process per card: `--gpu 0|1` takes every other backbone.
  plan_time (CPU)    wall-clock of `leaf plan` (price search and rescaling) for every released latency budget of every
                     backbone, from the released probes, head groups and SALT model in schedules/leaf/planning/.
  overheads (GPU/CPU) time spent in stations and in the forecast D, from kernel times of the profiler (eager mode, one
                     batch per forward, PROFILE_ITERS forwards after warm-up), and the peak memory of one forward beyond
                     the weights; rows of the efficiency table (GPU batch 32 and 4, CPU batch 1).
  time_bs32 (GPU0)   one exclusive timing session at batch 32 of the latency-planned (SALT) and GMAC-planned schedules of
                     the final method and the dense model on DeiT-S/B and ViT-B (MAE), three interleaved rounds (middle
                     round reversed), as in the main comparison; nothing else on the machine.
  report (CPU)       pairs SALT- vs GMAC-planned schedules at equal measured latency (tau = max(0.05 ms, 2 SD)), McNemar
                     with Holm over the pairs of a split on the stored per-image predictions; writes report.json.

Accuracy needs no new evaluation: every schedule here was evaluated by its campaign (per-image arrays found by config key).
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
import numpy as np

ROOT = REPO / 'results/efficiency_and_cost'
BBS = ('deit_t', 'deit_s', 'deit_b', 'augreg_s', 'augreg_b', 'mae_b', 'mae_l')
PRICING_BBS = ('deit_s', 'deit_b', 'mae_b')
PROBE_SAMPLE = 24
PROFILE_ITERS = 20
GRID_THETAS, GRID_ALPHAS = (0., .5, 1.), (0., 1 / 4, 1 / 3, 1 / 2, 2 / 3, 3 / 4, 1.)   # the paper's grids
ROUNDS = 3
# Rows of the efficiency table: (backbone, label, schedule-book key); 'dense' is the unreduced model.
EFFICIENCY_ROWS = [
    ('deit_b', 'Dense', 'dense'), ('deit_b', 'DiffRate', 'diffrate/DR8.7'), ('deit_b', 'LEAF', 'leaf/gmac/DR8.7'),
    ('deit_b', 'DiffRate', 'diffrate/DR11.5'), ('deit_b', 'LEAF', 'leaf/gmac/DR11.5'),
    ('mae_b', 'Dense', 'dense'), ('mae_b', 'DiffRate', 'diffrate/DR8.7'), ('mae_b', 'LEAF', 'leaf/gmac/DR8.7'),
    ('mae_b', 'DiffRate', 'diffrate/DR11.5'), ('mae_b', 'LEAF', 'leaf/gmac/DR11.5'),
    ('mae_l', 'Dense', 'dense'), ('mae_l', 'DiffRate', 'diffrate/DR31.0'), ('mae_l', 'LEAF', 'leaf/gmac/DR31.0'),
]
# Other regimes (DeiT-B): the SALT-planned schedule of the latency-pricing study at the 8.7-GMAC anchor.
REGIME_ROWS = [('gpu_bs4', 'cuda', 4, 'salt/gpu_bs4/DR8.7'), ('cpu_bs1', 'cpu', 1, 'salt/cpu_bs1/DR8.7')]


def load(p): return json.loads(Path(p).read_text())


def save(p, obj):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix('.tmp'); tmp.write_text(json.dumps(obj, indent=1)); os.replace(tmp, p)


def book(bb): return load(REPO / 'assets/schedules' / f'{bb}.json')


def config_of(bb, key):
    if key == 'dense': return dict(kind='dense')
    s = book(bb)['schedules']
    if key in s: return s[key]['config']
    hits = [k for k in s if k.endswith(key)]          # e.g. 'main/diffrate/DR8.7'
    if len(hits) != 1: raise KeyError(f'{bb}: {key} matches {hits}')
    return s[hits[0]]['config']


def runner(bb, root=None):
    from leaf.api import LeafRunner
    return LeafRunner(bb, root=root or ROOT / 'scratch' / bb, batch=32)


# ---------------------------------------------------------------- probe rate
def probe_rate(gpu):
    import torch
    from leaf.backbones import SPECS
    from leaf.cli import default_deltas, forecast_config, probe_jobs
    for bb in BBS[gpu::2]:
        dest = ROOT / 'probe_rate' / f'{bb}.json'
        if dest.exists(): continue
        spec = SPECS[bb]
        jobs = list(probe_jobs(spec, forecast_config(REPO / 'schedules/leaf/planning' / bb / 'groups.json', 16, 32),
                               GRID_THETAS, GRID_ALPHAS, default_deltas(spec)))
        sample = [jobs[i] for i in np.linspace(0, len(jobs) - 1, PROBE_SAMPLE).round().astype(int)]
        root = ROOT / 'scratch' / f'probe_{bb}_{time.time_ns()}'
        r = runner(bb, root)
        r.run(dict(kind='dense'), 'half')                  # data, teacher and weights loaded outside the clock
        secs = []
        for job in sample:
            torch.cuda.synchronize(); t0 = time.monotonic()
            r.run(job['config'], 'half'); torch.cuda.synchronize(); secs.append(time.monotonic() - t0)
        s = float(np.mean(secs))
        save(dest, dict(backbone=bb, device=torch.cuda.get_device_name(0), grid=len(jobs), sample=len(sample),
                        seconds=secs, mean_s=s, full_grid_gpu_h=s * len(jobs) / 3600, time=time.time()))
        print(bb, f'{s:.2f} s/probe, {len(jobs)} probes -> {s * len(jobs) / 3600:.2f} GPU-h', flush=True)
        subprocess.run(['rm', '-rf', str(root)])


# ---------------------------------------------------------------- planning time
def plan_time():
    for bb in BBS:
        dest = ROOT / 'plan_time' / f'{bb}.json'
        if dest.exists(): continue
        p = REPO / 'schedules/leaf/planning' / bb
        out = ROOT / 'scratch' / f'plan_{bb}.json'
        t0 = time.monotonic()
        subprocess.run([sys.executable, '-m', 'leaf', 'plan', '--backbone', bb, '--probes', str(p / 'probes.json'),
                        '--groups', str(p / 'groups.json'), '--salt', str(p / 'salt.json'), '--budgets', str(p / 'budgets.json'),
                        '--out', str(out)], check=True, cwd=REPO, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
        wall = time.monotonic() - t0
        per = [v['planner']['seconds'] for v in load(out)['schedules'].values() if 'planner' in v]
        save(dest, dict(backbone=bb, budgets=len(per), seconds=per, mean_s=float(np.mean(per)), process_s=wall, time=time.time()))
        print(bb, f'{len(per)} budgets, {np.mean(per):.2f} s per budget (process {wall:.1f} s)', flush=True)


# ---------------------------------------------------------------- overheads
def instrument(model):
    """Wrap scoring and token compression in profiler ranges: 'leaf_forecast' around the score at stations with D,
    'leaf_station' around the score elsewhere and around every compression (sort, gather, merge)."""
    import torch
    import leaf.model.mlerp_vit as mv
    import leaf.model.reduced_vit as rv
    score = model.score

    def scored(l, x, size, raw_p):
        key = str(l)
        uses_d = key in getattr(model, 'future', {}) and getattr(model, 'alpha', {key: 1.}).get(key, 0.) > 0
        with torch.profiler.record_function('leaf_forecast' if uses_d else 'leaf_station'):
            return score(l, x, size, raw_p)
    model.score = scored
    for mod, name in ((rv, 'compress'), (mv, 'compress_norm')):
        f = getattr(mod, name)
        if getattr(f, '_leaf_wrapped', False): continue

        def wrapped(*a, _f=f, **k):
            with torch.profiler.record_function('leaf_station'): return _f(*a, **k)
        wrapped._leaf_wrapped = True; setattr(mod, name, wrapped)
    return model


def profile_row(r, cfg, device, batch):
    import torch
    from torch.profiler import ProfilerActivity, profile
    from leaf.evaluation.determinism import native
    model = r.inference_model(cfg)
    if device == 'cpu':
        from leaf.latency.cpu_timing import cpu_inference_copy, forward_cpu
        model = cpu_inference_copy(model)
        run = lambda x: forward_cpu(model, x)
    else:
        run = lambda x: model(x)
    if cfg['kind'] != 'dense': instrument(model)
    x = torch.randn(batch, 3, 224, 224, device=device)
    with native(), torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
        for _ in range(5): run(x)
        peak = None
        if device == 'cuda':
            torch.cuda.synchronize(); base = torch.cuda.memory_allocated(); torch.cuda.reset_peak_memory_stats()
            run(x); torch.cuda.synchronize(); peak = (torch.cuda.max_memory_allocated() - base) / 2 ** 20
        acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device == 'cuda' else [])
        with profile(activities=acts) as prof:
            for _ in range(PROFILE_ITERS): run(x)
            if device == 'cuda': torch.cuda.synchronize()
    ev = prof.events()
    dev = lambda e: getattr(e, 'device_time_total', None) or getattr(e, 'cuda_time_total', 0)
    if device == 'cuda':
        from torch.autograd import DeviceType
        total = sum(e.self_device_time_total for e in ev if e.device_type == DeviceType.CUDA)   # kernels only
        part = lambda n: sum(dev(e) for e in ev if e.name == n)
    else:
        total = sum(e.cpu_time_total for e in ev if e.name.startswith('ProfilerStep') or e.cpu_parent is None)
        part = lambda n: sum(e.cpu_time_total for e in ev if e.name == n)
    ms = lambda us: us / 1e3 / PROFILE_ITERS
    return dict(total_ms=ms(total), station_ms=ms(part('leaf_station')), forecast_ms=ms(part('leaf_forecast')),
                peak_activation_mb=peak, profile_iters=PROFILE_ITERS)


def overheads(device):
    import torch
    rows = []
    if device == 'cuda':
        for bb, label, key in EFFICIENCY_ROWS:
            r = runner(bb); cfg = config_of(bb, key)
            rows.append(dict(backbone=bb, label=label, key=key, batch=32, device='cuda', **profile_row(r, cfg, 'cuda', 32)))
            print(rows[-1], flush=True)
        for regime, dev, batch, key in REGIME_ROWS:
            if dev != 'cuda': continue
            r = runner('deit_b'); jobs = load(REPO / 'results/latency_pricing/deit_b/plans.json')['jobs']
            for k, cfg in (('dense', dict(kind='dense')), (key, jobs[key]['config'])):
                rows.append(dict(backbone='deit_b', label=k, key=k, regime=regime, batch=batch, device='cuda', **profile_row(r, cfg, 'cuda', batch)))
                print(rows[-1], flush=True)
        save(ROOT / 'overheads_cuda.json', dict(rows=rows, device=torch.cuda.get_device_name(0), time=time.time()))
    else:
        from leaf.latency.cpu_timing import pin
        info = pin(tuple(range(8, 16)))
        from leaf.api import LeafRunner
        r = LeafRunner('deit_b', root=ROOT / 'scratch' / 'deit_b_cpu', batch=32)
        jobs = load(REPO / 'results/latency_pricing/deit_b/plans.json')['jobs']
        for regime, dev, batch, key in REGIME_ROWS:
            if dev != 'cpu': continue
            for k, cfg in (('dense', dict(kind='dense')), (key, jobs[key]['config'])):
                rows.append(dict(backbone='deit_b', label=k, key=k, regime=regime, batch=batch, device='cpu', **cpu_profile(r, cfg, batch)))
                print(rows[-1], flush=True)
        save(ROOT / 'overheads_cpu.json', dict(rows=rows, pin=info, time=time.time()))


def cpu_profile(r, cfg, batch):
    """CPU rows: the backbone and wrapper on the CPU (as in the latency-pricing study), profiler CPU times."""
    import torch
    from torch.profiler import ProfilerActivity, profile
    from leaf.evaluation.runner import build
    from leaf.latency.cpu_timing import cpu_inference_copy, forward_cpu
    from leaf.backbones import load as load_backbone
    from leaf.evaluation.determinism import native
    if not hasattr(r, '_cpu_vit'): r._cpu_vit, r._cpu_spec = load_backbone('deit_b', 'cpu')
    from leaf.api import build_model
    from leaf.forecast.cpu_forecast import enable_cpu_lowrank
    model = cpu_inference_copy(build_model(r._cpu_vit, r._cpu_spec, cfg))
    enable_cpu_lowrank(model)              # low-rank D on the CPU, as in latency_pricing.cpu_build
    if cfg['kind'] != 'dense': instrument(model)
    x = torch.randn(batch, 3, 224, 224)
    with native(), torch.no_grad():
        for _ in range(5): forward_cpu(model, x)
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            t0 = time.perf_counter()
            for _ in range(PROFILE_ITERS):
                with torch.profiler.record_function('leaf_forward'): forward_cpu(model, x)
            wall = (time.perf_counter() - t0) * 1e3 / PROFILE_ITERS
    ev = prof.events(); part = lambda n: sum(e.cpu_time_total for e in ev if e.name == n) / 1e3 / PROFILE_ITERS
    return dict(total_ms=part('leaf_forward'), wall_ms_profiled=wall, station_ms=part('leaf_station'),
                forecast_ms=part('leaf_forecast'), peak_activation_mb=None, profile_iters=PROFILE_ITERS)


# ---------------------------------------------------------------- batch-32 session: SALT vs GMAC planning
def bs32_tasks():
    from leaf.evaluation.runner import config_hash
    out = []
    for bb in PRICING_BBS:
        s = book(bb)['schedules']; seen = set()
        for name, e in [('dense', dict(config=dict(kind='dense')))] + [(k, v) for k, v in s.items() if k.startswith(('leaf/latency/', 'leaf/gmac/'))]:
            h = config_hash(e['config'])
            if h in seen: continue
            seen.add(h); out.append(dict(backbone=bb, name=name, config=e['config'], key=h))
    return out


def time_bs32():
    import torch
    from leaf.reproduction.runners.exclusive_timing import cast_exclusive
    from leaf.reproduction.experiments.latency_pricing import card_idle
    ts = bs32_tasks(); save(ROOT / 'bs32_tasks.json', dict(tasks=[dict(t) for t in ts], time=time.time()))
    for bb in PRICING_BBS:
        dest = ROOT / 'bs32' / bb
        if (dest / 'done.json').exists(): continue
        while not card_idle(): time.sleep(30)
        r = runner(bb); mine = [t for t in ts if t['backbone'] == bb]; rows = []
        for rnd in range(ROUNDS):
            for t in (mine if rnd % 2 == 0 else mine[::-1]):
                if not card_idle(): raise RuntimeError('timed card shared during timing; session discarded')
                m = r.inference_model(t['config'])
                meas = cast_exclusive(m, 32, iters=50, warmup=10)
                rows.append(dict(name=t['name'], key=t['key'], round=rnd, **meas)); del m; torch.cuda.empty_cache()
                print(bb, rnd, t['name'], meas['ms_mean'], flush=True)
        save(dest / 'rounds.json', dict(rows=rows, device=torch.cuda.get_device_name(0), time=time.time()))
        save(dest / 'done.json', dict(time=time.time()))


def correct(bb, key, split):
    for camp in ('main_comparison', 'ablation_and_alpha_grid', 'alpha_kappa_equivalence', 'latency_pricing'):
        p = REPO / 'results' / camp / bb / split / 'runs' / f'{key}.npz'
        if p.exists(): return np.load(p)['correct']
    raise FileNotFoundError(f'{bb} {key} {split}')


def report():
    from leaf.evaluation.data import mcnemar, holm
    out = dict(pairs=[], unpaired=[])
    for bb in PRICING_BBS:
        rows = load(ROOT / 'bs32' / bb / 'rounds.json')['rows']
        ms = {}
        for row in rows: ms.setdefault((row['name'], row['key']), []).append(row['ms_mean'])
        arms = {n: dict(key=k, ms=float(np.mean(v)), sd=float(np.std(v, ddof=1))) for (n, k), v in ms.items()}
        gm = {n: a for n, a in arms.items() if n.startswith('leaf/gmac/')}
        for n, a in arms.items():
            if not n.startswith('leaf/latency/'): continue
            best = None
            for gn, g in gm.items():
                tau = max(.05, 2 * np.sqrt((a['sd'] ** 2 + g['sd'] ** 2) / 2)); gap = abs(a['ms'] - g['ms'])
                if gap <= tau and (best is None or gap < best[0]): best = (gap, gn, tau)
            if best is None: out['unpaired'].append(dict(backbone=bb, salt=n, ms=a['ms'])); continue
            g = gm[best[1]]
            row = dict(backbone=bb, salt=n, gmac=best[1], salt_ms=a['ms'], gmac_ms=g['ms'], tau=best[2], same_config=a['key'] == g['key'])
            for split in ('val', 'v2'):
                ca, cb = correct(bb, a['key'], split), correct(bb, g['key'], split)
                p, _, _ = mcnemar(ca, cb); row[split] = dict(delta=float(100 * (ca.mean() - cb.mean())), p=p)
            out['pairs'].append(row)
        out.setdefault('dense', {})[bb] = arms['dense']
    for split in ('val', 'v2'): holm([t[split] for t in out['pairs'] if not t['same_config']])
    save(ROOT / 'bs32_report.json', out)
    for t in out['pairs']:
        print(t['backbone'], t['salt'], 'vs', t['gmac'], f"{t['salt_ms']:.2f}/{t['gmac_ms']:.2f}",
              f"val {t['val']['delta']:+.2f} (holm {t['val']['holm']:.3g}) v2 {t['v2']['delta']:+.2f} (holm {t['v2']['holm']:.3g})")
    print('unpaired', len(out['unpaired']))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=['probe_rate', 'plan_time', 'overheads', 'time_bs32', 'report'])
    ap.add_argument('--gpu', type=int, default=0, help='probe_rate: 0 or 1, every other backbone')
    ap.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    a = ap.parse_args()
    if a.stage == 'probe_rate': probe_rate(a.gpu)
    elif a.stage == 'plan_time': plan_time()
    elif a.stage == 'overheads': overheads(a.device)
    elif a.stage == 'time_bs32': time_bs32()
    else: report()


if __name__ == '__main__':
    main()
