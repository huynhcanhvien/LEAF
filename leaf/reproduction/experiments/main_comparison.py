"""Main comparison campaign: LEAF = station-rich SALT + P+D (all-future, G16/r32) + M3, seven backbones, nine re-run families.

Append-only (results/notes/plan/q1-revision-plan-2026-10-18.md, MAIN_COMPARISON). Stages:
  plan        (CPU) DP at the 48 paper budgets with the frozen station-rich SALT and historical M3 probes;
              references (paper M0, DP-M3 GMAC, DiffRate, nine baseline families incl. new ToFu/RAPID/One-More grids);
              ablations on DeiT-B/MAE-B (P-only, prune-only, merge-only, M0 fixed/replan, D-off fixed, Future1/3 fixed,
              calibration seeds 251/351 also on AugReg-S). Sealed by SHA before any evaluation.
  worker      (GPU, one per card) confirm/val/V2 at BF16 batch32, exact cache reuse from audited campaigns
  time        (GPU0, GPU1 idle) one session per backbone: bs8/bs32 for every arm, bs1/bs128 for core arms, 3 rounds
  report      (CPU) latency-paired McNemar/Holm vs every family; tables for the paper
"""
import argparse
from copy import deepcopy
import gc
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
from leaf.reproduction.runners.exclusive_timing import cast_exclusive, cpu_exclusive

from leaf.reproduction.experiments.latency_pricing import save, load, immutable, key, gpus_idle, card_idle, ROUNDS

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / 'results/main_comparison'
LOW = REPO / 'results/low_budget_and_factorial'
SOURCE = REPO / 'results/gmac_planning'
VALIDATION = REPO / 'results/salt_validation'
EXTRA_ABL = REPO / 'results/horizon_norm_seed_ablation'
BBS = ('deit_t', 'deit_s', 'deit_b', 'augreg_s', 'augreg_b', 'mae_b', 'mae_l')
ABLATION = ('deit_b', 'mae_b')
SEEDS = {'deit_b': (251, 351), 'mae_b': (251, 351), 'augreg_s': (251, 351)}
SPLITS = dict(confirm=20018, val=50000, v2=10000)
CORE_BATCHES, ALL_BATCHES = (1, 8, 32, 128), (8, 32)


def design():
    dest = ROOT / 'design.json'
    if dest.exists(): return load(dest)
    from leaf.reproduction.experiments.gmac_planning import sha
    low = load(LOW / 'design.json')
    doc = dict(backbones=low['backbones'], data_cache=low['data_cache'], original_root=low['original_root'],
               merge_root=low['merge_root'], parent_design_sha=low['sha256'], models=low['models'],
               rules=dict(method='station-rich SALT (frozen historical fit) + P+D G16/r32 + M3; unchanged DP',
                          budgets='the 48 paper budgets (24 DiffRate anchors x0.97, extras, AugReg dense fractions)',
                          selection='no confirm/val/V2 feedback; every planned arm evaluated; aliases once',
                          accuracy='BF16 deterministic batch32; exact (config,batch,n) cache reuse only',
                          timing='one session per backbone, GPU0 with GPU1 idle, three interleaved rounds'))
    doc['sha256'] = sha(doc)
    immutable(dest, doc)
    return doc


def runner(bb):
    import leaf.reproduction.experiments.horizon_norm_seed_ablation as extra
    r = extra.ExtraRunner(bb, ROOT, design())
    r.spec = __import__('leaf.backbones', fromlist=['SPECS']).SPECS[bb]
    r.teacher_cache = SOURCE.resolve() / 'teachers'
    r.reuse_roots = [SOURCE.resolve() / bb, VALIDATION.resolve() / bb, LOW.resolve() / bb, EXTRA_ABL.resolve() / bb]
    return r


def dcfg(bb):
    from leaf.planner.configs import ForecastConfig
    return ForecastConfig(Path(design()['backbones'][bb]['root']) / 'groups.json', G=16, rank=32)


def seed_curves(bb, seed):
    from leaf.planner.surrogate import ProbeCurve
    from leaf.backbones import SPECS
    rows = {}
    for a in load(EXTRA_ABL / bb / f'probe_means_seed{seed}.json'):
        rows.setdefault((a['theta'], a['kappa']), {}).setdefault(a['block'], {})[SPECS[bb].n_tokens - a['count']] = a['kl']
    return {k: ProbeCurve(v, SPECS[bb].n_tokens) for k, v in rows.items()}


def plan(bb):
    dest = ROOT / bb / 'plans.json'
    if dest.exists(): return load(dest)
    from leaf.backbones import SPECS
    from leaf.planner.plan import plan_latency, operator_config, config_details
    from leaf.planner.gmac import config_cost
    from leaf.reproduction.experiments.low_budget_and_factorial import probe_curves, baseline_jobs
    from leaf.reproduction.experiments.gmac_planning import sha
    spec, d = SPECS[bb], dcfg(bb)
    cur = {m: probe_curves(bb, d, m) for m in ('M0', 'M3')}
    model = design()['models'][bb]['main']
    budgets = design()['backbones'][bb]['budgets']
    frozen = load(SOURCE / bb / 'frozen.json')['jobs']
    jobs = {}
    for name, row in baseline_jobs(bb).items():           # dense + historical grids + low-budget extension grids
        jobs['ref/' + name] = dict(config=row['config'], group=row['group'])
    for kind, values in (('tofu', (4, 6, 8, 10, 12, 14)), ('rapid', (4, 6, 8, 10, 12, 14))):
        for v in values: jobs[f'ref/baseline/{kind}/r{v}'] = dict(config=dict(kind=kind, r=v), group='baseline', grid='q1_main_v2')
    for n in (40, 70, 100, 130, 160, 190):
        jobs[f'ref/baseline/onemore/n{n}'] = dict(config=dict(kind='onemore', n=n), group='baseline', grid='q1_main_v2')
    for name, row in frozen.items():
        if name.startswith(('paper/', 'M3/gmac/', 'M0/gmac/')):
            jobs['ref/' + name] = dict(config=row['config'], group='reference')
    for b in budgets:
        bid, L = b['id'], b['latency_ms']; t0 = time.monotonic()
        full = plan_latency(cur['M3'], spec, d, model, L, 'M3', True)
        jobs[f'leaf_alpha4/{bid}'] = dict(full, group='leaf_alpha4', budget=b, search_seconds=time.monotonic() - t0)
        if bb in ABLATION and 'config' in full:
            def keep(fn): return {k: v for k, v in cur['M3'].items() if fn(k)}
            for arm, cv, mode, use_d in (('no_forecast', cur['M3'], 'M3', False), ('prune_only', keep(lambda k: k[0] == 0), 'M3', True),
                                         ('merge_only', keep(lambda k: k[0] == 1), 'M3', True), ('m0_operator', cur['M0'], 'M0', True)):
                jobs[f'ablation/{arm}/{bid}'] = dict(plan_latency(cv, spec, d, model, L, mode, use_d), group='ablation', budget=b)
            for arm, mode, use_d in (('m0_operator_same_path', 'M0', True), ('no_forecast_same_path', 'M3', False)):
                jobs[f'ablation/{arm}/{bid}'] = dict(config=operator_config(full['config'], mode, use_d), group='ablation_fixed', budget=b)
            for variant in ('future1', 'future3'):
                c = deepcopy(full['config']); c['kind'] = 'extra_pdm'; c['ablation'] = dict(variant=variant, seed=1810)
                jobs[f'ablation/horizon{variant[-1]}_same_path/{bid}'] = dict(config=c, group='ablation_fixed', budget=b)
        for seed in SEEDS.get(bb, ()):
            jobs[f'seed{seed}/{bid}'] = dict(plan_latency(seed_curves(bb, seed), spec, d, model, L, 'M3', True), group='seed', budget=b)
    for name, row in jobs.items():
        if 'config' in row:
            row['config_sha'] = key(row['config'])
            if row['config']['kind'] in ('pd', 'pdm'):
                row['details'] = config_details(row['config'], spec); row['gmac'] = config_cost(row['config'], spec)
    out = dict(backbone=bb, jobs=jobs, sha=sha(jobs), time=time.time())
    immutable(dest, out)
    return out


def core(name):
    return name.startswith(('leaf_alpha4/', 'ref/baseline/DR', 'ref/dense', 'ref/paper/latency/'))


def tasks(bbs=BBS):
    out = {}
    for bb in bbs:
        if not (ROOT / bb / 'plans.json').exists(): continue
        for name, row in load(ROOT / bb / 'plans.json')['jobs'].items():
            if 'config' not in row: continue
            t = out.setdefault(bb + '-' + row['config_sha'], dict(id=bb + '-' + row['config_sha'], backbone=bb, config=row['config'], names=[]))
            t['names'].append(name)
    # Core arms first so the main table completes early; large backbones last.
    return sorted(out.values(), key=lambda t: (0 if any(core(n) for n in t['names']) else 1, BBS.index(t['backbone']), t['id']))


def claim(split, tid):
    p = ROOT / 'claims' / split / f'{tid}.lock'; p.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.close(os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)); return True
    except FileExistsError:
        return False


def worker():
    import torch
    from leaf.reproduction.runners.exclusive_timing import guard_foreign
    guard_foreign()
    rs = {}
    for split in SPLITS:
        for t in tasks():
            done = ROOT / 'completed' / split / f"{t['id']}.json"
            if done.exists() or not claim(split, t['id']): continue
            bb = t['backbone']
            if bb not in rs:
                rs.clear(); gc.collect(); torch.cuda.empty_cache(); rs[bb] = runner(bb)
            try:
                o = rs[bb].run(t['config'], split)
                save(done, dict(status='ok' if o is not None else 'infeasible', top1=float(o['correct'].mean() * 100) if o is not None else None,
                                kl=float(o['kl'].mean()) if o is not None else None, names=t['names'], time=time.time()))
            except torch.cuda.OutOfMemoryError:
                save(done, dict(status='OOM', names=t['names']))
            print(split, t['id'], t['names'][0], flush=True)


def timing(bbs=BBS):
    import torch
    from leaf.latency.timing import measure_latency
    for bb in bbs:
        done = ROOT / bb / 'latency_done.json'
        if done.exists(): continue
        while not card_idle(): time.sleep(30)
        r = runner(bb); ts = [t for t in tasks((bb,))]
        dest = ROOT / bb / 'latency'
        if dest.exists(): dest.rename(ROOT / bb / f'latency_interrupted_{time.time_ns()}')
        for rnd in range(ROUNDS):
            for t in (ts if rnd % 2 == 0 else ts[::-1]):
                if not card_idle(): raise RuntimeError('timed card shared during timing; session discarded')
                batches = CORE_BATCHES if any(core(n) for n in t['names']) else ALL_BATCHES
                meas = {}
                try:
                    m = r.inference_model(t['config'])
                    for b in batches:
                        try: meas[str(b)] = cast_exclusive(m, b, iters=50 if b <= 32 else 20, warmup=10 if b <= 32 else 5)
                        except torch.cuda.OutOfMemoryError: meas[str(b)] = 'OOM'; torch.cuda.empty_cache()
                except Exception as e:
                    meas = dict(status='error', error=repr(e))
                save(dest / f"{t['id']}#{rnd}.json", dict(id=t['id'], names=t['names'], round=rnd, measurements=meas,
                                                          clean=card_idle(), device=torch.cuda.get_device_name(0), time=time.time()))
                m = None; gc.collect(); torch.cuda.empty_cache()
        save(done, dict(rounds=ROUNDS, n=len(ts), time=time.time()))
        r = None; gc.collect(); torch.cuda.empty_cache()
        print('timed', bb, flush=True)



# ---------------------------------------------------------------- timing selection (user, 2026-10-18)
BASELINE_KINDS = {'tome', 'tome_m3', 'pitome', 'evit', 'mctf', 'ppt', 'tofu', 'rapid', 'onemore'}


def historical_ms(bb):
    """config_sha -> mean bs32 ms over the audited low-budget and DP-M3 sessions (cost only; never accuracy)."""
    import glob
    rows = {}
    for f in glob.glob(str(LOW / bb / 'latency' / '*.json')):
        d = load(f); m = d.get('measurement')
        if isinstance(m, dict) and 'ms_mean' in m: rows.setdefault(d['config_sha'], []).append(m['ms_mean'])
    for f in glob.glob(str(SOURCE / bb / 'latency' / '*.json')):
        d = load(f); m = (d.get('measurements') or {}).get('32')
        if isinstance(m, dict) and 'ms_mean' in m: rows.setdefault(d['config_sha'], []).append(m['ms_mean'])
    return {k: float(np.mean(v)) for k, v in rows.items()}


def timing_selection(bb, lo=.9, hi=1.1):
    """Pre-declared, cost-only pruning of baseline grid points far outside the LEAF latency range.

    Every non-baseline arm is kept. For each baseline family, keep the points whose historical bs32 latency is in
    [lo * smallest LEAF budget, hi * largest], plus the nearest point outside on each side (pairing/envelope),
    and every point without a historical latency. Dropped points are recorded with their reason."""
    dest = ROOT / bb / 'timing_selection.json'
    if dest.exists(): return load(dest)
    jobs = load(ROOT / bb / 'plans.json')['jobs']
    budgets = [row['budget']['latency_ms'] for name, row in jobs.items() if name.startswith('leaf_alpha4/')]
    lo_ms, hi_ms = lo * min(budgets), hi * max(budgets)
    hist = historical_ms(bb); ts = tasks((bb,)); keep, drop = [], []
    fam = {}
    for t in ts:
        kind = t['config']['kind']
        is_baseline = kind in BASELINE_KINDS and all(n.startswith('ref/') for n in t['names'])
        if not is_baseline: keep.append(t['id']); continue
        sha = t['id'].split('-', 1)[1]
        if sha not in hist: keep.append(t['id']); continue
        fam.setdefault(kind, []).append((hist[sha], t['id']))
    for kind, pts in fam.items():
        pts.sort()
        inside = [i for i, (ms, _) in enumerate(pts) if lo_ms <= ms <= hi_ms]
        below = [i for i, (ms, _) in enumerate(pts) if ms < lo_ms]
        above = [i for i, (ms, _) in enumerate(pts) if ms > hi_ms]
        chosen = set(inside) | ({below[-1]} if below else set()) | ({above[0]} if above else set())
        for i, (ms, tid) in enumerate(pts):
            (keep if i in chosen else drop).append(tid if i in chosen else dict(id=tid, family=kind, historical_ms=ms,
                                                    reason=f'outside [{lo_ms:.2f}, {hi_ms:.2f}] ms and not the nearest outside point'))
    out = dict(backbone=bb, window_ms=[lo_ms, hi_ms], rule=timing_selection.__doc__.strip(), keep=keep, drop=drop,
               n_tasks=len(ts), n_keep=len(keep), n_drop=len(drop), time=time.time())
    save(dest, out); return out


def timing_v2(bbs=BBS):
    """timing() on the selected tasks only (timing_selection); same session/round protocol."""
    import torch
    for bb in bbs:
        done = ROOT / bb / 'latency_done.json'
        if done.exists(): continue
        while not card_idle(): time.sleep(30)
        sel = set(timing_selection(bb)['keep'])
        r = runner(bb); ts = [t for t in tasks((bb,)) if t['id'] in sel]
        dest = ROOT / bb / 'latency'
        if dest.exists(): dest.rename(ROOT / bb / f'latency_interrupted_{time.time_ns()}')
        for rnd in range(ROUNDS):
            for t in (ts if rnd % 2 == 0 else ts[::-1]):
                if not card_idle(): raise RuntimeError('timed card shared during timing; session discarded')
                batches = CORE_BATCHES if any(core(n) for n in t['names']) else ALL_BATCHES
                meas = {}
                try:
                    m = r.inference_model(t['config'])
                    for b in batches:
                        try: meas[str(b)] = cast_exclusive(m, b, iters=50 if b <= 32 else 20, warmup=10 if b <= 32 else 5)
                        except torch.cuda.OutOfMemoryError: meas[str(b)] = 'OOM'; torch.cuda.empty_cache()
                except Exception as e:
                    meas = dict(status='error', error=repr(e))
                save(dest / f"{t['id']}#{rnd}.json", dict(id=t['id'], names=t['names'], round=rnd, measurements=meas,
                                                          device=torch.cuda.get_device_name(0), time=time.time()))
                m = None; gc.collect(); torch.cuda.empty_cache()
        save(done, dict(rounds=ROUNDS, n=len(ts), selection=str(ROOT / bb / 'timing_selection.json'), time=time.time()))
        r = None; gc.collect(); torch.cuda.empty_cache()
        print('timed', bb, len(ts), flush=True)


def claim(split, tid):   # overrides the earlier definition: crash-safe claims (leaf/reproduction/runners/exclusive_timing.py)
    from leaf.reproduction.runners.exclusive_timing import claim as common_claim
    return common_claim(ROOT, split, tid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['plan', 'worker', 'time'])
    ap.add_argument('--backbones', default=','.join(BBS))
    a = ap.parse_args(); bbs = a.backbones.split(',')
    if a.stage == 'plan':
        for bb in bbs:
            p = plan(bb); n = sum('config' in v for v in p['jobs'].values())
            print('PLANNED', bb, len(p['jobs']), 'slots', n, 'configs',
                  sum(1 for k, v in p['jobs'].items() if k.startswith('leaf_alpha4/') and 'config' in v), 'leaf_alpha4', flush=True)
    elif a.stage == 'worker':
        worker()
    else:
        timing_v2(bbs)


if __name__ == '__main__':
    main()
