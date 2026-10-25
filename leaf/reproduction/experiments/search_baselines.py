"""Black-box search baselines for the planner: is the exact DP needed, or does a generic search over station layouts
do as well?

Search space: the station blocks and a keep ratio per station. The token counts are the ratios raised to the smallest
exponent eta on the grid of the surrogate check (64 values in [0.01, 20]) whose schedule meets the SALT budget minus
the margin, and (theta, alpha) of every station is the probe arg-min, as in the DP (ablation_and_alpha_grid.Ctx.fixed).
Every candidate thus meets the same predicted budget as LEAF and differs from it only in where and how much it reduces.
Same probes (M3, four-value alpha grid), SALT model and budgets as the reference schedules leaf_alpha4 of the ablation.

Methods: random search (RS; initial draws as in the surrogate check: 2-8 stations, ratios in [0.5, 0.97]) and
regularised evolution (RE, Real et al. 2019: population 40, tournament 8, the oldest member leaves; a mutation adds,
removes or shifts a station or perturbs one keep ratio). Only distinct feasible schedules count as evaluations.

Stages (python leaf/reproduction/experiments/search_baselines.py <stage>; results in ROOT):
  surrogate (CPU)  RS and RE on F-hat at the three budgets of the surrogate check (DeiT-B, MAE-B), up to 10,000
                   evaluations, five seeds, against the DP -> surrogate.json
  true      (GPU)  RE on the measured calibration KL F (the 5,000 probe images), budgets interleaved, until --evals per
                   budget or --deadline; one process per backbone; every evaluation logged -> <bb>/history.jsonl.
                   Resumable: the RE population is the last 40 evaluations of a budget.
  seal      (CPU)  best RE schedule (lowest F) per budget and the leaf_alpha4 schedule of that budget -> sealed.json
  val       (GPU)  ImageNet val and V2 of the sealed schedules (one process per backbone)
  time      (GPU)  bs32 latency of the sealed schedules and dense, one GPU, both GPUs otherwise idle -> latency.json
  report    (CPU)  F, val/V2 top-1, McNemar of RE-best against LEAF -> report.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from leaf.reproduction.experiments import ablation_and_alpha_grid as A

ROOT = REPO / 'results/search_baselines'
BBS = ('deit_b', 'mae_b')
ETAS = np.geomspace(.01, 20, 64)
POP, TOURNAMENT = 40, 8
CHECKPOINTS = (100, 300, 1000, 3000, 10000)
SEEDS = (0, 1, 2, 3, 4)
SEED = 4101


def load(p): return json.loads(Path(p).read_text())


class Space:
    """Decoder of (sites, ratios) at one budget, with a cache of the distinct schedules seen."""

    def __init__(self, c, L):
        self.c, self.L, self.cv, self.seen = c, L, c.base['M3'], {}

    def decode(self, sites, ratios):
        """Smallest eta on ETAS whose schedule meets L - margin: dict(config, fhat, state, predicted_ms, eta) or None."""
        N = self.c.spec.n_tokens

        def at(i):
            n, counts = N, []
            for r in ratios:
                cnt = max(3, min(n - 1, round(n * r ** ETAS[i])))
                if cnt >= n: return None
                counts.append(cnt); n = cnt
            row = self.c.fixed(sites, counts, self.cv, 'M3', True)
            return dict(row, predicted_ms=self.c.cost(row['config'], 'M3'), eta=float(ETAS[i]))
        ok = lambda v: v is not None and v['predicted_ms'] <= self.L - A.MARGIN
        if at(0) is None: return None
        lo, hi = 0, len(ETAS) - 1                      # largest eta whose counts stay valid (counts shrink with eta)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if at(mid) is not None: lo = mid
            else: hi = mid - 1
        hi, lo = lo, 0                                 # bisection on the (nearly monotone) latency, then a local check
        if not ok(at(hi)): return None
        while lo < hi:
            mid = (lo + hi) // 2
            if ok(at(mid)): hi = mid
            else: lo = mid + 1
        while lo > 0 and ok(at(lo - 1)): lo -= 1
        return at(lo)


def draw(rng, depth):
    k = int(rng.integers(2, 9)); sites = sorted(rng.choice(np.arange(1, depth), size=k, replace=False).tolist())
    return sites, rng.uniform(.5, .97, size=k).tolist()


def mutate(rng, sites, ratios, depth):
    g = dict(zip(sites, ratios)); free = [l for l in range(1, depth) if l not in g]; op = int(rng.integers(4))
    if op == 0 and free: g[int(rng.choice(free))] = float(rng.uniform(.5, .97))
    elif op == 1 and len(g) > 1: del g[int(rng.choice(list(g)))]
    elif op == 2:
        l = int(rng.choice(list(g))); m = l + int(rng.choice((-1, 1)))
        if 1 <= m < depth and m not in g: g[m] = g.pop(l)
    else:
        l = int(rng.choice(list(g))); g[l] = float(np.clip(g[l] * np.exp(rng.normal(0, .1)), .3, .99))
    s = sorted(g)
    return s, [g[l] for l in s]


def candidate(space, rng, parent=None):
    """A distinct feasible schedule (new to the space), drawn or mutated from parent; None after many failures."""
    depth = space.c.spec.depth
    for _ in range(200):
        sites, ratios = draw(rng, depth) if parent is None else mutate(rng, *parent, depth)
        row = space.decode(sites, ratios)
        if row is None: continue
        k = A.sha(row['config'])
        if k in space.seen: continue
        space.seen[k] = row
        return dict(row, sites=sites, ratios=ratios, key=k)
    return None


def run_search(space, rng, method, n, score):
    """Best score after each of CHECKPOINTS (RS or RE); score(row) -> lower is better."""
    pop, best, trace = [], np.inf, {}
    for i in range(n):
        parent = None
        if method == 'RE' and len(pop) >= POP:
            idx = rng.choice(len(pop), size=TOURNAMENT, replace=False)
            parent = min((pop[j] for j in idx), key=lambda r: r['F'])
            parent = (parent['sites'], parent['ratios'])
        row = candidate(space, rng, parent)
        if row is None: break
        row['F'] = score(row); best = min(best, row['F'])
        pop.append(row)
        if len(pop) > POP: pop.pop(0)
        if i + 1 in CHECKPOINTS: trace[i + 1] = best
    return trace


# ----------------------------------------------------------------------------------------------- surrogate (CPU)
def surrogate():
    out = {}
    for bb in BBS:
        c = A.Ctx(bb)
        for b in c.b3:
            L = b['latency_ms']; dp = c.plan(c.base['M3'], L)
            ref = c.fixed(dp['state']['sites'], dp['state']['counts'], c.base['M3'], 'M3', True)
            row = dict(L=L, dp=dict(fhat=ref['fhat'], predicted_ms=c.cost(ref['config'], 'M3'), state=dp['state']), runs={})
            for method in ('RS', 'RE'):
                for seed in SEEDS:
                    t = time.time(); rng = np.random.default_rng(SEED + 100 * seed + BBS.index(bb))
                    trace = run_search(Space(c, L), rng, method, max(CHECKPOINTS), lambda r: r['fhat'])
                    row['runs'].setdefault(method, []).append(dict(seed=seed, best=trace, seconds=time.time() - t))
                    print(bb, b['id'], method, seed, {k: round(v / ref['fhat'], 3) for k, v in trace.items()}, flush=True)
            out[f"{bb}/{b['id']}"] = row
            A.save(ROOT / 'surrogate.json', out)
    return out


# ----------------------------------------------------------------------------------------------- true F (GPU)
def true_search(bb, evals, deadline):
    import torch
    from leaf.reproduction.runners.alpha_runner import alpha_runner
    from leaf.reproduction.runners.exclusive_timing import guard_foreign
    guard_foreign()
    c = A.Ctx(bb); r = alpha_runner(bb, ROOT)
    hist_path = ROOT / bb / 'history.jsonl'; hist_path.parent.mkdir(parents=True, exist_ok=True)
    hist = [json.loads(l) for l in hist_path.read_text().splitlines()] if hist_path.exists() else []
    spaces = {b['id']: Space(c, b['latency_ms']) for b in c.b3}
    pops = {bid: [] for bid in spaces}
    for h in hist:
        spaces[h['budget']].seen[h['key']] = h; pops[h['budget']].append(h)
    for b in c.b3:                                                       # LEAF's own F on the same images
        cfg = c.leaf_alpha4[b['id']]; r.run(cfg, 'half')
    rng = np.random.default_rng(SEED + len(hist) + 7 * BBS.index(bb))
    log = open(hist_path, 'a')
    while True:
        todo = [bid for bid in spaces if len(pops[bid]) < evals]
        if not todo or time.time() > deadline: break
        for bid in todo:
            if time.time() > deadline: break
            pop = pops[bid][-POP:]; parent = None
            if len(pops[bid]) >= POP:
                idx = rng.choice(len(pop), size=TOURNAMENT, replace=False)
                p = min((pop[j] for j in idx), key=lambda h: h['F']); parent = (p['sites'], p['ratios'])
            row = candidate(spaces[bid], rng, parent)
            if row is None: continue
            arr = r.run(row['config'], 'half')
            F = float('inf') if arr is None else float(arr['kl'].mean())
            h = dict(budget=bid, step=len(pops[bid]), key=row['key'], sites=row['sites'], ratios=row['ratios'], eta=row['eta'],
                     counts=row['state']['counts'], fhat=row['fhat'], predicted_ms=row['predicted_ms'], F=F, config=row['config'], time=time.time())
            pops[bid].append(h); log.write(json.dumps(h) + '\n'); log.flush()
    print('true search', bb, {bid: len(p) for bid, p in pops.items()}, flush=True)
    del r; torch.cuda.empty_cache()


def seal():
    dest = ROOT / 'sealed.json'
    if dest.exists(): return load(dest)
    jobs = {}
    for bb in BBS:
        c = A.Ctx(bb); hist = [json.loads(l) for l in (ROOT / bb / 'history.jsonl').read_text().splitlines()]
        for b in c.b3:
            h = [x for x in hist if x['budget'] == b['id']]
            if not h: continue
            best = min(h, key=lambda x: x['F'])
            jobs[f"{bb}/RE/{b['id']}"] = dict(config=best['config'], F=best['F'], evals=len(h), step=best['step'], predicted_ms=best['predicted_ms'])
            jobs[f"{bb}/LEAF/{b['id']}"] = dict(config=c.leaf_alpha4[b['id']], predicted_ms=c.cost(c.leaf_alpha4[b['id']], 'M3'))
    out = dict(jobs=jobs, sha=A.sha(jobs), time=time.time())
    A.save(dest, out)
    return out


def val(bb):
    from leaf.reproduction.runners.alpha_runner import alpha_runner
    r = alpha_runner(bb, ROOT)
    for name, job in load(ROOT / 'sealed.json')['jobs'].items():
        if name.startswith(bb + '/'):
            for split in ('half', 'val', 'v2'): r.run(job['config'], split)


def gpus_idle():
    """No process of anyone (this one excepted) on either GPU."""
    import os
    import subprocess
    pids = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).split()
    return all(int(p) == os.getpid() for p in pids)


def timing(rounds=3):
    """bs32 latency of the sealed schedules and the dense model, both backbones in one session on one GPU with both
    GPUs otherwise idle: three interleaved rounds, the middle one reversed, CUDA graphs (cast_exclusive, as the paper)."""
    import gc
    import torch
    from leaf.reproduction.runners.alpha_runner import alpha_runner
    from leaf.reproduction.runners.exclusive_timing import cast_exclusive
    dest = ROOT / 'latency'
    if (ROOT / 'latency.json').exists(): return
    while not gpus_idle(): time.sleep(60)
    sealed = load(ROOT / 'sealed.json')['jobs']
    jobs = [(bb, name, job['config']) for bb in BBS for name, job in [(f'{bb}/dense', dict(config=dict(kind='dense')))] + sorted(sealed.items())
            if name.startswith(bb + '/')]
    runners = {bb: alpha_runner(bb, ROOT) for bb in BBS}
    for rnd in range(rounds):
        for bb, name, cfg in (jobs if rnd % 2 == 0 else jobs[::-1]):
            if not gpus_idle(): raise SystemExit('another GPU process appeared during timing; session discarded')
            m = runners[bb].inference_model(cfg)
            meas = cast_exclusive(m, 32, iters=50, warmup=10)
            A.save(dest / f"{name.replace('/', '__')}#{rnd}.json", dict(name=name, round=rnd, ms=meas['ms_mean'], measurement=meas,
                                                                      device=torch.cuda.get_device_name(0), time=time.time()))
            m = None; gc.collect(); torch.cuda.empty_cache()
    out = {}
    for _, name, _ in jobs:
        ms = [load(dest / f"{name.replace('/', '__')}#{r}.json")['ms'] for r in range(rounds)]
        out[name] = dict(ms=float(np.mean(ms)), sd=float(np.std(ms, ddof=1)), rounds=ms)
    A.save(ROOT / 'latency.json', dict(batch=32, rounds=rounds, device=torch.cuda.get_device_name(0), time=time.time(), latency=out))
    print(json.dumps(out, indent=1))


def report():
    from leaf.evaluation.data import mcnemar
    from leaf.evaluation.runner import config_hash
    sealed = load(ROOT / 'sealed.json')['jobs']; out = {}
    for name, job in sealed.items():
        bb = name.split('/')[0]; h = config_hash(job['config'])
        out[name] = dict(predicted_ms=job['predicted_ms'], evals=job.get('evals'))
        for split in ('half', 'val', 'v2'):
            p = ROOT / bb / split / 'runs' / f'{h}.json'
            if p.exists(): row = load(p); out[name][split] = dict(top1=row['top1'], kl=row['kl'])
    for name in [n for n in sealed if '/RE/' in n]:
        ref = name.replace('/RE/', '/LEAF/'); bb = name.split('/')[0]
        for split in ('val', 'v2'):
            pa = ROOT / bb / split / 'runs' / f"{config_hash(sealed[name]['config'])}.npz"
            pb = ROOT / bb / split / 'runs' / f"{config_hash(sealed[ref]['config'])}.npz"
            if pa.exists() and pb.exists():
                a, b = np.load(pa)['correct'].astype(bool), np.load(pb)['correct'].astype(bool)
                p, leaf_only, re_only = mcnemar(b, a)
                out[name][f'{split}_vs_leaf'] = dict(leaf_minus_re=100 * float(b.mean() - a.mean()), p=p, leaf_only=leaf_only, re_only=re_only)
    A.save(ROOT / 'report.json', out)
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=('surrogate', 'true', 'seal', 'val', 'time', 'report'))
    ap.add_argument('--backbones', default=','.join(BBS))
    ap.add_argument('--evals', type=int, default=400, help='true: evaluations per budget')
    ap.add_argument('--deadline', default=None, help="true: stop at this time, e.g. '2026-10-23 08:20'")
    a = ap.parse_args(); bbs = a.backbones.split(',')
    deadline = time.mktime(time.strptime(a.deadline, '%Y-%m-%d %H:%M')) if a.deadline else float('inf')
    if a.stage == 'surrogate': surrogate()
    elif a.stage == 'true':
        for bb in bbs: true_search(bb, a.evals, deadline)
    elif a.stage == 'seal': print('SEALED', len(seal()['jobs']), flush=True)
    elif a.stage == 'val':
        for bb in bbs: val(bb)
    elif a.stage == 'time': timing()
    else: report()


if __name__ == '__main__':
    main()
