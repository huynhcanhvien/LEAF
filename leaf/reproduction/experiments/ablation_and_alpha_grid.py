"""Component ablation and hyperparameter sensitivity of LEAF with the alpha score (plan alpha-ablation-paper-v3-plan-2026-10-19).

Every schedule uses the convex score (1 - alpha)*z(log P) + alpha*z(D) (leaf/model/alpha_score.py). Existing probes
(q1_main_v2 / latency_d_m3_low_budget, k-keyed, 5,000 calibration images) are relabelled to alpha; new
probes are run only for settings that do not exist yet.

Stages (python leaf/reproduction/experiments/ablation_and_alpha_grid.py <stage> [--sets ...]):
  estimate  (CPU) number of new probes per set and backbone and the GPU-hours they need
  probe     (GPU, one process per idle card) single-station probes of the selected sets, calibration half
  plan      (CPU) every arm whose probes are complete; --dry-run prints the arms without writing.
            Writing seals ROOT/<bb>/plans/<group>.json by sha256 (immutable) once the whole group is planned.
  worker    (GPU, one process per idle card) val 50k and V2 10k, BF16 batch 32
  time      (GPU0, all GPUs idle) bs32 latency, three interleaved rounds, after all accuracy work
  report    (CPU) differences to the full method, McNemar + Holm, sensitivity and accuracy-latency figures

Groups (arm keys '<group>/<variant>/<budget>'; results in ROOT/<bb>/plans/<group>.json):
  leaf                  the paper's method: every budget re-planned on alpha in {0, 1/4, 1/3, 1/2, 2/3, 3/4, 1}
  grid_theta, grid_theta_alpha   the same with theta in {0, 1/4, 1/2, 3/4, 1} (and both grids)
  reference             the main-campaign schedules on the four-value alpha grid (leaf_alpha4) and the dense model
  schedule_structure    uniform / single-station / DiffRate-station schedules (x score x operator)
  shared_settings       one (theta, alpha), one theta or one alpha for every station
  forecast_only         alpha = 1 (rank by D alone), re-planned and on the same stations
  alpha_grid            coarser and extended alpha grids
  forecast_compression  head groups G and rank r of D (same stations)
  probe_sample_size     plans from 1,250 / 2,500 probe images
  theta_grid            theta grids
  final_station, final_station_all   final-block station removed (DeiT-B / MAE-B; all backbones)
  fixed_station_count   exactly K stations
  surrogate_check       30 random schedules per budget: surrogate F-hat vs measured F
  planner_switches, random_head_groups, probe_removal_grid, salt_workload_count   designed, not in the final run
"""
import argparse
from copy import copy, deepcopy
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from leaf.model.alpha_score import config_to_alpha, kappa_to_alpha
from leaf.planner.alpha import to_alpha_config, kappa_view

ROOT = REPO / 'results/ablation_and_alpha_grid'
EQUIV = REPO / 'results/alpha_kappa_equivalence'
COMPARISON = REPO / 'results/salt_fit'
P0_BBS, P1_BBS = ('deit_s', 'deit_b', 'mae_b'), ('deit_b', 'mae_b')
ALL_BUDGET_BBS = ('deit_b', 'mae_b')            # A1/A2 at every standard budget; others at three budgets
ALPHA_BASE = (0., .5, 2 / 3, .75)
ALPHA_EXT = (1.,)                               # forecast only; with ALPHA_BASE: {0, 1/2, 2/3, 3/4, 1}
ALPHA_FINE = (.25, 1 / 3)                       # user grid 2026-10-20: {0, 1/4, 1/3, 1/2, 2/3, 3/4, 1}
ALPHA_SUBSETS = dict(a2=(0., .5), a3=(0., .5, .75))   # coarser grids, no new probes
FIXED_K = (2, 3, 4, 6, 8)                       # stations including the final block
FULL_PROBES = False                             # True: also re-probe every G/r variant and the 17-point delta grid
H_SCHEDULES, H_SEED = 30, 2610
# 2026-10-20 (user): only the important ablations. DeiT-B and MAE-B, three budgets; A1 only along the schedule axis
# (P+D, M3); A4, A5, S5, S6, A7/forbid_final and DeiT-S dropped. CORE = False restores the full design.
CORE = True
CORE_GROUPS = {'reference', 'schedule_structure', 'shared_settings', 'forecast_only', 'alpha_grid', 'forecast_compression', 'probe_sample_size', 'theta_grid', 'final_station', 'fixed_station_count', 'surrogate_check', 'leaf'}
# R1 (user, 2026-10-20): every backbone and budget of the main campaign re-planned on the user's alpha grid
# {0, 1/4, 1/3, 1/2, 2/3, 3/4, 1}; the backbones outside P1_BBS run only this group (plus their leaf_alpha4/dense references).
ALL_BBS = ('deit_t', 'deit_s', 'deit_b', 'augreg_s', 'augreg_b', 'mae_b', 'mae_l')
LEAF_ONLY_BBS = tuple(bb for bb in ALL_BBS if bb not in P1_BBS)
RUN_BBS = ALL_BBS if CORE else P0_BBS
# PHASE 'grid' (user, 2026-10-20): first only re-plan the main schedules of all seven backbones on wider grids --
# R1: alpha in {0, 1/4, 1/3, 1/2, 2/3, 3/4, 1}; R3: theta in {0, 1/4, 1/2, 3/4, 1}; R2: both -- and compare with leaf_alpha4
# on val and V2. PHASE 'core' adds the component ablations afterwards.
PHASE = os.environ.get('ABL_PHASE', 'grid')    # 'core' for the component ablations (scripts/alpha_full_chain.sh)
GRID_GROUPS = {'reference', 'leaf', 'grid_theta_alpha', 'grid_theta'}
THETA_BASE, THETA_EXT = (0., .5, 1.), (.25, .75)
DELTAS = (8, 24, 48, 72, 96, 120, 144, 168, 194)
DELTA_MID, DELTA5 = (16, 36, 60, 84, 108, 132, 156, 181), (8, 48, 96, 144, 194)
D_DEFAULT = 'G16r32'
D_VARIANTS = ('G4r32', 'G8r32', 'G32r32', 'G16r8', 'G16r16', 'G16r64', 'G16rfull')
PLACEBO = 'rand16r32'
SUBSETS, SUBSET_SEEDS = (1250, 2500), (0, 1)
SALT_GROUPS, SALT_SEEDS = (10, 20), (0, 1)
SPLITS = dict(val=50000, v2=10000)
CALIB = 5000
THROUGHPUT = dict(deit_s=2931., deit_b=915., mae_b=915.,   # forward img/s, BF16 bs32 (paper Tab. search cost)
                  deit_t=7000., augreg_s=2931., augreg_b=915., mae_l=280.)   # rough guesses, estimate only
MARGIN = .1


def load(p): return json.loads(Path(p).read_text())


def save(p, obj):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp'); tmp.write_text(json.dumps(obj, indent=1) + '\n'); tmp.replace(p)


def sha(obj): return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def spec_of(bb):
    from leaf.backbones import SPECS
    return SPECS[bb]


def design():
    import leaf.reproduction.experiments.main_comparison as main
    return main.design()


# ----------------------------------------------------------------------------------------------- forecast variants
def dconfig(bb, name=D_DEFAULT):
    """ForecastConfig of a forecast variant: 'G<G>r<r>' (r = 'full' for no truncation) or 'rand<G>r<r>' (random groups
    with the clustered group sizes, seed fixed per block)."""
    from leaf.planner.configs import ForecastConfig
    path = Path(design()['backbones'][bb]['root']) / 'groups.json'
    rand = name.startswith('rand')
    g, r = name[4 if rand else 1:].split('r')
    rank = None if r == 'full' else int(r)
    d = ForecastConfig(path, G=int(g), rank=rank)
    if not rand: return d

    class RandomGroups(ForecastConfig):
        def at(self, l):
            labels = np.array(super().at(l))
            return [int(v) for v in np.random.default_rng(1910 + 1009 * int(l)).permutation(labels)]
    return RandomGroups(path, G=int(g), rank=rank)


# ----------------------------------------------------------------------------------------------- probes
def probe_sets(bb):
    """{set name: (variant, thetas, alphas, deltas)} of new single-station probes for this backbone."""
    sets = {'alpha_ext': (D_DEFAULT, THETA_BASE, ALPHA_EXT, DELTAS), 'alpha_fine': (D_DEFAULT, THETA_BASE, ALPHA_FINE, DELTAS)}
    if bb in P1_BBS or PHASE == 'grid':
        sets['theta_ext'] = (D_DEFAULT, THETA_EXT, ALPHA_BASE, DELTAS)
    if PHASE == 'grid':
        sets['theta_alpha_new'] = (D_DEFAULT, THETA_EXT, ALPHA_EXT + ALPHA_FINE, DELTAS)   # completes the 5 x 7 grid
    if bb in P1_BBS:
        if FULL_PROBES:
            sets['delta_mid'] = (D_DEFAULT, THETA_BASE, ALPHA_BASE, DELTA_MID)
            for v in D_VARIANTS + (PLACEBO,):
                sets['d_' + v] = (v, THETA_BASE, ALPHA_BASE[1:], DELTAS)
    return sets


def probe_config(bb, l, n, th, a, d, mode='M3'):
    """'pda' config of one station, built like the historical probes (pure pruning stays the plain operator)."""
    from leaf.planner.configs import config_from_state
    from leaf.planner.plan import operator_config
    from leaf.planner.probes import probe_state
    cfg = config_from_state(probe_state(l, n, th, a), spec_of(bb), d)    # 'k' field holds alpha
    if mode == 'M3' and th != 0: cfg = operator_config(cfg, 'M3')
    return to_alpha_config(cfg)


def probe_jobs(bb, name):
    variant, thetas, alphas, deltas = probe_sets(bb)[name]
    spec, d = spec_of(bb), dconfig(bb, variant)
    for l in range(1, spec.depth):
        for delta in deltas:
            for th in thetas:
                for a in alphas:
                    if l == spec.depth - 1 and a > 0: continue          # no later block, no forecast
                    yield dict(block=l, count=spec.n_tokens - delta, theta=th, alpha=a,
                               config=probe_config(bb, l, spec.n_tokens - delta, th, a, d))


def runner(bb):
    from leaf.reproduction.runners.alpha_runner import alpha_runner
    return alpha_runner(bb, ROOT, reuse_roots=[EQUIV / bb])


def claim(*parts):
    p = ROOT.joinpath('claims', *parts[:-1], parts[-1] + '.lock'); p.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.close(os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)); return True
    except FileExistsError:
        return False


def probe(sets=None, bbs=RUN_BBS):
    import torch
    from leaf.evaluation.runner import config_hash
    from leaf.reproduction.runners.exclusive_timing import guard_foreign
    guard_foreign()
    for bb in bbs:
        r = None
        for name in probe_sets(bb):
            if sets and name not in sets: continue
            dest = ROOT / bb / 'probes' / f'{name}.json'
            if dest.exists(): continue
            jobs = list(probe_jobs(bb, name))
            for job in jobs:
                tid = config_hash(job['config'])
                if (ROOT / bb / 'probe_done' / f'{tid}.json').exists() or not claim(bb, 'probe', tid): continue
                if r is None: r = runner(bb)
                arr = r.run(job['config'], 'half')
                save(ROOT / bb / 'probe_done' / f'{tid}.json', dict(kl=None if arr is None else float(arr['kl'].mean()),
                                                                     path=str(r.path(job['config'], 'half').with_suffix('.npz'))))
            collect(bb, name, jobs)
        del r; gc.collect(); torch.cuda.empty_cache()


def collect(bb, name, jobs=None):
    """Write probes/<set>.json once every job of the set is done (by any worker)."""
    from leaf.evaluation.runner import config_hash
    jobs = jobs or list(probe_jobs(bb, name)); rows = []
    for job in jobs:
        done = ROOT / bb / 'probe_done' / f"{config_hash(job['config'])}.json"
        if not done.exists(): return False
        row = load(done)
        if row['kl'] is None: raise RuntimeError(f'infeasible probe {job}')
        rows.append(dict({k: v for k, v in job.items() if k != 'config'}, kl=row['kl'], arrays=row['path'], config=job['config']))
    save(ROOT / bb / 'probes' / f'{name}.json', rows)
    return True


def estimate():
    total = 0.
    for bb in RUN_BBS:
        for name in probe_sets(bb):
            n = sum(1 for _ in probe_jobs(bb, name)); h = n * CALIB / THROUGHPUT[bb] / 3600; total += h
            print(f'{bb:7s} {name:12s} {n:5d} probes  ~{h:5.2f} GPU-h (forward-only lower bound)')
    print(f'total ~{total:.1f} GPU-h for probes')


# ----------------------------------------------------------------------------------------------- curves
def base_records(bb, mode='M3'):
    """Historical k probes (exact deployed configs, 5,000 images) relabelled to alpha."""
    import leaf.reproduction.experiments.main_comparison as main
    from leaf.reproduction.experiments.low_budget_and_factorial import probe_curves
    path = main.LOW / bb / f'probe_means_{mode}.json'
    if not path.exists(): probe_curves(bb, main.dcfg(bb), mode)     # writes the read-only export
    return [dict(r, alpha=kappa_to_alpha(r['kappa'])) for r in load(path)]


def new_records(bb, name):
    p = ROOT / bb / 'probes' / f'{name}.json'
    return load(p) if p.exists() else None


def curves_from(records, bb, idx=None, deltas=None):
    """{(theta, alpha): ProbeCurve} in record order; idx: mean KL over a subset of the calibration images."""
    from leaf.planner.surrogate import ProbeCurve
    spec = spec_of(bb); rows = {}
    for a in records:
        delta = spec.n_tokens - a['count']
        if deltas is not None and delta not in deltas: continue
        kl = a['kl'] if idx is None else float(np.load(a['arrays'])['kl'][idx].mean())
        rows.setdefault((a['theta'], a['alpha']), {}).setdefault(a['block'], {})[delta] = kl
    return {k: ProbeCurve(v, spec.n_tokens) for k, v in rows.items()}


def restrict(curves, depth, thetas=None, alphas=None):
    """Keep settings in thetas x alphas; alpha = 0 curves of allowed thetas stay available at the last block,
    where no forecast exists, so the final-block station is not lost by the restriction."""
    out = {}
    for (th, a), cv in curves.items():
        tok = thetas is None or th in thetas
        aok = alphas is None or a in alphas
        if tok and aok: out[(th, a)] = cv
        elif tok and a == 0 and depth - 1 in cv.curves:
            c = copy(cv); c.curves = {depth - 1: cv.curves[depth - 1]}; out[(th, a)] = c
    return out


def drop_block(curves, block):
    out = {}
    for k, cv in curves.items():
        c = copy(cv); c.curves = {l: v for l, v in cv.curves.items() if l != block}
        if c.curves: out[k] = c
    return out


# ----------------------------------------------------------------------------------------------- schedules
class ArgMin:
    """leaf.planner.surrogate.JointTables.best/lookup without precomputing the transition tables (cheap per schedule)."""

    def __init__(self, curves, total, depth):
        self.curves, self.total, self.depth = curves, total, depth

    def best(self, l, n_in, n_out):
        from leaf.planner.surrogate import JointTables
        return JointTables.best(self, l, n_in, n_out)


class Ctx:
    def __init__(self, bb):
        import leaf.reproduction.experiments.main_comparison as main
        self.bb, self.spec, self.d = bb, spec_of(bb), dconfig(bb)
        self.model = design()['models'][bb]['main']
        self.budgets = design()['backbones'][bb]['budgets']
        order = sorted(self.budgets, key=lambda b: b['latency_ms'])
        self.b3 = [order[0], order[len(order) // 2], order[-1]]
        self.records = {m: base_records(bb, m) for m in ('M0', 'M3')}
        self.base = {m: curves_from(self.records[m], bb) for m in ('M0', 'M3')}
        jobs = load(main.ROOT / bb / 'plans.json')['jobs']
        self.leaf_alpha4 = {b['id']: config_to_alpha(jobs[f"leaf_alpha4/{b['id']}"]['config']) for b in self.budgets
                      if 'config' in jobs.get(f"leaf_alpha4/{b['id']}", {})}

    def cost(self, cfg, mode=None, model=None):
        from leaf.planner.plan import OperatorCost
        mode = mode or ('M0' if kappa_view(cfg)['kind'] == 'pd' and 'merge' not in cfg else 'M3')
        return OperatorCost(model or self.model, self.spec, self.d, mode)(kappa_view(cfg))

    def plan(self, curves, L, mode='M3', use_d=True, d=None, model=None, **variant):
        """DP on alpha-keyed curves -> row with a 'pda' config (or the planner's infeasibility record)."""
        from leaf.reproduction.ablation_models.planner_ablation_switches import plan_latency_variant
        res = plan_latency_variant(curves, self.spec, d or self.d, model or self.model, L, mode, use_d, **variant)
        if 'config' not in res: return dict(res)
        res = dict(res, config=to_alpha_config(res['config']))
        res['state'] = dict(res['state']); res['state']['alphas'] = res['state'].pop('ks')
        return res

    def fixed(self, sites, counts, curves, mode, use_d, d=None):
        """Config of a given station layout; (theta, alpha) per station by the probe arg-min."""
        from leaf.planner.configs import config_from_state
        from leaf.planner.surrogate import lookup
        from leaf.planner.plan import operator_config
        cv = curves if use_d else {k: v for k, v in curves.items() if k[1] == 0}
        jt = ArgMin(cv, self.spec.n_tokens, self.spec.depth)
        th, al = lookup(jt, sites, counts)
        n, fhat = self.spec.n_tokens, 0.
        for l, c, t, a in zip(sites, counts, th, al): fhat += jt.curves[(t, a)].transition(l, n, c); n = c
        cfg = operator_config(config_from_state(dict(sites=list(sites), counts=list(counts), thetas=th, ks=al), self.spec, d or self.d), mode, use_d)
        return dict(config=to_alpha_config(cfg), fhat=float(fhat), state=dict(sites=list(sites), counts=list(counts), thetas=th, alphas=al))

    def fit_family(self, layout, L, curves, mode, use_d):
        """Smallest removal parameter of a layout family whose SALT estimate meets L - margin."""
        for r in range(1, self.spec.n_tokens):
            sites, counts = layout(r)
            if not sites: continue
            row = self.fixed(sites, counts, curves, mode, use_d)
            row['predicted_ms'] = self.cost(row['config'], mode)
            if row['predicted_ms'] <= L - MARGIN: return dict(row, family_parameter=r, status='planned')
        return dict(status='no_feasible_candidate')

    def uniform(self, L, curves, mode, use_d):
        """ToMe-like: a station in every block >= 1 removing the same number of tokens."""
        N, B = self.spec.n_tokens, self.spec.depth

        def layout(r):
            sites, counts, n = [], [], N
            for l in range(1, B):
                c = max(3, n - r)
                if c >= n: break
                sites.append(l); counts.append(c); n = c
            return sites, counts
        return self.fit_family(layout, L, curves, mode, use_d)

    def single(self, L, curves, mode, use_d):
        """One station (Pruning-One-More-Token style), block chosen by the probe estimate."""
        best = None
        for l in range(1, self.spec.depth - 1):
            row = self.fit_family(lambda r, l=l: ([l], [self.spec.n_tokens - r]) if self.spec.n_tokens - r >= 3 else ([], []),
                                  L, curves, mode, use_d)
            if row.get('status') == 'planned' and (best is None or row['fhat'] < best['fhat']): best = dict(row, block=l)
        return best or dict(status='no_feasible_candidate')

    def diffrate(self, bid, curves, mode, use_d):
        """Released DiffRate schedule (its own prune/merge counts) with our score and/or operator."""
        from leaf.planner.configs import diffrate_config
        from leaf.planner.plan import operator_config
        cfg = diffrate_config(self.spec, bid[2:])
        if use_d:
            n, alphas = self.spec.n_tokens, {}
            for l, (p, m) in enumerate(zip(cfg['prune_keep'], cfg['merge_keep'])):
                p, m = min(n, p), min(n, p, m)
                if m < n and l < self.spec.depth - 1:
                    th = min(THETA_BASE, key=lambda t: abs(t - (p - m) / (n - m)))
                    sub = {k: v for k, v in curves.items() if k[0] == th and l in v.curves}
                    a = min(sub, key=lambda k: sub[k].transition(l, n, m))[1]
                    if a > 0: alphas[str(l)] = a
                n = m
            cfg = dict(cfg, k=alphas, groups={l: self.d.at(int(l)) for l in alphas}, rank=self.d.rank if alphas else None)
        cfg = operator_config(cfg, mode, use_d)
        return dict(config=to_alpha_config(cfg), status='planned', predicted_ms=self.cost(to_alpha_config(cfg), mode))


def with_forecast(cfg, d, alpha=None):
    """Fixed-path variant of a 'pda' config: same stations and counts, another forecast (groups/rank/alpha)."""
    out = deepcopy(cfg)
    if alpha is not None: out['alpha'] = {l: float(alpha) for l in out['alpha']}
    out['groups'] = {l: d.at(int(l)) for l in out['alpha']}
    out['rank'] = d.rank if out['alpha'] else None
    return out


def without_final(cfg, spec):
    out = deepcopy(cfg); n = spec.n_tokens
    for l in range(spec.depth - 1): n = min(n, out['prune_keep'][l], out['merge_keep'][l])
    out['prune_keep'][-1] = out['merge_keep'][-1] = n
    return out


def salt_subset(bb, n_groups, seed):
    """Station-rich SALT refit on n_groups random workloads (same features and ridge as the frozen fit)."""
    from leaf.latency.salt import fit
    model = design()['models'][bb]['main']
    rows = [r for r in load(COMPARISON / 'dataset.json')['rows'] if r['backbone'] == bb]
    groups = sorted({r['group'] for r in rows})
    keep = set(np.random.default_rng(seed).choice(groups, size=min(n_groups, len(groups)), replace=False).tolist())
    sub = [r for r in rows if r['group'] in keep]
    x = np.array([[r['features'].get(k, 0.) for k in model['feature_names']] for r in sub])
    out = fit(x, np.array([r['ms'] for r in sub]), model['alpha'])
    out.update(feature_names=model['feature_names'], formula=model['formula'], n=len(sub), groups=len(keep), seed=seed)
    return out


# ----------------------------------------------------------------------------------------------- plan
def arms(bb, sets=None):
    """{name: row} of every arm for this backbone; arms whose probes are missing get status 'pending_probes'."""
    c = Ctx(bb); spec, jobs = c.spec, {}
    allowed = GRID_GROUPS if PHASE == 'grid' else ({'reference', 'leaf', 'final_station_all'} if bb in LEAF_ONLY_BBS else CORE_GROUPS | {'final_station_all'}) if CORE else None
    want = (lambda s: (allowed is None or s in allowed) and (sets is None or s in sets))

    def put(name, row, budget, group, **meta):
        jobs[name] = dict(row, budget=budget, group=group, **meta)

    budgets_a = c.budgets if bb in ALL_BUDGET_BBS and not CORE else c.b3
    for b in c.budgets:
        if b['id'] in c.leaf_alpha4: put(f"ref/leaf_alpha4/{b['id']}", dict(config=c.leaf_alpha4[b['id']], status='sealed'), b, 'reference')
    put('ref/dense', dict(config=dict(kind='dense'), status='reference'), None, 'reference')
    m3 = c.base['M3']
    for b in budgets_a:
        L, bid = b['latency_ms'], b['id']
        if want('schedule_structure'):
            for mode in ('M0', 'M3'):
                for score, use_d in (('P', False), ('PD', True)):
                    if CORE and (score, mode) != ('PD', 'M3'): continue
                    cv = c.base[mode]
                    if not CORE: put(f'schedule_structure/dp/{score}/{mode}/{bid}', c.plan(cv, L, mode, use_d), b, 'schedule_structure')
                    put(f'schedule_structure/uniform/{score}/{mode}/{bid}', c.uniform(L, cv, mode, use_d), b, 'schedule_structure')
                    put(f'schedule_structure/single/{score}/{mode}/{bid}', c.single(L, cv, mode, use_d), b, 'schedule_structure')
                    if bid.startswith('DR'):
                        put(f'schedule_structure/diffrate/{score}/{mode}/{bid}', c.diffrate(bid, cv, mode, use_d), b, 'schedule_structure')
        if want('shared_settings'):
            for kind, grid in (('both', [(t, a) for t in THETA_BASE for a in ALPHA_BASE]), ('theta', [(t, None) for t in THETA_BASE]),
                               ('alpha', [(None, a) for a in ALPHA_BASE])):
                best = None
                for t, a in grid:
                    res = c.plan(restrict(m3, spec.depth, None if t is None else (t,), None if a is None else (a,)), L)
                    if 'config' in res and (best is None or res['fhat'] < best['fhat']): best = dict(res, shared=dict(theta=t, alpha=a))
                put(f'shared_settings/shared_{kind}/{bid}', best or dict(status='no_feasible_candidate'), b, 'shared_settings', selection='lowest F-hat')
    ext = new_records(bb, 'alpha_ext')
    cache = {}
    if want('leaf'):
        fine = new_records(bb, 'alpha_fine')
        for b in c.budgets:
            put(f"leaf/alpha7/{b['id']}", c.plan(curves_from(c.records['M3'] + ext + fine, bb), b['latency_ms']) if ext and fine
                else dict(status='pending_probes', probes='alpha_ext+alpha_fine'), b, 'leaf', alphas=sorted(set(ALPHA_BASE + ALPHA_EXT + ALPHA_FINE)))
    if want('grid_theta') or want('grid_theta_alpha'):
        th = new_records(bb, 'theta_ext')
        full = [new_records(bb, n) for n in ('alpha_ext', 'alpha_fine', 'theta_ext', 'theta_alpha_new')]
        for b in c.budgets:
            if want('grid_theta'):
                put(f"grid_theta/theta5/{b['id']}", c.plan(curves_from(c.records['M3'] + th, bb), b['latency_ms']) if th
                    else dict(status='pending_probes', probes='theta_ext'), b, 'grid_theta', thetas=sorted(THETA_BASE + THETA_EXT))
            if want('grid_theta_alpha'):
                put(f"grid_theta_alpha/theta5_alpha7/{b['id']}", c.plan(curves_from(c.records['M3'] + sum(full, []), bb), b['latency_ms']) if all(full)
                    else dict(status='pending_probes', probes='alpha_ext+alpha_fine+theta_ext+theta_alpha_new'), b, 'grid_theta_alpha')
    if want('final_station_all'):
        # paper Sec. ablation / Limitations: latency share of the final-block station, every backbone, both planners
        import leaf.reproduction.experiments.main_comparison as main
        qjobs = load(main.ROOT / bb / 'plans.json')['jobs']
        for b in c.budgets:
            g = qjobs.get(f"ref/M3/gmac/{b['id']}", {}).get('config')
            for src, cfg in (('leaf_alpha4', c.leaf_alpha4.get(b['id'])), ('gmac', g and config_to_alpha(g))):
                if cfg is None: continue
                nf = without_final(cfg, spec)
                if min(cfg['prune_keep'][-1], cfg['merge_keep'][-1]) >= nf['prune_keep'][-1]: continue   # no cut in the last block
                if src == 'gmac': put(f"final_station_all/gmac_ref/{b['id']}", dict(config=cfg, status='reference'), b, 'final_station_all', source=src)
                put(f"final_station_all/{src}_no_final/{b['id']}", dict(config=nf, status='fixed'), b, 'final_station_all', source=src)
    if bb in LEAF_ONLY_BBS or PHASE == 'grid': return _shas(jobs)

    def subset_curves():
        if not cache:
            for n in SUBSETS:
                for seed in SUBSET_SEEDS:
                    idx = np.sort(np.random.default_rng(seed).choice(CALIB, size=n, replace=False))
                    cache[(n, seed)] = curves_from(c.records['M3'], bb, idx=idx)
        return cache
    for b in c.b3:
        L, bid = b['latency_ms'], b['id']; sealed = c.leaf_alpha4.get(bid)
        if want('forecast_only') and sealed: put(f'forecast_only/d_only_fixed/{bid}', dict(config=with_forecast(sealed, c.d, alpha=1.), status='fixed'), b, 'forecast_only')
        if want('forecast_only') or want('alpha_grid'):
            if ext is None:
                for arm in ('forecast_only/d_only', 'alpha_grid/alpha_ext'): put(f'{arm}/{bid}', dict(status='pending_probes', probes='alpha_ext'), b, arm.split('/')[0])
            else:
                cv = curves_from(c.records['M3'] + ext, bb)
                if want('forecast_only'): put(f'forecast_only/d_only/{bid}', c.plan(restrict(cv, spec.depth, alphas=(1.,)), L), b, 'forecast_only')
                if want('alpha_grid'): put(f'alpha_grid/alpha_ext/{bid}', c.plan(cv, L), b, 'alpha_grid')
        if want('alpha_grid'):
            for name, grid in ALPHA_SUBSETS.items():
                put(f'alpha_grid/{name}/{bid}', c.plan(restrict(m3, spec.depth, alphas=grid), L), b, 'alpha_grid', alphas=list(grid))
            fine = new_records(bb, 'alpha_fine')
            if ext is None or fine is None: put(f'alpha_grid/alpha_fine/{bid}', dict(status='pending_probes', probes='alpha_ext+alpha_fine'), b, 'alpha_grid')
            else: put(f'alpha_grid/alpha_fine/{bid}', c.plan(curves_from(c.records['M3'] + ext + fine, bb), L), b, 'alpha_grid')
        if want('forecast_compression') and sealed:
            for v in D_VARIANTS:
                put(f'forecast_compression/{v}_fixed/{bid}', dict(config=with_forecast(sealed, dconfig(bb, v)), status='fixed'), b, 'forecast_compression', variant=v)
        if want('probe_sample_size'):
            for (n, seed), cv in subset_curves().items():
                put(f'probe_sample_size/n{n}_s{seed}/{bid}', c.plan(cv, L), b, 'probe_sample_size', n_images=n, seed=seed)
        if want('final_station') and sealed:
            put(f'final_station/no_final_fixed/{bid}', dict(config=without_final(sealed, spec), status='fixed'), b, 'final_station')
            if not CORE: put(f'final_station/forbid_final/{bid}', c.plan(drop_block(m3, spec.depth - 1), L), b, 'final_station')
        if bb not in P1_BBS: continue
        for v in D_VARIANTS + (PLACEBO,):
            arm = 'random_head_groups' if v == PLACEBO else 'forecast_compression'
            if not want(arm): continue
            if arm == 'random_head_groups' and sealed: put(f'random_head_groups/{v}_fixed/{bid}', dict(config=with_forecast(sealed, dconfig(bb, v)), status='fixed'), b, 'random_head_groups')
            if not FULL_PROBES: continue
            rec = new_records(bb, 'd_' + v)
            if rec is None: put(f'{arm}/{v}/{bid}', dict(status='pending_probes', probes='d_' + v), b, arm, variant=v); continue
            cv = curves_from([r for r in c.records['M3'] if r['alpha'] == 0] + rec, bb)
            put(f'{arm}/{v}/{bid}', c.plan(cv, L, d=dconfig(bb, v)), b, arm, variant=v)
        if want('theta_grid'):
            rec = new_records(bb, 'theta_ext')
            put(f'theta_grid/theta5/{bid}', c.plan(curves_from(c.records['M3'] + rec, bb), L) if rec else dict(status='pending_probes', probes='theta_ext'), b, 'theta_grid')
            put(f'theta_grid/theta01/{bid}', c.plan(restrict(m3, spec.depth, thetas=(0., 1.)), L), b, 'theta_grid', thetas=[0., 1.])
            union = [new_records(bb, n) for n in ('theta_ext', 'alpha_ext', 'alpha_fine')]
            put(f'theta_grid/theta5_alpha_fine/{bid}', c.plan(curves_from(c.records['M3'] + sum(union, []), bb), L) if all(union)
                else dict(status='pending_probes', probes='theta_ext+alpha_ext+alpha_fine'), b, 'theta_grid')
        if want('probe_removal_grid'):
            put(f'probe_removal_grid/delta5/{bid}', c.plan(curves_from(c.records['M3'], bb, deltas=DELTA5), L), b, 'probe_removal_grid')
            rec = new_records(bb, 'delta_mid') if FULL_PROBES else False
            if rec is not False: put(f'probe_removal_grid/delta17/{bid}', c.plan(curves_from(c.records['M3'] + rec, bb), L) if rec else dict(status='pending_probes', probes='delta_mid'), b, 'probe_removal_grid')
        if want('fixed_station_count'):
            for k in FIXED_K:
                put(f'fixed_station_count/fixed{k}/{bid}', c.plan(m3, L, fixed_k=k), b, 'fixed_station_count', fixed_k=k)
        if want('planner_switches'):
            put(f'planner_switches/no_rescale/{bid}', c.plan(m3, L, rescale=False), b, 'planner_switches')
            put(f'planner_switches/single_price/{bid}', c.plan(m3, L, factors=(1.,)), b, 'planner_switches')
        if want('salt_workload_count'):
            for n in SALT_GROUPS:
                for seed in SALT_SEEDS:
                    model = salt_subset(bb, n, seed)
                    put(f'salt_workload_count/salt_g{n}_s{seed}/{bid}', c.plan(m3, L, model=model), b, 'salt_workload_count', salt=model)
    return _shas(jobs)


def _shas(jobs):
    from leaf.evaluation.runner import config_hash
    for row in jobs.values():
        if 'config' in row: row['config_sha'] = config_hash(row['config'])
    return jobs


def plan(bbs=RUN_BBS, sets=None, dry_run=False):
    """Seal ROOT/<bb>/plans/<group>.json once all arms of the group are planned; groups waiting for probes are
    reported and sealed by a later call after the probe stage."""
    for bb in bbs:
        jobs = arms(bb, sets); by = {}
        for name, row in jobs.items(): by.setdefault(row['group'], {})[name] = row
        n_cfg = len({row['config_sha'] for row in jobs.values() if 'config' in row})
        status = {g: dict(__import__('collections').Counter(r.get('status', '?') for r in rows.values())) for g, rows in by.items()}
        print(bb, f'{len(jobs)} arms, {n_cfg} unique configs', json.dumps(status))
        if dry_run: continue
        for group, rows in by.items():
            dest = ROOT / bb / 'plans' / f'{group}.json'
            if dest.exists(): continue
            pending = sorted(n for n, r in rows.items() if r.get('status') == 'pending_probes')
            if pending: print(f'  {group}: waiting for probes ({len(pending)} arms)'); continue
            save(dest, dict(backbone=bb, group=group, jobs=rows, sha=sha(rows), time=time.time()))
            print(f'  {group}: sealed {len(rows)} arms')
        if bb in P1_BBS and (sets is None or 'surrogate_check' in sets):
            print(f"  H: {len(h_schedules(bb)['schedules'])} random schedules sealed")


def sealed_jobs(bb):
    out = {}
    for p in sorted((ROOT / bb / 'plans').glob('*.json')): out.update(load(p)['jobs'])
    return out


# ----------------------------------------------------------------------------------------------- surrogate check H
def h_schedules(bb):
    """H_SCHEDULES random schedules per main budget (2-8 stations, random blocks, keep ratios and (theta, alpha) from
    the method's grids), scaled by eta like the planner to meet the SALT budget; F-hat from the M3 probe curves.
    Sealed once to ROOT/<bb>/h_schedules.json; F is the mean KL on the same 5,000 calibration images."""
    dest = ROOT / bb / 'surrogate_check_schedules.json'
    if dest.exists(): return load(dest)
    from leaf.planner.configs import config_from_state
    from leaf.planner.plan import operator_config
    c = Ctx(bb); spec = c.spec; rng = np.random.default_rng(H_SEED + P0_BBS.index(bb)); m3 = c.base['M3']; rows = []
    for b in c.b3:
        L = b['latency_ms']; made = 0
        while made < H_SCHEDULES:
            k = int(rng.integers(2, 9)); sites = sorted(rng.choice(np.arange(1, spec.depth), size=k, replace=False).tolist())
            ratios = rng.uniform(.5, .97, size=k)
            thetas = [float(rng.choice(THETA_BASE)) for _ in sites]
            alphas = [0. if l == spec.depth - 1 else float(rng.choice(ALPHA_BASE)) for l in sites]

            def at(eta):
                n, counts = spec.n_tokens, []
                for r in ratios:
                    cnt = max(3, min(n - 1, round(n * r ** eta)))
                    if cnt >= n: return None
                    counts.append(cnt); n = cnt
                st = dict(sites=sites, counts=counts, thetas=thetas, ks=alphas)
                cfg = to_alpha_config(operator_config(config_from_state(st, spec, c.d), 'M3', True))
                return cfg, counts, c.cost(cfg, 'M3')
            ok = [(e, v) for e in np.geomspace(.01, 20, 64) if (v := at(float(e))) is not None and v[2] <= L - MARGIN]
            if not ok: continue
            cfg, counts, ms = ok[0][1]
            n, fhat = spec.n_tokens, 0.
            for l, cnt, th, a in zip(sites, counts, thetas, alphas): fhat += m3[(th, a)].transition(l, n, cnt); n = cnt
            from leaf.evaluation.runner import config_hash
            rows.append(dict(id=f'{bb}-h-{config_hash(cfg)}', budget=b['id'], config=cfg, fhat=float(fhat), predicted_ms=ms, K=k))
            made += 1
    doc = dict(backbone=bb, seed=H_SEED, schedules=rows, sha=sha(rows), time=time.time())
    save(dest, doc)
    return doc


def surrogate(bbs=P1_BBS):
    """GPU: F = mean calibration KL of every H schedule (split 'half', the probe images), then val KL/top-1 of the
    same schedules (does the calibration ranking carry over to held-out data?)."""
    import torch
    from leaf.reproduction.runners.exclusive_timing import guard_foreign
    guard_foreign()
    for bb in bbs:
        r = None
        for row in h_schedules(bb)['schedules']:
            done = ROOT / bb / 'surrogate_check_done' / f"{row['id']}.json"
            if done.exists() or not claim(bb, 'h', row['id']): continue
            if r is None: r = runner(bb)
            arr = r.run(row['config'], 'half')
            save(done, dict(F=None if arr is None else float(arr['kl'].mean()), fhat=row['fhat'], budget=row['budget'], K=row['K']))
        for row in h_schedules(bb)['schedules']:
            done = ROOT / bb / 'surrogate_check_val' / f"{row['id']}.json"
            if done.exists() or not claim(bb, 'hval', row['id']): continue
            if r is None: r = runner(bb)
            arr = r.run(row['config'], 'val')
            save(done, dict(kl=None if arr is None else float(arr['kl'].mean()), top1=None if arr is None else float(arr['correct'].mean() * 100)))
        del r; gc.collect(); torch.cuda.empty_cache()


def h_summary(bbs=P1_BBS):
    from scipy.stats import spearmanr
    out = {}
    for bb in bbs:
        done = [load(p) for p in sorted((ROOT / bb / 'surrogate_check_done').glob('*.json'))]
        done = [d for d in done if d['F'] is not None]
        for bid in sorted({d['budget'] for d in done}):
            sel = [d for d in done if d['budget'] == bid]
            ids = [p.stem for p in sorted((ROOT / bb / 'surrogate_check_done').glob('*.json')) if load(p)['budget'] == bid and load(p)['F'] is not None]
            val = {i: load(ROOT / bb / 'surrogate_check_val' / f'{i}.json') for i in ids if (ROOT / bb / 'surrogate_check_val' / f'{i}.json').exists()}
            pair = [(load(ROOT / bb / 'surrogate_check_done' / f'{i}.json')['F'], v['kl'], v['top1']) for i, v in val.items() if v['kl'] is not None]
            out[f'{bb}/{bid}'] = dict(n=len(sel), spearman=float(spearmanr([d['fhat'] for d in sel], [d['F'] for d in sel])[0]),
                                      calib_vs_val_kl=float(spearmanr([x[0] for x in pair], [x[1] for x in pair])[0]) if len(pair) > 2 else None,
                                      calib_kl_vs_val_top1=float(spearmanr([x[0] for x in pair], [-x[2] for x in pair])[0]) if len(pair) > 2 else None,
                                      ratio_by_K={k: float(np.median([d['F'] / d['fhat'] for d in sel if d['K'] == k]))
                                                  for k in sorted({d['K'] for d in sel})})
    return out


# ----------------------------------------------------------------------------------------------- evaluation
def tasks(bbs=RUN_BBS):
    out = {}
    for bb in bbs:
        for name, row in sealed_jobs(bb).items():
            if 'config' not in row: continue
            t = out.setdefault(f"{bb}-{row['config_sha']}", dict(id=f"{bb}-{row['config_sha']}", backbone=bb, config=row['config'], names=[]))
            t['names'].append(name)
    return sorted(out.values(), key=lambda t: (ALL_BBS.index(t['backbone']), t['id']))


def worker():
    import torch
    from leaf.reproduction.runners.exclusive_timing import guard_foreign
    guard_foreign(); r, bb_last = None, None
    for split in SPLITS:
        for t in tasks():
            done = ROOT / 'completed' / split / f"{t['id']}.json"
            if done.exists() or not claim(split, t['id']): continue
            if t['backbone'] != bb_last:
                r = None; gc.collect(); torch.cuda.empty_cache(); r = runner(t['backbone']); bb_last = t['backbone']
            try:
                o = r.run(t['config'], split)
                save(done, dict(status='ok' if o is not None else 'infeasible', names=t['names'],
                                path=str(r.path(t['config'], split).with_suffix('.npz')),
                                top1=None if o is None else float(o['correct'].mean() * 100), time=time.time()))
            except torch.cuda.OutOfMemoryError:
                save(done, dict(status='OOM', names=t['names']))
            print(split, t['id'], t['names'][0], flush=True)


def timing(bbs=RUN_BBS, rounds=3):
    """bs32 latency with every GPU idle, after all accuracy work; interleaved rounds, middle round reversed."""
    import torch
    from leaf.reproduction.runners.alpha_runner import build_any
    from leaf.latency.timing import measure_latency
    from leaf.reproduction.runners.exclusive_timing import measure_exclusive
    missing = [t['id'] for t in tasks(bbs) for s in SPLITS if not (ROOT / 'completed' / s / f"{t['id']}.json").exists()]
    if missing: raise RuntimeError(f'{len(missing)} accuracy evaluations missing; timing runs last')
    for bb in bbs:
        if (ROOT / bb / 'latency_done.json').exists(): continue
        r = runner(bb); _ = r.vit; ts = [t for t in tasks((bb,))]
        for rnd in range(rounds):
            for t in (ts if rnd % 2 == 0 else ts[::-1]):
                dest = ROOT / bb / 'latency' / f"{t['id']}#{rnd}.json"
                if dest.exists(): continue
                model = r.vit if t['config']['kind'] == 'dense' else build_any(r.vit, r.spec, t['config'])
                with measure_exclusive() as paused:
                    m = measure_latency(model, 32, iters=50, warmup=10)
                save(dest, dict(task=t['id'], round=rnd, measurement=m, paused=paused, time=time.time()))
                del model; gc.collect(); torch.cuda.empty_cache()
        save(ROOT / bb / 'latency_done.json', dict(rounds=rounds, n_configs=len(ts), time=time.time()))


# ----------------------------------------------------------------------------------------------- report
def arrays(bb, cfg, split):
    from leaf.evaluation.runner import config_hash
    for base in (ROOT / bb, EQUIV / bb):
        p = base / split / 'runs' / config_hash(cfg)
        if p.with_suffix('.npz').exists(): return dict(np.load(p.with_suffix('.npz')))
    return None


def latency(bb, cfg):
    from leaf.evaluation.runner import config_hash
    vals = [load(p)['measurement']['ms_mean'] for p in sorted((ROOT / bb / 'latency').glob(f'{bb}-{config_hash(cfg)}#*.json'))]
    return float(np.mean(vals)) if vals else None


def report():
    from leaf.evaluation.data import holm, mcnemar
    out = {}
    for bb in ALL_BBS:
        jobs = sealed_jobs(bb); rows = []
        if not jobs: continue
        for split in SPLITS:
            tests = []
            for name, row in jobs.items():
                if 'config' not in row or name.startswith('ref/') or row.get('budget') is None: continue
                ref = jobs.get(f"ref/leaf_alpha4/{row['budget']['id']}")
                a, b = arrays(bb, row['config'], split), ref and arrays(bb, ref['config'], split)
                if a is None or b is None: continue
                pv, nb, nc = mcnemar(a['correct'].astype(bool), b['correct'].astype(bool))
                tests.append(dict(name=name, split=split, group=row['group'], budget=row['budget']['id'], top1=100 * float(a['correct'].mean()),
                                  ref_top1=100 * float(b['correct'].mean()), delta=100 * float(a['correct'].mean() - b['correct'].mean()),
                                  p=pv, same_config=row['config_sha'] == ref['config_sha'], ms=latency(bb, row['config']),
                                  ref_ms=latency(bb, ref['config']), gmac=row.get('gmac')))
            rows += holm([t for t in tests if not t['same_config']]) + [dict(t, holm=1.) for t in tests if t['same_config']]
        out[bb] = rows
    hs = h_summary()
    save(ROOT / 'summary.json', dict(out, _surrogate_check=hs))
    figures(out)
    lines = ['# Ablation v3 (alpha score) — results', '', f'Raw: `{ROOT.relative_to(REPO)}`. Plan: `results/notes/plan/alpha-ablation-paper-v3-plan-2026-10-19.md`.', '',
             'Δ = arm − full LEAF (same latency budget), in percentage points; **bold**: Holm p < 0.05 across all contrasts of the backbone and split.', '']
    for bb, rows in out.items():
        lines += [f'## {bb}', '', '| Arm | Budget | ms | Δ val | Δ V2 |', '|---|---|---:|---:|---:|']
        by = {}
        for t in rows: by.setdefault((t['name'], t['budget']), {})[t['split']] = t
        for (name, bid), d in sorted(by.items()):
            cell = lambda s: '' if s not in d else (f"**{d[s]['delta']:+.2f}**" if d[s]['holm'] < .05 else f"{d[s]['delta']:+.2f}")
            ms = next((x['ms'] for x in d.values() if x['ms']), None)
            lines.append(f"| {name.rsplit('/', 1)[0]} | {bid} | {'' if ms is None else f'{ms:.2f}'} | {cell('val')} | {cell('v2')} |")
        lines += ['', f'![acc-latency {bb}](figures/ablation_v3/{bb}_acc_latency.png)', f'![sensitivity {bb}](figures/ablation_v3/{bb}_sensitivity.png)', '']
    r1 = []
    for bb in ALL_BBS:
        jobs = sealed_jobs(bb)
        for name, row in sorted(jobs.items()):
            if not name.startswith('leaf/') or 'config' not in row: continue
            ref = jobs.get(f"ref/leaf_alpha4/{row['budget']['id']}")
            a1 = sum(v == 1. for v in row['config']['alpha'].values()); anew = sum(v in ALPHA_FINE for v in row['config']['alpha'].values())
            d = {t['split']: t for t in out.get(bb, []) if t['name'] == name}
            cell = lambda sp: '' if sp not in d else (f"**{d[sp]['delta']:+.2f}**" if d[sp]['holm'] < .05 else f"{d[sp]['delta']:+.2f}")
            ms = next((x['ms'] for x in d.values() if x.get('ms')), None); rms = next((x['ref_ms'] for x in d.values() if x.get('ref_ms')), None)
            r1.append(f"| {bb} | {row['budget']['id']} | {'=' if ref and ref['config_sha'] == row['config_sha'] else 'different'} | {a1}/{anew}/{len(row['config']['alpha'])} | "
                      f"{cell('val')} | {cell('v2')} | {'' if ms is None else f'{ms:.2f}'} | {'' if rms is None else f'{rms:.2f}'} |")
    if r1:
        lines += ['## leaf (alpha grid of 7 values): schedules re-planned with the α grid {0, ¼, ⅓, ½, ⅔, ¾, 1}, 7 backbones × all budgets', '',
                  '| Backbone | Budget | vs leaf_alpha4 | stations with α=1 / α∈{¼,⅓} / stations with D | Δ val | Δ V2 | ms | ms leaf_alpha4 |', '|---|---|---|---:|---:|---:|---:|---:|'] + r1 + ['']
    a7 = []
    for bb in ALL_BBS:
        jobs = sealed_jobs(bb)
        for name, row in sorted(jobs.items()):
            if not name.startswith('final_station_all/') or not name.split('/')[1].endswith('_no_final'): continue
            src, bid = row.get('source'), row['budget']['id']
            ref = jobs.get(f'ref/leaf_alpha4/{bid}' if src == 'leaf_alpha4' else f'final_station_all/gmac_ref/{bid}')
            if not ref: continue
            ms, rms = latency(bb, row['config']), latency(bb, ref['config'])
            x, y = arrays(bb, row['config'], 'val'), arrays(bb, ref['config'], 'val')
            same = None if x is None or y is None else bool((x['pred'] == y['pred']).all())
            a7.append((bb, bid, src, same, ms, rms))
    if a7:
        lines += ['## final_station_all: latency of the final station (final station removed, same schedule)', '',
                  '| Backbone | Budget | schedule | identical val predictions | ms without final station | ms original | Δ ms (original − removed) |', '|---|---|---|---|---:|---:|---:|']
        for bb, bid, src, same, ms, rms in a7:
            d = '' if ms is None or rms is None else f'{rms - ms:+.3f}'
            lines.append(f"| {bb} | {bid} | {src} | {'' if same is None else ('yes' if same else '**no**')} | {'' if ms is None else f'{ms:.2f}'} | {'' if rms is None else f'{rms:.2f}'} | {d} |")
        lines.append('')
    if hs:
        lines += ['## surrogate_check: F-hat vs F on random schedules', '', '| Backbone/budget | n | ρ(F-hat, F) | ρ(KL calib, KL val) | ρ(KL calib, −top-1 val) | median F/F-hat by K |',
                  '|---|---:|---:|---:|---:|---|']
        f2 = lambda x: '' if x is None else f'{x:.2f}'
        for key, v in hs.items():
            lines.append(f"| {key} | {v['n']} | {v['spearman']:.2f} | {f2(v['calib_vs_val_kl'])} | {f2(v['calib_kl_vs_val_top1'])} | "
                         + ', '.join(f'K{k}: {r:.2f}' for k, r in v['ratio_by_K'].items()) + ' |')
    (REPO / 'results/reports/ablation-v3-2026-10-19.md').write_text('\n'.join(lines) + '\n')


def figures(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig_dir = REPO / 'results/reports/figures/ablation_v3'; fig_dir.mkdir(parents=True, exist_ok=True)
    for bb, rows in out.items():
        val = [t for t in rows if t['split'] == 'val' and t['ms']]
        if val:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            groups = sorted({t['name'].rsplit('/', 1)[0] for t in val})
            for g in groups:
                pts = sorted((t['ms'], t['top1']) for t in val if t['name'].startswith(g + '/'))
                ax.plot(*zip(*pts), marker='o', ms=3, lw=1, label=g)
            ref = sorted({(t['ref_ms'], t['ref_top1']) for t in val if t['ref_ms']})
            if ref: ax.plot(*zip(*ref), 'k-', lw=2.5, marker='s', label='LEAF (full)')
            ax.set_xlabel('measured latency, bs32 (ms)'); ax.set_ylabel('ImageNet val top-1 (%)'); ax.set_title(bb)
            ax.legend(fontsize=5, ncol=2); fig.tight_layout(); fig.savefig(fig_dir / f'{bb}_acc_latency.png', dpi=160); plt.close(fig)
        sens = {'alpha_grid': 'alpha grid', 'forecast_compression': 'G / r', 'probe_sample_size': '|C|', 'theta_grid': 'Theta', 'probe_removal_grid': 'Delta grid', 'salt_workload_count': 'SALT workloads'}
        present = [k for k in sens if any(t['group'] == k for t in rows)]
        if not present: continue
        fig, axes = plt.subplots(1, len(present), figsize=(3.2 * len(present), 3), squeeze=False)
        for ax, k in zip(axes[0], present):
            sel = [t for t in rows if t['group'] == k and t['split'] == 'val']
            names = sorted({t['name'].rsplit('/', 1)[0].split('/', 1)[1] for t in sel})
            for bid in sorted({t['budget'] for t in sel}):
                ys = [next((t['delta'] for t in sel if t['budget'] == bid and t['name'].rsplit('/', 1)[0].endswith(n)), np.nan) for n in names]
                ax.plot(range(len(names)), ys, marker='o', label=bid)
            ax.axhline(0, color='k', lw=.8); ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=60, fontsize=6)
            ax.set_title(sens[k], fontsize=8); ax.set_ylabel('Δ val top-1 vs default')
        axes[0][0].legend(fontsize=6); fig.tight_layout(); fig.savefig(fig_dir / f'{bb}_sensitivity.png', dpi=160); plt.close(fig)


def status(bbs=RUN_BBS):
    """One progress line: probes done/total per backbone, sealed plan groups, accuracy evaluations, timing files."""
    from leaf.evaluation.runner import config_hash
    parts = []
    for bb in bbs:
        ids = {config_hash(j['config']) for name in probe_sets(bb) for j in probe_jobs(bb, name)}
        done = sum((ROOT / bb / 'probe_done' / f'{t}.json').exists() for t in ids)
        parts.append(f"{bb}: probes {done}/{len(ids)} sealed {len(list((ROOT / bb / 'plans').glob('*.json')))}")
    ts = tasks(bbs)
    for s in SPLITS:
        parts.append(f"{s} {sum((ROOT / 'completed' / s / (t['id'] + '.json')).exists() for t in ts)}/{len(ts)}")
    parts.append(f"latency {sum(len(list((ROOT / bb / 'latency').glob('*.json'))) for bb in bbs)}/{3 * len(ts)}")
    print(' | '.join(parts))


def release():
    """Remove claim locks without a result (a worker died); run only while no worker is alive."""
    n = 0
    for p in (ROOT / 'claims').rglob('*.lock'):
        rel = p.relative_to(ROOT / 'claims').parts
        if rel[0] in SPLITS: done = ROOT / 'completed' / rel[0] / (p.stem + '.json')
        elif rel[1] == 'h': done = ROOT / rel[0] / 'surrogate_check_done' / (p.stem + '.json')
        elif rel[1] == 'hval': done = ROOT / rel[0] / 'surrogate_check_val' / (p.stem + '.json')
        else: done = ROOT / rel[0] / 'probe_done' / (p.stem + '.json')
        if not done.exists(): p.unlink(); n += 1
    print(f'released {n} stale claims')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=('estimate', 'probe', 'plan', 'worker', 'time', 'report', 'status', 'release', 'surrogate'))
    ap.add_argument('--sets', nargs='*', default=None, help='probe sets (probe) or arm groups (plan), e.g. A1 A2 S3')
    ap.add_argument('--bbs', nargs='*', default=list(RUN_BBS))
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    if a.stage == 'estimate': estimate()
    elif a.stage == 'probe': probe(a.sets, tuple(a.bbs))
    elif a.stage == 'plan': plan(tuple(a.bbs), a.sets, a.dry_run)
    elif a.stage == 'worker': worker()
    elif a.stage == 'time': timing(tuple(a.bbs))
    elif a.stage == 'status': status(tuple(a.bbs))
    elif a.stage == 'release': release()
    elif a.stage == 'surrogate': surrogate()
    else: report()


if __name__ == '__main__':
    main()
