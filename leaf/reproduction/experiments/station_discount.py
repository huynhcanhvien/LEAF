"""Station-count correction of the separable distortion (mentor comment L775): does the overcharge of schedules with
many stations by F-hat cost accuracy?

On the 180 random schedules of the surrogate check (ablation_and_alpha_grid, DeiT-B and ViT-B MAE), F-hat ranks
schedules within a budget poorly and overcharges many stations. A correction that keeps the planner exact charges the
k-th station rho**(k-1) times its probed damage (the dynamic program carries the station count in its state). rho is
fitted once, pooled over both backbones, as the value that maximises the mean within-budget Spearman correlation
between the corrected F-hat and the measured calibration KL F; then every budget of all seven backbones is re-planned
with the full method's probes, score, operator and SALT, and compared with the full method at the same latency budget.

Stages (python leaf/reproduction/experiments/station_discount.py <stage>):
  fit     (CPU) rho and its leave-one-budget-out / leave-one-backbone-out check -> ROOT/fit.json (sealed)
  plan    (CPU) discounted schedules of every budget + the full method + dense -> ROOT/<bb>/plans.json (sealed)
  worker  (GPU, one process per idle card) val 50k and V2 10k, BF16 batch 32 (the full method's runs are reused)
  time    (GPU0, all GPUs idle) bs32 latency of every config of a backbone in one session, three interleaved rounds
  report  (CPU) differences to the full method, McNemar + Holm per split over all non-identical contrasts
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
import leaf.reproduction.experiments.ablation_and_alpha_grid as A

ROOT = REPO / 'results/station_discount'
FIT_BBS = A.P1_BBS                      # backbones with random schedules
RHO_GRID = np.round(np.arange(.20, 1.0001, .01), 2)
load, save, sha = A.load, A.save, A.sha


def claim(*parts):
    import os
    p = ROOT.joinpath('claims', *parts[:-1], parts[-1] + '.lock'); p.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.close(os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)); return True
    except FileExistsError:
        return False


# ----------------------------------------------------------------------------------------------- fit
def h_terms(bb):
    """Per-station probed damages of the sealed random schedules (replays the draw of A.h_schedules) and their F."""
    from leaf.planner.configs import config_from_state
    from leaf.planner.plan import operator_config
    from leaf.evaluation.runner import config_hash
    c = A.Ctx(bb); spec = c.spec; rng = np.random.default_rng(A.H_SEED + A.P0_BBS.index(bb)); m3 = c.base['M3']
    sealed = {r['id']: r for r in A.h_schedules(bb)['schedules']}; rows = []
    for b in c.b3:
        L, made = b['latency_ms'], 0
        while made < A.H_SCHEDULES:
            k = int(rng.integers(2, 9)); sites = sorted(rng.choice(np.arange(1, spec.depth), size=k, replace=False).tolist())
            ratios = rng.uniform(.5, .97, size=k); thetas = [float(rng.choice(A.THETA_BASE)) for _ in sites]
            alphas = [0. if l == spec.depth - 1 else float(rng.choice(A.ALPHA_BASE)) for l in sites]

            def at(eta):
                n, counts = spec.n_tokens, []
                for r in ratios:
                    cnt = max(3, min(n - 1, round(n * r ** eta)))
                    if cnt >= n: return None
                    counts.append(cnt); n = cnt
                cfg = A.to_alpha_config(operator_config(config_from_state(dict(sites=sites, counts=counts, thetas=thetas, ks=alphas), spec, c.d), 'M3', True))
                return cfg, counts, c.cost(cfg, 'M3')
            ok = [v for e in np.geomspace(.01, 20, 64) if (v := at(float(e))) is not None and v[2] <= L - A.MARGIN]
            if not ok: continue
            cfg, counts, _ = ok[0]; rid = f'{bb}-h-{config_hash(cfg)}'
            n, f = spec.n_tokens, []
            for l, cnt, th, a in zip(sites, counts, thetas, alphas): f.append(float(m3[(th, a)].transition(l, n, cnt))); n = cnt
            if rid not in sealed or abs(sum(f) - sealed[rid]['fhat']) > 1e-9: raise RuntimeError(f'replay differs from the sealed schedule {rid}')
            done = load(A.ROOT / bb / 'surrogate_check_done' / f'{rid}.json')
            if done['F'] is not None: rows.append(dict(budget=b['id'], F=done['F'], f=f, id=rid, config=cfg,
                                                          st=dict(sites=sites, counts=counts, thetas=thetas, alphas=alphas)))
            made += 1
    return rows


def within(rows, rho):
    """Spearman(corrected F-hat, F) of every budget."""
    from scipy.stats import spearmanr
    out = {}
    for b in sorted({r['budget'] for r in rows}):
        sel = [r for r in rows if r['budget'] == b]
        out[b] = float(spearmanr([sum(rho ** i * v for i, v in enumerate(r['f'])) for r in sel], [r['F'] for r in sel])[0])
    return out


def best_rho(groups):
    return float(max(RHO_GRID, key=lambda r: np.mean([v for g in groups for v in within(g, r).values()])))


def fit():
    dest = ROOT / 'fit.json'
    if dest.exists(): print(json.dumps(load(dest), indent=1)); return load(dest)
    rows = {bb: h_terms(bb) for bb in FIT_BBS}
    rho = best_rho(list(rows.values()))
    lobo = {}
    for bb in FIT_BBS:
        for b in sorted({r['budget'] for r in rows[bb]}):
            train = [[r for r in rows[x] if not (x == bb and r['budget'] == b)] for x in FIT_BBS]
            r_ = best_rho(train)
            lobo[f'{bb}/{b}'] = dict(rho=r_, rho_fhat=within([r for r in rows[bb] if r['budget'] == b], 1.)[b],
                                     rho_corrected=within([r for r in rows[bb] if r['budget'] == b], r_)[b])
    lobb = {bb: dict(rho=(r_ := best_rho([rows[x] for x in FIT_BBS if x != bb])), fhat=within(rows[bb], 1.), corrected=within(rows[bb], r_)) for bb in FIT_BBS}
    doc = dict(rho=rho, grid=[float(RHO_GRID[0]), float(RHO_GRID[-1]), .01], n={bb: len(v) for bb, v in rows.items()},
               within_fhat={bb: within(rows[bb], 1.) for bb in FIT_BBS}, within_corrected={bb: within(rows[bb], rho) for bb in FIT_BBS},
               leave_one_budget_out=lobo, leave_one_backbone_out=lobb, time=time.time())
    save(dest, doc); print(json.dumps(doc, indent=1))
    return doc


# ----------------------------------------------------------------------------------------------- plan
def plan(bbs=A.ALL_BBS):
    rho = load(ROOT / 'fit.json')['rho']
    for bb in bbs:
        dest = ROOT / bb / 'plans.json'
        if dest.exists(): continue
        c = A.Ctx(bb); full = A.load(A.ROOT / bb / 'plans' / 'leaf.json')['jobs']
        cv = A.curves_from(c.records['M3'] + A.new_records(bb, 'alpha_ext') + A.new_records(bb, 'alpha_fine'), bb)
        jobs = {'ref/dense': dict(config=dict(kind='dense'), status='reference', budget=None, group='reference')}
        for b in c.budgets:
            ref = full[f"leaf/alpha7/{b['id']}"]
            jobs[f"ref/leaf/{b['id']}"] = dict(config=ref['config'], status='sealed', budget=b, group='reference',
                                               K=len(ref['state']['sites']), predicted_ms=ref.get('predicted_ms'), gmac=ref.get('gmac'))
            res = c.plan(cv, b['latency_ms'], discount=rho)
            jobs[f"discount/{b['id']}"] = dict(res, budget=b, group='discount', rho=rho, K=len(res['state']['sites']) if 'state' in res else None)
        jobs = A._shas(jobs)
        save(dest, dict(backbone=bb, rho=rho, jobs=jobs, sha=sha(jobs), time=time.time()))
        ch = sum(jobs[f"discount/{b['id']}"].get('config_sha') != jobs[f"ref/leaf/{b['id']}"]['config_sha'] for b in c.budgets)
        print(bb, f'{len(c.budgets)} budgets, {ch} changed, K', [jobs[f"ref/leaf/{b['id']}"]['K'] for b in c.budgets], '->',
              [jobs[f"discount/{b['id']}"]['K'] for b in c.budgets], flush=True)


def tasks(bbs=A.ALL_BBS):
    out = {}
    for bb in bbs:
        p = ROOT / bb / 'plans.json'
        if not p.exists(): continue
        for name, row in load(p)['jobs'].items():
            if 'config' not in row: continue
            t = out.setdefault(f"{bb}-{row['config_sha']}", dict(id=f"{bb}-{row['config_sha']}", backbone=bb, config=row['config'], names=[]))
            t['names'].append(name)
    return sorted(out.values(), key=lambda t: (A.ALL_BBS.index(t['backbone']), t['id']))


def runner(bb):
    from leaf.reproduction.runners.alpha_runner import alpha_runner
    return alpha_runner(bb, ROOT, reuse_roots=[A.ROOT / bb, A.EQUIV / bb])


def worker():
    import torch
    from leaf.reproduction.runners.exclusive_timing import guard_foreign
    guard_foreign(); r, bb_last = None, None
    for split in A.SPLITS:
        for t in tasks():
            done = ROOT / 'completed' / split / f"{t['id']}.json"
            if done.exists() or not claim(split, t['id']): continue
            if t['backbone'] != bb_last:
                r = None; gc.collect(); torch.cuda.empty_cache(); r = runner(t['backbone']); bb_last = t['backbone']
            o = r.run(t['config'], split)
            save(done, dict(status='ok' if o is not None else 'infeasible', names=t['names'],
                            top1=None if o is None else float(o['correct'].mean() * 100), time=time.time()))
            print(split, t['id'], t['names'][0], flush=True)


def timing(bbs=A.ALL_BBS, rounds=3):
    """bs32 latency with every GPU idle, after all accuracy work; one session per backbone, middle round reversed."""
    import torch
    from leaf.reproduction.runners.alpha_runner import build_any
    from leaf.latency.timing import measure_latency
    from leaf.reproduction.runners.exclusive_timing import measure_exclusive
    missing = [t['id'] for t in tasks(bbs) for s in A.SPLITS if not (ROOT / 'completed' / s / f"{t['id']}.json").exists()]
    if missing: raise RuntimeError(f'{len(missing)} accuracy evaluations missing; timing runs last')
    for bb in bbs:
        if (ROOT / bb / 'latency_done.json').exists(): continue
        r = runner(bb); _ = r.vit; ts = tasks((bb,))
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
        print(bb, 'timed', len(ts), 'configs', flush=True)


# ----------------------------------------------------------------------------------------------- report
def correct(bb, cfg, split):
    from leaf.evaluation.runner import config_hash
    for base in (ROOT / bb, A.ROOT / bb, A.EQUIV / bb):
        p = base / split / 'runs' / f'{config_hash(cfg)}.npz'
        if p.exists(): return np.load(p)['correct'].astype(bool)
    raise FileNotFoundError(f'{bb} {split} {config_hash(cfg)}')


def latency(bb, sha_):
    vals = [load(p)['measurement']['ms_mean'] for p in sorted((ROOT / bb / 'latency').glob(f'{bb}-{sha_}#*.json'))]
    return float(np.mean(vals)) if vals else None


def report():
    from leaf.evaluation.data import mcnemar, holm
    rows = []
    for bb in A.ALL_BBS:
        p = ROOT / bb / 'plans.json'
        if not p.exists(): continue
        jobs = load(p)['jobs']
        for name, row in jobs.items():
            if not name.startswith('discount/') or 'config' not in row: continue
            bid = name.split('/', 1)[1]; ref = jobs[f'ref/leaf/{bid}']
            t = dict(backbone=bb, budget=bid, K=row['K'], K_ref=ref['K'], same=row['config_sha'] == ref['config_sha'],
                     ms=latency(bb, row['config_sha']), ms_ref=latency(bb, ref['config_sha']), gmac=row.get('gmac'), gmac_ref=ref.get('gmac'))
            for split in A.SPLITS:
                a, b = correct(bb, row['config'], split), correct(bb, ref['config'], split)
                pv, _, _ = mcnemar(a, b); t[split] = dict(delta=float(100 * (a.mean() - b.mean())), p=pv, top1=float(100 * a.mean()))
            rows.append(t)
    for split in A.SPLITS:
        holm([t[split] for t in rows if not t['same']])
        for t in rows:
            if t['same']: t[split]['holm'] = 1.
    ch = [t for t in rows if not t['same']]
    timed = [t for t in rows if t['ms'] and t['ms_ref']]
    summary = dict(rho=load(ROOT / 'fit.json')['rho'], n=len(rows), changed=len(ch),
                   mean_K=float(np.mean([t['K'] for t in rows])), mean_K_ref=float(np.mean([t['K_ref'] for t in rows])),
                   **{f'mean_delta_{s}': float(np.mean([t[s]['delta'] for t in rows])) for s in A.SPLITS},
                   **{f'win_{s}': sum(t[s]['holm'] < .05 and t[s]['delta'] > 0 for t in rows) for s in A.SPLITS},
                   **{f'loss_{s}': sum(t[s]['holm'] < .05 and t[s]['delta'] < 0 for t in rows) for s in A.SPLITS},
                   mean_latency_change_pct=float(np.mean([100 * (t['ms'] / t['ms_ref'] - 1) for t in timed])) if timed else None)
    save(ROOT / 'report.json', dict(summary=summary, rows=rows, time=time.time()))
    print(json.dumps(summary, indent=1))
    for t in rows:
        cell = lambda s: f"{t[s]['delta']:+.2f}" + ('*' if t[s]['holm'] < .05 else '')
        print(f"{t['backbone']:9s} {t['budget']:8s} K {t['K_ref']}->{t['K']} {'same' if t['same'] else '    '} "
              f"ms {t['ms'] or 0:.2f}/{t['ms_ref'] or 0:.2f}  val {cell('val')}  v2 {cell('v2')}")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=('fit', 'plan', 'worker', 'time', 'report'))
    ap.add_argument('--bbs', nargs='*', default=list(A.ALL_BBS))
    a = ap.parse_args()
    if a.stage == 'fit': fit()
    elif a.stage == 'plan': plan(tuple(a.bbs))
    elif a.stage == 'worker': worker()
    elif a.stage == 'time': timing(tuple(a.bbs))
    else: report()


if __name__ == '__main__':
    main()
