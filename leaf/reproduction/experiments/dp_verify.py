"""DP + measured verification: does a cheap check of the planner's candidates on the measured calibration KL recover
the accuracy that regularised evolution found at the strongest budgets (search_baselines.py), without its cost?

For every latency budget of a backbone, the candidates are (i) the seven dynamic-program solutions at the multiples
{0.5, ..., 1.6} of the bisected price, each rescaled to the budget (the planner's own candidate set; the released
schedule is the one with the lowest F-hat), and (ii) N_MUT local mutants of the released schedule (search_baselines.mutate:
add, remove or move a station, or perturb one keep ratio; decoded to the SALT budget as there). Every candidate's
calibration KL F is measured on the 5,000 probe images ('half'); the candidate with the lowest F is selected. Same
probes (alpha7 grid), SALT model and budgets as the released schedules (ablation_and_alpha_grid, group leaf/alpha7).

Stages (python leaf/reproduction/experiments/dp_verify.py <stage> [--backbones deit_b,mae_b]; results in ROOT):
  plan    (CPU)  candidates per budget -> <bb>/plans.json (immutable)
  worker  (GPU)  F of every candidate and of the released schedule -> <bb>/half/runs (one process per backbone)
  select  (CPU)  lowest-F candidate per budget -> sealed.json (immutable)
  val     (GPU)  ImageNet val and V2 of the selected schedules (the released ones are reused from the ablation campaign)
  time    (GPU)  bs32 latency of selected, released and dense, one GPU, both GPUs otherwise idle -> latency.json
  report  (CPU)  F, top-1, McNemar + Holm of selected against released, latency -> report.json
Nothing in results/ is overwritten: every stage skips what exists.
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

from leaf.reproduction.experiments import ablation_and_alpha_grid as A
from leaf.reproduction.experiments import search_baselines as SB

ROOT = REPO / 'results/dp_verify'
BBS = ('deit_b', 'mae_b')
N_MUT = 24
SEED = 2611
FACTORS = (.5, .7, .85, 1., 1.15, 1.3, 1.6)


def load(p): return json.loads(Path(p).read_text())


def save(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(obj, indent=1, sort_keys=True, default=float))


def alpha7_curves(bb, c):
    recs = list(c.records['M3'])
    for name in ('alpha_ext', 'alpha_fine'):
        r = A.new_records(bb, name)
        if r: recs += r
    return A.curves_from(recs, bb)


def released(bb):
    jobs = A.load(A.ROOT / bb / 'plans' / 'leaf.json')['jobs']
    return {name.split('/')[-1]: row for name, row in jobs.items() if name.startswith('leaf/alpha7/') and 'config' in row}


def ratios_of(state, n_tokens):
    out, n = [], n_tokens
    for c in state['counts']: out.append(c / n); n = c
    return out


# ----------------------------------------------------------------------------------------------- plan (CPU)
def plan(bbs):
    for bb in bbs:
        dest = ROOT / bb / 'plans.json'
        if dest.exists(): print('exists', dest); continue
        c = A.Ctx(bb); cv = alpha7_curves(bb, c); ref = released(bb)
        jobs = {}
        for b in c.budgets:
            bid, L = b['id'], b['latency_ms']
            if bid not in ref: continue
            r = ref[bid]
            jobs[f'ref/{bid}'] = dict(config=r['config'], state=r['state'], fhat=r.get('fhat'), predicted_ms=r.get('predicted_ms'),
                                       budget=b, group='released')
            seen = {A.sha(r['config'])}
            for f in FACTORS:                                             # the planner's own candidate set
                res = c.plan(cv, L, factors=(f,))
                if 'config' not in res: continue
                k = A.sha(res['config'])
                if k in seen: continue
                seen.add(k)
                jobs[f'dp/{bid}/f{f:g}'] = dict(config=res['config'], state=res['state'], fhat=res['fhat'], predicted_ms=res['predicted_ms'],
                                                 budget=b, group='dp', factor=f)
            space = SB.Space(c, L); space.cv = cv                            # local mutants of the released schedule
            parent = (list(r['state']['sites']), ratios_of(r['state'], c.spec.n_tokens))
            rng = np.random.default_rng(SEED + 7 * c.budgets.index(b) + 101 * BBS.index(bb) if bb in BBS else SEED)
            n_try = 0
            while sum(1 for k in jobs if k.startswith(f'mut/{bid}/')) < N_MUT and n_try < 40 * N_MUT:
                n_try += 1
                row = SB.candidate(space, rng, parent)
                if row is None: break
                k = A.sha(row['config'])
                if k in seen: continue
                seen.add(k)
                i = sum(1 for k2 in jobs if k2.startswith(f'mut/{bid}/'))
                jobs[f'mut/{bid}/{i:02d}'] = dict(config=row['config'], state=row['state'], fhat=row['fhat'], predicted_ms=row['predicted_ms'],
                                                   budget=b, group='mutant', sites=row['sites'], ratios=row['ratios'], eta=row['eta'])
        jobs = A._shas(jobs)
        save(dest, dict(backbone=bb, jobs=jobs, sha=A.sha(jobs), n_mut=N_MUT, factors=list(FACTORS), time=time.time()))
        print(bb, len(jobs), 'candidates incl. released', {g: sum(1 for j in jobs.values() if j['group'] == g) for g in ('released', 'dp', 'mutant')}, flush=True)


# ----------------------------------------------------------------------------------------------- worker (GPU)
def runner(bb):
    from leaf.reproduction.runners.alpha_runner import alpha_runner
    return alpha_runner(bb, ROOT, reuse_roots=[A.ROOT / bb, SB.ROOT / bb, A.EQUIV / bb])


def worker(bbs, splits=('half',)):
    import torch
    from leaf.reproduction.runners.exclusive_timing import guard_foreign
    guard_foreign()
    for bb in bbs:
        jobs = load(ROOT / bb / 'plans.json')['jobs']; r = runner(bb)
        names = list(jobs)
        if splits != ('half',):
            sealed = load(ROOT / 'sealed.json')['jobs']
            names = [n for n in sealed if n.startswith(bb + '/')]
            jobs = {n: sealed[n] for n in names}
        for i, name in enumerate(names):
            for split in splits:
                arr = r.run(jobs[name]['config'], split)
                F = None if arr is None else float(arr['kl'].mean())
                if split == 'half': print(f'{bb} {i + 1}/{len(names)} {name} F={F}', flush=True)
                else: print(f'{bb} {name} {split} top1={None if arr is None else 100 * float(arr["correct"].mean()):.2f}', flush=True)
        del r; gc.collect(); torch.cuda.empty_cache()


def measured_F(bb, cfg):
    from leaf.evaluation.runner import config_hash
    for base in (ROOT / bb, A.ROOT / bb, SB.ROOT / bb, A.EQUIV / bb):
        p = base / 'half' / 'runs' / f'{config_hash(cfg)}.json'
        if p.exists(): return float(load(p)['kl'])
    return None


# ----------------------------------------------------------------------------------------------- select (CPU)
def select(bbs):
    dest = ROOT / 'sealed.json'
    if dest.exists(): print('exists', dest); return load(dest)
    out = {}
    for bb in bbs:
        jobs = load(ROOT / bb / 'plans.json')['jobs']
        for b in A.Ctx(bb).budgets:
            bid = b['id']
            rows = {n: j for n, j in jobs.items() if n.split('/')[1] == bid}
            Fs = {n: measured_F(bb, j['config']) for n, j in rows.items()}
            if any(v is None for v in Fs.values()):
                missing = [n for n, v in Fs.items() if v is None]
                raise RuntimeError(f'{bb} {bid}: {len(missing)} candidates without measured F, e.g. {missing[:3]}')
            best = min(rows, key=lambda n: (Fs[n], rows[n]['fhat']))
            out[f'{bb}/selected/{bid}'] = dict(rows[best], source=best, F=Fs[best], F_released=Fs[f'ref/{bid}'])
            out[f'{bb}/released/{bid}'] = dict(rows[f'ref/{bid}'], F=Fs[f'ref/{bid}'])
            print(bb, bid, 'selected', best, f"F {Fs[best]:.5f} vs released {Fs[f'ref/{bid}']:.5f}", flush=True)
    doc = dict(jobs=out, sha=A.sha(out), time=time.time())
    save(dest, doc)
    return doc


# ----------------------------------------------------------------------------------------------- timing (GPU)
def timing(rounds=3):
    import os
    import torch
    from leaf.reproduction.runners.exclusive_timing import cast_exclusive
    os.environ['LEAF_EXCLUSIVE_TIMING'] = '1'
    if (ROOT / 'latency.json').exists(): print('exists'); return
    while not SB.gpus_idle(): time.sleep(60)
    sealed = load(ROOT / 'sealed.json')['jobs']
    bbs = sorted({n.split('/')[0] for n in sealed}, key=lambda b: A.ALL_BBS.index(b))
    jobs = [(bb, name, job['config']) for bb in bbs for name, job in [(f'{bb}/dense', dict(config=dict(kind='dense')))] + sorted(sealed.items())
            if name.startswith(bb + '/')]
    runners = {bb: runner(bb) for bb in bbs}; dest = ROOT / 'latency'
    for rnd in range(rounds):
        for bb, name, cfg in (jobs if rnd % 2 == 0 else jobs[::-1]):
            if not SB.gpus_idle(): raise SystemExit('another GPU process appeared during timing; session discarded')
            p = dest / f"{name.replace('/', '__')}#{rnd}.json"
            if p.exists(): continue
            m = runners[bb].inference_model(cfg)
            meas = cast_exclusive(m, 32, iters=50, warmup=10)
            save(p, dict(name=name, round=rnd, ms=meas['ms_mean'], measurement=meas, device=torch.cuda.get_device_name(0), time=time.time()))
            m = None; gc.collect(); torch.cuda.empty_cache()
    out = {}
    for _, name, _ in jobs:
        ms = [load(dest / f"{name.replace('/', '__')}#{r}.json")['ms'] for r in range(rounds)]
        out[name] = dict(ms=float(np.mean(ms)), sd=float(np.std(ms, ddof=1)), rounds=ms)
    save(ROOT / 'latency.json', dict(batch=32, rounds=rounds, device=torch.cuda.get_device_name(0), time=time.time(), latency=out))


# ----------------------------------------------------------------------------------------------- report (CPU)
def correct(bb, cfg, split):
    from leaf.evaluation.runner import config_hash
    for base in (ROOT / bb, A.ROOT / bb, SB.ROOT / bb, A.EQUIV / bb):
        p = base / split / 'runs' / f'{config_hash(cfg)}.npz'
        if p.exists(): return np.load(p)['correct'].astype(bool)
    return None


def report():
    from leaf.evaluation.data import mcnemar, holm
    sealed = load(ROOT / 'sealed.json')['jobs']
    lat = load(ROOT / 'latency.json')['latency'] if (ROOT / 'latency.json').exists() else {}
    rows = []
    for name, job in sealed.items():
        bb, kind, bid = name.split('/')
        if kind != 'selected': continue
        ref = sealed[f'{bb}/released/{bid}']
        t = dict(backbone=bb, budget=bid, source=job['source'], same=A.sha(job['config']) == A.sha(ref['config']),
                 K=len(job['state']['sites']), K_ref=len(ref['state']['sites']), F=job['F'], F_ref=ref['F'],
                 fhat=job['fhat'], fhat_ref=ref['fhat'], predicted_ms=job['predicted_ms'], predicted_ms_ref=ref['predicted_ms'],
                 ms=lat.get(name, {}).get('ms'), ms_ref=lat.get(f'{bb}/released/{bid}', {}).get('ms'), ms_dense=lat.get(f'{bb}/dense', {}).get('ms'))
        n_eval = sum(1 for n in load(ROOT / bb / 'plans.json')['jobs'] if n.split('/')[1] == bid)
        t['n_candidates'] = n_eval
        for split in A.SPLITS:
            a, b = correct(bb, job['config'], split), correct(bb, ref['config'], split)
            if a is None or b is None: t[split] = None; continue
            pv, _, _ = mcnemar(a, b)
            t[split] = dict(delta=float(100 * (a.mean() - b.mean())), p=pv, top1=float(100 * a.mean()), top1_ref=float(100 * b.mean()))
        rows.append(t)
    for split in A.SPLITS:
        holm([t[split] for t in rows if t[split] is not None and not t['same']])
        for t in rows:
            if t['same'] and t[split] is not None: t[split]['holm'] = 1.
    summary = dict(n=len(rows), changed=sum(not t['same'] for t in rows),
                   mean_F_gain_pct=float(np.mean([100 * (1 - t['F'] / t['F_ref']) for t in rows])),
                   **{f'mean_delta_{s}': float(np.mean([t[s]['delta'] for t in rows if t[s]])) for s in A.SPLITS},
                   **{f'win_{s}': sum(t[s] is not None and t[s].get('holm', 1) < .05 and t[s]['delta'] > 0 for t in rows) for s in A.SPLITS},
                   **{f'loss_{s}': sum(t[s] is not None and t[s].get('holm', 1) < .05 and t[s]['delta'] < 0 for t in rows) for s in A.SPLITS},
                   mean_candidates=float(np.mean([t['n_candidates'] for t in rows])))
    save(ROOT / 'report.json', dict(summary=summary, rows=rows, time=time.time()))
    print(json.dumps(summary, indent=1))
    for t in rows:
        cell = lambda s: '  n/a ' if t[s] is None else f"{t[s]['delta']:+.2f}" + ('*' if t[s].get('holm', 1) < .05 else ' ')
        print(f"{t['backbone']:7s} {t['budget']:7s} {t['source']:14s} K {t['K_ref']}->{t['K']} F {t['F_ref']:.4f}->{t['F']:.4f} "
              f"ms {t['ms_ref'] or 0:.2f}->{t['ms'] or 0:.2f}  val {cell('val')} v2 {cell('v2')}")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=('plan', 'worker', 'select', 'val', 'time', 'report'))
    ap.add_argument('--backbones', default=','.join(BBS))
    a = ap.parse_args(); bbs = a.backbones.split(',')
    if a.stage == 'plan': plan(bbs)
    elif a.stage == 'worker': worker(bbs)
    elif a.stage == 'select': select(bbs)
    elif a.stage == 'val': worker(bbs, splits=('val', 'v2'))
    elif a.stage == 'time': timing()
    else: report()


if __name__ == '__main__':
    main()
