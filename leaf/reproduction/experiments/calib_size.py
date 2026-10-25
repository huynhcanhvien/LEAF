"""Calibration-set size: are the released schedules limited by the number of probe images?

The released schedules (ablation_and_alpha_grid, group leaf/alpha7) are planned from single-station probes measured
on the 5,000 even-indexed images ('half') of the class-balanced 10,000-image calibration set. This experiment re-plans
every latency budget of a backbone from the same probe configurations with
  n1000_s0..s2         1,000 images drawn from the 5,000 (three draws),
  n2500_s0, n2500_s1   2,500 images (draws 0 and 1, as in the probe-sample ablation),
  n5000                the 5,000 images (must reproduce the released schedule),
  n5000_odd            the other 5,000 images alone (an independent calibration set of the same size),
  n10000               all 10,000 images: the 5,000 even images plus the 5,000 odd ones ('odd'), probed here.
Each re-planned schedule is evaluated on ImageNet val and V2 and timed in one session with the released ones.

Stages (python leaf/reproduction/experiments/calib_size.py <stage> [--backbones deit_b,deit_s,mae_b]; results in ROOT):
  probe   (GPU)  the 1,917 alpha7 probe configurations on the odd half -> <bb>/odd/runs, <bb>/probes_odd.json
                 (several processes may share a backbone: claims under <bb>/claims)
  plan    (CPU)  re-plan every budget from each calibration subset -> <bb>/plans.json (immutable)
  val     (GPU)  ImageNet val and V2 of every distinct re-planned schedule (released ones reused)
  time    (GPU)  bs32 latency of all schedules and dense, one GPU, both GPUs otherwise idle -> latency.json
  report  (CPU)  top-1, McNemar + Holm of every subset against the released schedules, latency -> report.json
Nothing in results/ is overwritten: every stage skips what exists.
"""
import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from leaf.reproduction.experiments import ablation_and_alpha_grid as A
from leaf.reproduction.experiments import dp_verify as DV
from leaf.reproduction.experiments import search_baselines as SB

ROOT = REPO / 'results/calib_size'
BBS = ('deit_b', 'deit_s', 'mae_b')
SUBSETS = (('n1000_s0', 1000, 0), ('n1000_s1', 1000, 1), ('n1000_s2', 1000, 2), ('n2500_s0', 2500, 0), ('n2500_s1', 2500, 1),
           ('n5000', 5000, None), ('n5000_odd', 5000, 'odd'), ('n10000', 10000, None))
NEW_SETS = ('alpha_ext', 'alpha_fine')


def load(p): return json.loads(Path(p).read_text())


def save(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(p) + '.tmp'); tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, default=float)); tmp.replace(p)


def alpha7_records(bb, c):
    """The released probe records (5,000 images): historical M3 probes + the alpha_ext / alpha_fine sets."""
    recs = list(c.records['M3'])
    for name in NEW_SETS:
        r = A.new_records(bb, name)
        if r is None: raise RuntimeError(f'{bb}: probe set {name} missing')
        recs += r
    return recs


def rec_key(rec): return (rec['block'], rec['count'], float(rec['theta']), float(rec['alpha']))


def runner(bb):
    from leaf.reproduction.runners.alpha_runner import alpha_runner
    return alpha_runner(bb, ROOT, reuse_roots=[A.ROOT / bb, DV.ROOT / bb, SB.ROOT / bb, A.EQUIV / bb])


def claim(bb, tid):
    p = ROOT / bb / 'claims' / f'{tid}.lock'; p.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.close(os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)); return True
    except FileExistsError:
        return False


# ----------------------------------------------------------------------------------------------- probe (GPU)
def probe(bbs):
    """Every alpha7 probe configuration on the odd half of the calibration set."""
    import torch
    from leaf.evaluation.runner import config_hash
    from leaf.reproduction.runners.exclusive_timing import guard_foreign
    guard_foreign()
    for bb in bbs:
        dest = ROOT / bb / 'probes_odd.json'
        if dest.exists(): print('exists', dest); continue
        c = A.Ctx(bb); recs = alpha7_records(bb, c); r = None; t0 = time.time()
        for i, rec in enumerate(recs):
            tid = config_hash(rec['config'])
            done = ROOT / bb / 'odd' / 'runs' / f'{tid}.npz'
            if done.exists() or not claim(bb, tid): continue
            if r is None: r = runner(bb)
            arr = r.run(rec['config'], 'odd')
            if arr is None: raise RuntimeError(f'{bb}: infeasible probe {rec_key(rec)}')
            if i % 50 == 0: print(f'{bb} probe {i + 1}/{len(recs)} kl_odd={float(arr["kl"].mean()):.5f} {time.time() - t0:.0f}s', flush=True)
        collect(bb, recs)
        del r; gc.collect(); torch.cuda.empty_cache()


def collect(bb, recs=None):
    """<bb>/probes_odd.json once every probe of the backbone is done (by any process)."""
    from leaf.evaluation.runner import config_hash
    dest = ROOT / bb / 'probes_odd.json'
    if dest.exists(): return True
    recs = recs or alpha7_records(bb, A.Ctx(bb)); rows = []
    for rec in recs:
        p = ROOT / bb / 'odd' / 'runs' / f"{config_hash(rec['config'])}.npz"
        if not p.exists(): return False
        kl = np.load(p)['kl']
        rows.append(dict({k: v for k, v in rec.items() if k not in ('kl', 'arrays', 'metadata', 'metadata_sha', 'arrays_sha')},
                         kl_half=rec['kl'], arrays_half=rec['arrays'], kl_odd=float(kl.mean()), n_odd=int(kl.size), arrays_odd=str(p.relative_to(REPO))))
    save(dest, rows)
    print(bb, 'collected', len(rows), 'odd probes', flush=True)
    return True


# ----------------------------------------------------------------------------------------------- plan (CPU)
def subset_curves(bb, c, recs, name, n, seed):
    if name in ('n10000', 'n5000_odd'):
        odd = {rec_key(r): r for r in load(ROOT / bb / 'probes_odd.json')}
        merged = []
        for r in recs:
            o = odd[rec_key(r)]
            kl = float(o['kl_odd']) if name == 'n5000_odd' else 0.5 * (float(r['kl']) + float(o['kl_odd']))   # odd half alone, or mean over 10,000 images
            merged.append(dict(r, kl=kl))
        return A.curves_from(merged, bb)
    if n == A.CALIB: return A.curves_from(recs, bb)
    idx = np.sort(np.random.default_rng(seed).choice(A.CALIB, size=n, replace=False))
    return A.curves_from(recs, bb, idx=idx)


def plan(bbs):
    for bb in bbs:
        dest = ROOT / bb / 'plans.json'
        if dest.exists(): print('exists', dest); continue
        if not collect(bb): raise RuntimeError(f'{bb}: odd probes incomplete')
        c = A.Ctx(bb); recs = alpha7_records(bb, c); ref = DV.released(bb); jobs = {}
        for name, n, seed in SUBSETS:
            cv = subset_curves(bb, c, recs, name, n, seed)
            for b in c.budgets:
                bid, L = b['id'], b['latency_ms']
                if bid not in ref: continue
                res = c.plan(cv, L)
                if 'config' not in res: jobs[f'{name}/{bid}'] = dict(res, budget=b, subset=name, n_images=n, seed=seed); continue
                same = A.sha(res['config']) == A.sha(ref[bid]['config'])
                jobs[f'{name}/{bid}'] = dict(config=res['config'], state=res['state'], fhat=res['fhat'], predicted_ms=res['predicted_ms'],
                                             budget=b, subset=name, n_images=n, seed=seed, same_as_released=same)
                if name == 'n5000' and not same: print(f'WARNING {bb} {bid}: n5000 re-plan differs from the released schedule', flush=True)
        for bid, r in ref.items():
            jobs[f'released/{bid}'] = dict(config=r['config'], state=r['state'], fhat=r.get('fhat'), predicted_ms=r.get('predicted_ms'),
                                           budget=next(b for b in c.budgets if b['id'] == bid), subset='released', n_images=A.CALIB, seed=None)
        jobs = A._shas(jobs)
        save(dest, dict(backbone=bb, jobs=jobs, sha=A.sha(jobs), subsets=[s[0] for s in SUBSETS], time=time.time()))
        n_distinct = len({j['config_sha'] for j in jobs.values() if 'config' in j})
        changed = {s[0]: sum(1 for k, j in jobs.items() if k.startswith(s[0] + '/') and 'config' in j and not j['same_as_released']) for s in SUBSETS}
        print(bb, len(jobs), 'jobs,', n_distinct, 'distinct schedules; changed vs released:', changed, flush=True)


# ----------------------------------------------------------------------------------------------- val (GPU)
def val(bbs, splits=('val', 'v2')):
    import torch
    from leaf.reproduction.runners.exclusive_timing import guard_foreign
    guard_foreign()
    for bb in bbs:
        jobs = load(ROOT / bb / 'plans.json')['jobs']; r = runner(bb); seen = set()
        for name, job in jobs.items():
            if 'config' not in job or job['config_sha'] in seen: continue
            seen.add(job['config_sha'])
            for split in splits:
                arr = r.run(job['config'], split)
                print(f'{bb} {name} {split} top1={None if arr is None else 100 * float(arr["correct"].mean()):.2f}', flush=True)
        del r; gc.collect(); torch.cuda.empty_cache()


# ----------------------------------------------------------------------------------------------- timing (GPU)
def timing(bbs, rounds=3):
    import torch
    from leaf.reproduction.runners.exclusive_timing import cast_exclusive
    os.environ['LEAF_EXCLUSIVE_TIMING'] = '1'
    if (ROOT / 'latency.json').exists(): print('exists'); return
    while not SB.gpus_idle(): time.sleep(60)
    jobs, cfgs = [], {}
    for bb in bbs:
        plans = load(ROOT / bb / 'plans.json')['jobs']
        cfgs[f'{bb}/dense'] = dict(kind='dense'); jobs.append((bb, f'{bb}/dense'))
        for name, job in sorted(plans.items()):
            if 'config' not in job: continue
            key = f"{bb}/{job['config_sha']}"
            if key in cfgs: continue
            cfgs[key] = job['config']; jobs.append((bb, key))
    runners = {bb: runner(bb) for bb in bbs}; dest = ROOT / 'latency'
    for rnd in range(rounds):
        for bb, name in (jobs if rnd % 2 == 0 else jobs[::-1]):
            if not SB.gpus_idle(): raise SystemExit('another GPU process appeared during timing; session discarded')
            p = dest / f"{name.replace('/', '__')}#{rnd}.json"
            if p.exists(): continue
            m = runners[bb].inference_model(cfgs[name])
            meas = cast_exclusive(m, 32, iters=50, warmup=10)
            save(p, dict(name=name, round=rnd, ms=meas['ms_mean'], measurement=meas, device=torch.cuda.get_device_name(0), time=time.time()))
            m = None; gc.collect(); torch.cuda.empty_cache()
    out = {}
    for _, name in jobs:
        ms = [load(dest / f"{name.replace('/', '__')}#{r}.json")['ms'] for r in range(rounds)]
        out[name] = dict(ms=float(np.mean(ms)), sd=float(np.std(ms, ddof=1)), rounds=ms)
    save(ROOT / 'latency.json', dict(batch=32, rounds=rounds, device=torch.cuda.get_device_name(0), time=time.time(), latency=out))


# ----------------------------------------------------------------------------------------------- report (CPU)
def correct(bb, cfg, split):
    from leaf.evaluation.runner import config_hash
    for base in (ROOT / bb, A.ROOT / bb, DV.ROOT / bb, SB.ROOT / bb, A.EQUIV / bb):
        p = base / split / 'runs' / f'{config_hash(cfg)}.npz'
        if p.exists(): return np.load(p)['correct'].astype(bool)
    return None


def report(bbs):
    from leaf.evaluation.data import mcnemar, holm
    lat = load(ROOT / 'latency.json')['latency'] if (ROOT / 'latency.json').exists() else {}
    rows = []
    for bb in bbs:
        jobs = load(ROOT / bb / 'plans.json')['jobs']
        for name, job in jobs.items():
            sub, bid = name.split('/')
            if sub == 'released' or 'config' not in job: continue
            ref = jobs[f'released/{bid}']
            t = dict(backbone=bb, budget=bid, subset=sub, n_images=job['n_images'], seed=job['seed'], same=job['same_as_released'],
                     K=len(job['state']['sites']), K_ref=len(ref['state']['sites']), fhat=job['fhat'], fhat_ref=ref['fhat'],
                     predicted_ms=job['predicted_ms'], predicted_ms_ref=ref['predicted_ms'], budget_ms=job['budget']['latency_ms'],
                     ms=lat.get(f"{bb}/{job['config_sha']}", {}).get('ms'), ms_ref=lat.get(f"{bb}/{ref['config_sha']}", {}).get('ms'),
                     ms_dense=lat.get(f'{bb}/dense', {}).get('ms'))
            for split in A.SPLITS:
                a, b = correct(bb, job['config'], split), correct(bb, ref['config'], split)
                if a is None or b is None: t[split] = None; continue
                pv, _, _ = mcnemar(a, b)
                t[split] = dict(delta=float(100 * (a.mean() - b.mean())), p=pv, top1=float(100 * a.mean()), top1_ref=float(100 * b.mean()))
            rows.append(t)
    for sub, _, _ in SUBSETS:
        for split in A.SPLITS:
            holm([t[split] for t in rows if t['subset'] == sub and t[split] is not None and not t['same']])
            for t in rows:
                if t['subset'] == sub and t['same'] and t[split] is not None: t[split]['holm'] = 1.
    summary = {}
    for sub, n, seed in SUBSETS:
        rs = [t for t in rows if t['subset'] == sub]
        s = dict(n_images=n, seed=seed, n=len(rs), changed=sum(not t['same'] for t in rs),
                 mean_K=float(np.mean([t['K'] for t in rs])), mean_K_ref=float(np.mean([t['K_ref'] for t in rs])))
        for split in A.SPLITS:
            have = [t for t in rs if t[split] is not None]
            s[f'mean_delta_{split}'] = float(np.mean([t[split]['delta'] for t in have])) if have else None
            s[f'win_{split}'] = sum(t[split].get('holm', 1) < .05 and t[split]['delta'] > 0 for t in have)
            s[f'loss_{split}'] = sum(t[split].get('holm', 1) < .05 and t[split]['delta'] < 0 for t in have)
        timed = [t for t in rs if t['ms'] and t['ms_ref']]
        s['mean_dlat_pct'] = float(np.mean([100 * (t['ms'] / t['ms_ref'] - 1) for t in timed])) if timed else None
        s['met_budget'] = sum(t['ms'] <= t['budget_ms'] for t in timed) if timed else None
        s['met_budget_ref'] = sum(t['ms_ref'] <= t['budget_ms'] for t in timed) if timed else None
        summary[sub] = s
    save(ROOT / 'report.json', dict(summary=summary, rows=rows, backbones=list(bbs), time=time.time()))
    print(json.dumps(summary, indent=1))
    for t in rows:
        cell = lambda s: '  n/a ' if t[s] is None else f"{t[s]['delta']:+.2f}" + ('*' if t[s].get('holm', 1) < .05 else ' ')
        print(f"{t['backbone']:7s} {t['budget']:7s} {t['subset']:9s} {'same' if t['same'] else 'chg '} K {t['K_ref']}->{t['K']} "
              f"ms {t['ms_ref'] or 0:.2f}->{t['ms'] or 0:.2f}  val {cell('val')} v2 {cell('v2')}")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=('probe', 'collect', 'plan', 'val', 'time', 'report'))
    ap.add_argument('--backbones', default=','.join(BBS))
    a = ap.parse_args(); bbs = a.backbones.split(',')
    if a.stage == 'probe': probe(bbs)
    elif a.stage == 'collect': [collect(bb) for bb in bbs]
    elif a.stage == 'plan': plan(bbs)
    elif a.stage == 'val': val(bbs)
    elif a.stage == 'time': timing(bbs)
    else: report(bbs)


if __name__ == '__main__':
    main()
