"""Batch-size sensitivity of the main comparison: do the latency-matched conclusions hold at batch 64 and 128?

For each backbone, every arm of the main comparison that was timed (the released LEAF schedules of the alpha7 grid,
the baseline grid points kept by the timing selection, the dense model) is timed again in ONE exclusive session on one
card with the other card idle, at batch 32, 64 and 128, three interleaved rounds (50 graph replays at batch 32,
20 at 64 and 128). Accuracy is batch-independent and reused from the sealed campaigns (val and V2 per-image
correctness), so no new accuracy evaluation is needed. At every batch, each LEAF schedule is paired with the point of
every family nearest in measured latency within tau = max(0.05 ms, 2 SD), McNemar with Holm correction per split and
batch; the batch-32 session also serves as a replication of the main session.

Stages (python leaf/reproduction/experiments/batch_sensitivity.py <stage> [--backbones deit_b,deit_s,mae_b]):
  time    (GPU)  one session per backbone -> results/batch_sensitivity/<bb>/latency, latency_done.json
  report  (CPU)  pairs and summaries per batch -> report.json, table.tex
Nothing in results/ is overwritten: a finished backbone is skipped.
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

import leaf.reproduction.experiments.campaign_reports as q
import leaf.reproduction.experiments.ablation_and_alpha_grid as ab
from leaf.reproduction.experiments import search_baselines as SB
from leaf.evaluation.runner import config_hash

ROOT = REPO / 'results/batch_sensitivity'
BBS = ('deit_b', 'deit_s', 'mae_b')
BATCHES = (32, 64, 128)
ROUNDS = 3


def load(p): return json.loads(Path(p).read_text())


def save(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(obj, indent=1, sort_keys=True, default=float))


def tasks(bb):
    """Every arm: id, config, name, family, and where its per-image correctness lives."""
    out, seen = [], set()
    for a in q.main_arms(bb):                       # timed baselines and dense of the main campaign (family from the campaign)
        if a['family'] == 'LEAF': continue           # the alpha4 check schedules; the released ones come from the alpha7 group
        if a['sha'] in seen: continue
        seen.add(a['sha'])
        cfg = load(q.MAIN_COMPARISON / bb / 'plans.json')['jobs'][a['name']]['config']
        out.append(dict(id=f"{bb}-{a['sha']}", config=cfg, name=a['name'], family=a['family'], src='main', sha=a['sha']))
    dense = dict(kind='dense')
    if config_hash(dense) not in seen:                # the dense model, for the speed-ups
        seen.add(config_hash(dense))
        out.append(dict(id=f'{bb}-{config_hash(dense)}', config=dense, name='ref/dense', family='dense', src='main', sha=config_hash(dense)))
    for name, row in ab.sealed_jobs(bb).items():
        if not name.startswith('leaf/alpha7/') or 'config' not in row: continue
        sha = config_hash(row['config'])
        if sha in seen: continue
        seen.add(sha)
        out.append(dict(id=f'{bb}-{sha}', config=row['config'], name=name, family='LEAF', src='abl', sha=sha, budget=row['budget']['latency_ms']))
    return sorted(out, key=lambda t: (0 if t['family'] in ('LEAF', 'dense', 'DiffRate') else 1, t['id']))


def correct(bb, t, split):
    if t['src'] == 'main': return q.correct(q.MAIN_COMPARISON / bb, t['sha'], split)
    a = ab.arrays(bb, t['config'], split)
    return None if a is None else a['correct']


def runner(bb):
    """The alpha runner: builds the released 'pda' LEAF schedules as well as the campaign's baselines (the campaign
    runner alone raises 'unknown baseline pda'; the DeiT-B session of 2026-10-25 therefore has no LEAF timings)."""
    from leaf.reproduction.runners.alpha_runner import alpha_runner
    return alpha_runner(bb, ROOT, reuse_roots=[])


# ----------------------------------------------------------------------------------------------- timing (GPU)
def timing(bbs):
    import os
    import torch
    from leaf.reproduction.runners.exclusive_timing import cast_exclusive
    os.environ['LEAF_EXCLUSIVE_TIMING'] = '1'
    for bb in bbs:
        done = ROOT / bb / 'latency_done.json'
        if done.exists(): print('exists', done); continue
        while not SB.gpus_idle(): time.sleep(60)
        ts = tasks(bb); r = runner(bb); dest = ROOT / bb / 'latency'
        save(ROOT / bb / 'tasks.json', dict(backbone=bb, n=len(ts), tasks=[{k: v for k, v in t.items() if k != 'config'} for t in ts], time=time.time()))
        t0 = time.time()
        for rnd in range(ROUNDS):
            for i, t in enumerate(ts if rnd % 2 == 0 else ts[::-1]):
                if not SB.gpus_idle(): raise SystemExit('another GPU process appeared during timing; session discarded')
                p = dest / f"{t['id']}#{rnd}.json"
                if p.exists(): continue
                meas = {}
                try:
                    m = r.inference_model(t['config'])
                    for b in BATCHES:
                        try: meas[str(b)] = cast_exclusive(m, b, iters=50 if b <= 32 else 20, warmup=10 if b <= 32 else 5)
                        except torch.cuda.OutOfMemoryError: meas[str(b)] = 'OOM'; torch.cuda.empty_cache()
                except Exception as e:
                    meas = dict(status='error', error=repr(e))
                save(p, dict(id=t['id'], names=[t['name']], round=rnd, measurements=meas, device=torch.cuda.get_device_name(0), time=time.time()))
                m = None; gc.collect(); torch.cuda.empty_cache()
                if i % 20 == 0: print(f'{bb} round {rnd} {i + 1}/{len(ts)} {time.time() - t0:.0f}s', flush=True)
        save(done, dict(rounds=ROUNDS, n=len(ts), batches=list(BATCHES), time=time.time()))
        r = None; gc.collect(); torch.cuda.empty_cache()
        print('timed', bb, len(ts), flush=True)


# ----------------------------------------------------------------------------------------------- report (CPU)
def analyse(bb, bs):
    from leaf.evaluation.data import mcnemar, holm
    from leaf.reproduction.experiments.main_comparison_analysis import paired_ci
    ms = q.round_means(ROOT / bb / 'latency', bs)
    arms = [dict(t, ms=ms[t['id']][0], sd=ms[t['id']][1]) for t in tasks(bb) if t['id'] in ms]
    dense = next((a for a in arms if a['family'] == 'dense'), None)
    pairs = []
    for o in [a for a in arms if a['family'] == 'LEAF']:
        for fam in sorted({a['family'] for a in arms} - {'LEAF', 'dense'}):
            best = min((a for a in arms if a['family'] == fam), key=lambda a: abs(a['ms'] - o['ms']))
            t = q.tau((o['ms'], o['sd']), (best['ms'], best['sd']))
            if abs(best['ms'] - o['ms']) > t: continue
            rec = dict(backbone=bb, batch=bs, ours=o['name'], family=fam, ref=best['name'], ours_ms=o['ms'], ref_ms=best['ms'], tau=t)
            for s in ('val', 'v2'):
                ca, cb = correct(bb, o, s), correct(bb, best, s)
                if ca is None or cb is None: continue
                p, _, _ = mcnemar(ca, cb)
                rec[s] = dict(delta=float((ca.mean() - cb.mean()) * 100), p=p, **{f'ci_{k}': v for k, v in paired_ci(ca, cb).items()})
            pairs.append(rec)
    for s in ('val', 'v2'): holm([r[s] for r in pairs if s in r])
    summary = {}
    for r in pairs:
        f = summary.setdefault(r['family'], dict(pairs=0, **{s: dict(win=0, loss=0, deltas=[]) for s in ('val', 'v2')}))
        f['pairs'] += 1
        for s in ('val', 'v2'):
            if s not in r: continue
            f[s]['deltas'].append(r[s]['delta'])
            if r[s]['holm'] < .05: f[s]['win' if r[s]['delta'] > 0 else 'loss'] += 1
    for f in summary.values():
        for s in ('val', 'v2'): f[s]['mean'] = float(np.mean(f[s]['deltas'])) if f[s]['deltas'] else None
    points = [dict(name=a['name'], family=a['family'], ms=a['ms'], sd=a['sd'], speedup=(dense['ms'] / a['ms']) if dense else None,
                   budget=a.get('budget')) for a in arms]
    return dict(backbone=bb, batch=bs, pairs=pairs, summary=summary, points=points, dense_ms=dense['ms'] if dense else None)


def report(bbs):
    out = {}
    for bb in bbs:
        if not (ROOT / bb / 'latency_done.json').exists(): print('not timed', bb); continue
        out[bb] = {str(bs): analyse(bb, bs) for bs in BATCHES}
        # how the batch-32 replication compares with the main session, and how speed-ups move with the batch
        main_ms = q.round_means(q.MAIN_COMPARISON / bb / 'latency', 32)
        sha_of = {t['name']: t['id'] for t in tasks(bb)}
        same = [(p['ms'], main_ms[sha_of[p['name']]][0]) for p in out[bb]['32']['points'] if sha_of.get(p['name']) in main_ms]
        if same:
            ratio = np.array([a / b for a, b in same]); out[bb]['replication_vs_main_bs32'] = dict(n=len(same), ratio_mean=float(ratio.mean()), ratio_sd=float(ratio.std()))
        for bs in BATCHES:
            r = out[bb][str(bs)]
            leaf = [p for p in r['points'] if p['family'] == 'LEAF']
            r['leaf_speedup_mean'] = float(np.mean([p['speedup'] for p in leaf])) if leaf and leaf[0]['speedup'] else None
            r['totals'] = dict(pairs=sum(f['pairs'] for f in r['summary'].values()),
                               **{f'{s}_{k}': sum(f[s][k] for f in r['summary'].values()) for s in ('val', 'v2') for k in ('win', 'loss')})
    save(ROOT / 'report.json', dict(batches=list(BATCHES), backbones=out, time=time.time()))
    lines = [r'\begin{tabular}{l r r rr rr r}\toprule', r'Backbone & Batch & Pairs & \multicolumn{2}{c}{Val W/L, $\bar\Delta$} & \multicolumn{2}{c}{V2 W/L, $\bar\Delta$} & LEAF speed-up \\ \midrule']
    for bb, d in out.items():
        for bs in BATCHES:
            r = d[str(bs)]; t = r['totals']; dv = [x for f in r['summary'].values() for x in f['val']['deltas']]; d2 = [x for f in r['summary'].values() for x in f['v2']['deltas']]
            lines.append(f"{q.NAMES[bb]} & {bs} & {t['pairs']} & {t['val_win']}/{t['val_loss']} & {np.mean(dv):+.2f} & {t['v2_win']}/{t['v2_loss']} & {np.mean(d2):+.2f} & "
                         f"{r['leaf_speedup_mean']:.2f}$\\times$ \\\\" if dv else f"{q.NAMES[bb]} & {bs} & 0 & -- & -- & -- & -- & -- \\\\")
            print(f"{bb:7s} bs{bs:3d}: {t['pairs']:3d} pairs, val {t['val_win']}/{t['val_loss']} mean {np.mean(dv) if dv else 0:+.2f}, v2 {t['v2_win']}/{t['v2_loss']} mean {np.mean(d2) if d2 else 0:+.2f}, LEAF speed-up {r['leaf_speedup_mean']}")
    lines.append(r'\bottomrule\end{tabular}')
    (ROOT / 'table.tex').write_text('\n'.join(lines) + '\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=('tasks', 'time', 'report'))
    ap.add_argument('--backbones', default=','.join(BBS))
    a = ap.parse_args(); bbs = a.backbones.split(',')
    if a.stage == 'tasks':
        for bb in bbs:
            ts = tasks(bb); fams = {}
            for t in ts: fams[t['family']] = fams.get(t['family'], 0) + 1
            print(bb, len(ts), 'arms', fams)
    elif a.stage == 'time': timing(bbs)
    else: report(bbs)


if __name__ == '__main__':
    main()
