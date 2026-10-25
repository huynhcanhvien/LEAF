"""Final-block station of the latency-planned schedules of the final method (leaf/alpha7, the schedules of the main
comparison): how many have one, what removing it costs in latency, and whether it changes the predictions.

The earlier check (ablation_and_alpha_grid group final_station_all) used the leaf_alpha4 schedules; this one repeats the
latency-planned part on the alpha7 schedules so that the paper's numbers belong to the reported method. Every alpha7
schedule with a station in the last block gets a variant without it (A.without_final: the last-block caps set to the
incoming token count, nothing else changed); both are evaluated on val and V2 and timed in one exclusive session per
backbone, three interleaved rounds at batch 32. The GMAC-planned part of the paragraph uses the schedules of the
equal-GMAC section and is not repeated.

Stages: plan (CPU) | worker (GPU, one per card) | time (GPU0) | report (CPU)
Reuses the worker and timing of station_discount.py with ROOT pointed here.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
import leaf.reproduction.experiments.ablation_and_alpha_grid as A
import leaf.reproduction.experiments.station_discount as SD

ROOT = REPO / 'results/final_station_alpha7'
SD.ROOT = ROOT
load, save, sha = A.load, A.save, A.sha


def plan(bbs=A.ALL_BBS):
    for bb in bbs:
        dest = ROOT / bb / 'plans.json'
        if dest.exists(): continue
        spec = A.spec_of(bb); full = load(A.ROOT / bb / 'plans' / 'leaf.json')['jobs']; jobs = {}; n = 0
        for name, row in full.items():
            if not name.startswith('leaf/alpha7/') or 'config' not in row: continue
            bid = name.split('/')[-1]; cfg = row['config']; nf = A.without_final(cfg, spec); n += 1
            if min(cfg['prune_keep'][-1], cfg['merge_keep'][-1]) >= nf['prune_keep'][-1]: continue   # no station in the last block
            jobs[f'ref/leaf/{bid}'] = dict(config=cfg, status='sealed', budget=row.get('budget'), group='reference')
            jobs[f'no_final/{bid}'] = dict(config=nf, status='fixed', budget=row.get('budget'), group='no_final')
        jobs = A._shas(jobs)
        save(dest, dict(backbone=bb, n_schedules=n, jobs=jobs, sha=sha(jobs), time=time.time()))
        print(bb, n, 'schedules,', sum(k.startswith('no_final/') for k in jobs), 'with a final station', flush=True)


def report():
    from leaf.evaluation.data import mcnemar, holm
    rows, count = [], {}
    for bb in A.ALL_BBS:
        p = ROOT / bb / 'plans.json'
        if not p.exists(): continue
        jobs = load(p)['jobs']; spec = A.spec_of(bb)
        count[bb] = dict(n=load(p)['n_schedules'], final=sum(k.startswith('no_final/') for k in jobs))
        for name, row in jobs.items():
            if not name.startswith('no_final/'): continue
            bid = name.split('/', 1)[1]; ref = jobs[f'ref/leaf/{bid}']
            t = dict(backbone=bb, budget=bid, pool=spec.pool, ms=SD.latency(bb, row['config_sha']), ms_ref=SD.latency(bb, ref['config_sha']))
            for split in A.SPLITS:
                a, b = SD.correct(bb, row['config'], split), SD.correct(bb, ref['config'], split)
                pv, _, _ = mcnemar(a, b)
                t[split] = dict(delta=float(100 * (a.mean() - b.mean())), p=pv, changed=int((a != b).sum()))
            t['slowdown_pct'] = 100 * (t['ms'] / t['ms_ref'] - 1) if t['ms'] and t['ms_ref'] else None
            rows.append(t)
    for split in A.SPLITS: holm([t[split] for t in rows])
    tok = [t for t in rows if t['pool'] == 'token']; avg = [t for t in rows if t['pool'] != 'token']
    sl = lambda rs: [t['slowdown_pct'] for t in rs if t['slowdown_pct'] is not None]
    summary = dict(count=count, total=sum(c['n'] for c in count.values()), with_final=sum(c['final'] for c in count.values()),
                   slowdown_mean=float(np.mean(sl(rows))) if sl(rows) else None,
                   slowdown_range=[float(min(sl(rows))), float(max(sl(rows)))] if sl(rows) else None,
                   token_max_abs_delta={s: float(max(abs(t[s]['delta']) for t in tok)) for s in A.SPLITS} if tok else None,
                   avgpool={s: [round(t[s]['delta'], 2) for t in avg] for s in A.SPLITS},
                   avgpool_significant={s: sum(t[s]['holm'] < .05 for t in avg) for s in A.SPLITS})
    save(ROOT / 'report.json', dict(summary=summary, rows=rows, time=time.time()))
    print(json.dumps(summary, indent=1))
    for t in rows:
        print(f"{t['backbone']:9s} {t['budget']:8s} ms {t['ms'] or 0:.3f}->{t['ms_ref'] or 0:.3f} ({t['slowdown_pct'] or 0:+.2f}%)  "
              f"val {t['val']['delta']:+.2f} ({t['val']['changed']} images)  v2 {t['v2']['delta']:+.2f}")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=('plan', 'worker', 'time', 'report'))
    a = ap.parse_args()
    if a.stage == 'plan': plan()
    elif a.stage == 'worker': SD.worker()
    elif a.stage == 'time': SD.timing(A.ALL_BBS)
    else: report()


if __name__ == '__main__':
    main()
