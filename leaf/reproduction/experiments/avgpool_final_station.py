"""Final-block station on backbones with an average-pooled head (MAE ViT-B/L): the paper found that removing the
final-block station from the three GMAC-planned ViT-L schedules that have one raises validation accuracy by
0.30-0.34 points, because the separable surrogate does not see its damage there. This experiment forbids a station in
the last block at planning time (the probe curves of the last block are dropped, so the DP cannot place a station there;
leaf.reproduction.experiments.ablation_and_alpha_grid.drop_block) and re-plans the GMAC-planned and the latency-planned
schedules of MAE-B and MAE-L at the paper's budgets. The released schedules are kept and re-timed in the same session.

Stages (python leaf/reproduction/experiments/avgpool_final_station.py <stage>; results in ROOT, nothing overwritten):
  plan    (CPU)  re-planned schedules next to the released ones -> <bb>/plans.json
  worker  (GPU)  ImageNet val and V2 of the re-planned schedules (released ones reused from their campaigns)
  time    (GPU)  bs32 latency of re-planned, released and dense, one GPU, both GPUs otherwise idle -> latency.json
  report  (CPU)  delta top-1 (McNemar + Holm), GMAC and latency -> report.json
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
from leaf.reproduction.experiments import dp_verify as DV
import leaf.reproduction.experiments.main_comparison as main
from leaf.reproduction.experiments.low_budget_and_factorial import probe_curves

ROOT = REPO / 'results/avgpool_final_station'
BBS = ('mae_b', 'mae_l')


def load(p): return json.loads(Path(p).read_text())


def save(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(obj, indent=1, sort_keys=True, default=float))


def has_final_station(cfg, spec):
    n = spec.n_tokens
    for l in range(spec.depth - 1): n = min(n, cfg['prune_keep'][l], cfg['merge_keep'][l])
    return min(cfg['prune_keep'][-1], cfg['merge_keep'][-1]) < n


# ----------------------------------------------------------------------------------------------- plan (CPU)
def plan(bbs):
    from leaf.evaluation.runner import config_hash
    from leaf.planner.gmac import GmacPlanner, config_cost
    for bb in bbs:
        dest = ROOT / bb / 'plans.json'
        if dest.exists(): print('exists', dest); continue
        c = A.Ctx(bb); spec, d = c.spec, main.dcfg(bb); last = spec.depth - 1
        jobs = {'ref/dense': dict(config=dict(kind='dense'), group='reference')}
        # GMAC-planned: the paper's ref/M3/gmac/<bid> schedules of the main campaign (kappa grid = alpha4), planned at their GMAC
        mjobs = load(main.ROOT / bb / 'plans.json')['jobs']
        curves_k = probe_curves(bb, d, 'M3'); curves_k_nf = A.drop_block(curves_k, last)
        planner = GmacPlanner(spec, d, curves_k_nf, 'M3')
        for name, row in mjobs.items():
            if not name.startswith('ref/M3/gmac/') or 'config' not in row: continue
            bid = name.split('/')[-1]; cfg = row['config']
            jobs[f'ref/gmac/{bid}'] = dict(config=cfg, config_sha=config_hash(cfg), gmac=row['gmac'], group='released_gmac',
                                           final_station=has_final_station(cfg, spec))
            budget = row['gmac']                                     # same GMAC cap as the released schedule reached
            res = planner.plan(budget)
            if 'config' in res:
                jobs[f'nofinal/gmac/{bid}'] = dict(config=res['config'], config_sha=config_hash(res['config']), gmac=res['gmac'], fhat=res['fhat'],
                                                   state=res['state'], budget_gmac=budget, group='nofinal_gmac',
                                                   same_as_released=config_hash(res['config']) == config_hash(cfg))
            else: jobs[f'nofinal/gmac/{bid}'] = dict(res, budget_gmac=budget, group='nofinal_gmac')
        # latency-planned: the released leaf/alpha7 schedules, re-planned with the last block dropped
        cv = DV.alpha7_curves(bb, c); cv_nf = A.drop_block(cv, last); ref = DV.released(bb)
        for b in c.budgets:
            bid = b['id']
            if bid not in ref: continue
            r = ref[bid]
            jobs[f'ref/latency/{bid}'] = dict(config=r['config'], config_sha=config_hash(r['config']), gmac=r.get('gmac'), group='released_latency',
                                              final_station=has_final_station(r['config'], spec), budget=b)
            res = c.plan(cv_nf, b['latency_ms'])
            if 'config' in res:
                jobs[f'nofinal/latency/{bid}'] = dict(config=res['config'], config_sha=config_hash(res['config']), fhat=res['fhat'], state=res['state'],
                                                      predicted_ms=res['predicted_ms'], gmac=res.get('gmac'), budget=b, group='nofinal_latency',
                                                      same_as_released=config_hash(res['config']) == config_hash(r['config']))
            else: jobs[f'nofinal/latency/{bid}'] = dict(res, budget=b, group='nofinal_latency')
        save(dest, dict(backbone=bb, jobs=jobs, sha=A.sha(jobs), time=time.time()))
        g = [n for n in jobs if n.startswith('nofinal/gmac/') and 'config' in jobs[n]]
        print(bb, 'gmac re-planned', len(g), 'changed', sum(not jobs[n]['same_as_released'] for n in g),
              '| latency re-planned', sum(1 for n in jobs if n.startswith('nofinal/latency/') and 'config' in jobs[n]),
              'changed', sum(1 for n in jobs if n.startswith('nofinal/latency/') and 'config' in jobs[n] and not jobs[n]['same_as_released']),
              '| released with final station:', [n.split('/')[-1] for n in jobs if n.startswith('ref/') and jobs[n].get('final_station')], flush=True)


# ----------------------------------------------------------------------------------------------- worker (GPU)
def runner(bb):
    from leaf.reproduction.runners.alpha_runner import alpha_runner
    return alpha_runner(bb, ROOT, reuse_roots=[main.ROOT / bb, A.ROOT / bb, A.EQUIV / bb])


def worker(bbs):
    import torch
    from leaf.reproduction.runners.exclusive_timing import guard_foreign
    guard_foreign()
    for bb in bbs:
        jobs = load(ROOT / bb / 'plans.json')['jobs']; r = runner(bb)
        for name, job in jobs.items():
            if 'config' not in job or job['config']['kind'] == 'dense': continue
            if job['group'].startswith('nofinal') and job.get('same_as_released'): continue
            for split in A.SPLITS:
                arr = r.run(job['config'], split)
                print(bb, name, split, None if arr is None else round(100 * float(arr['correct'].mean()), 2), flush=True)
        del r; gc.collect(); torch.cuda.empty_cache()


# ----------------------------------------------------------------------------------------------- timing (GPU)
def timing(bbs, rounds=3):
    import os
    import torch
    from leaf.reproduction.runners.exclusive_timing import cast_exclusive
    os.environ['LEAF_EXCLUSIVE_TIMING'] = '1'
    if (ROOT / 'latency.json').exists(): print('exists'); return
    while not SB.gpus_idle(): time.sleep(60)
    jobs = []
    for bb in bbs:
        for name, job in load(ROOT / bb / 'plans.json')['jobs'].items():
            if 'config' in job and not (job['group'].startswith('nofinal') and job.get('same_as_released')): jobs.append((bb, f'{bb}/{name}', job['config']))
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
    for base in (ROOT / bb, main.ROOT / bb, A.ROOT / bb, A.EQUIV / bb):
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
            if not name.startswith('nofinal/') or 'config' not in job: continue
            axis, bid = name.split('/')[1:]; ref = jobs[f'ref/{axis}/{bid}']
            t = dict(backbone=bb, axis=axis, budget=bid, same=bool(job.get('same_as_released')), ref_final_station=ref['final_station'],
                     K=len(job['state']['sites']) if 'state' in job else None, gmac=job.get('gmac'), gmac_ref=ref.get('gmac'),
                     ms=lat.get(f'{bb}/{name}', {}).get('ms'), ms_ref=lat.get(f'{bb}/ref/{axis}/{bid}', {}).get('ms'), ms_dense=lat.get(f'{bb}/ref/dense', {}).get('ms'))
            for split in A.SPLITS:
                if t['same']: t[split] = dict(delta=0., p=1., holm=1.); continue
                a, b = correct(bb, job['config'], split), correct(bb, ref['config'], split)
                if a is None or b is None: t[split] = None; continue
                pv, _, _ = mcnemar(a, b)
                t[split] = dict(delta=float(100 * (a.mean() - b.mean())), p=pv, top1=float(100 * a.mean()), top1_ref=float(100 * b.mean()))
            rows.append(t)
    for split in A.SPLITS:
        holm([t[split] for t in rows if t[split] is not None and not t['same']])
    summary = {}
    for axis in ('gmac', 'latency'):
        sel = [t for t in rows if t['axis'] == axis]
        summary[axis] = dict(n=len(sel), changed=sum(not t['same'] for t in sel),
                             **{f'mean_delta_{s}': float(np.mean([t[s]['delta'] for t in sel if t[s]])) if sel else None for s in A.SPLITS},
                             **{f'win_{s}': sum(t[s] is not None and t[s].get('holm', 1) < .05 and t[s]['delta'] > 0 for t in sel) for s in A.SPLITS},
                             **{f'loss_{s}': sum(t[s] is not None and t[s].get('holm', 1) < .05 and t[s]['delta'] < 0 for t in sel) for s in A.SPLITS})
    save(ROOT / 'report.json', dict(summary=summary, rows=rows, time=time.time()))
    print(json.dumps(summary, indent=1))
    for t in rows:
        cell = lambda s: '  n/a ' if t[s] is None else f"{t[s]['delta']:+.2f}" + ('*' if t[s].get('holm', 1) < .05 else ' ')
        print(f"{t['backbone']:6s} {t['axis']:8s} {t['budget']:7s} {'same' if t['same'] else 'new '} refFinal={t['ref_final_station']} "
              f"gmac {t['gmac_ref'] or 0:.2f}->{t['gmac'] or 0:.2f} ms {t['ms_ref'] or 0:.2f}->{t['ms'] or 0:.2f} val {cell('val')} v2 {cell('v2')}")


def main_cli():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=('plan', 'worker', 'time', 'report'))
    ap.add_argument('--backbones', default=','.join(BBS))
    a = ap.parse_args(); bbs = a.backbones.split(',')
    {'plan': plan, 'worker': worker, 'time': timing, 'report': report}[a.stage](bbs)


if __name__ == '__main__':
    main_cli()
