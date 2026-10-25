"""SALT below the range of its timing data: does covering the low budgets with a few more timed workloads remove the
under-prediction reported for the low-budget campaign (results/low_budget_and_factorial: 21 schedules at 0.7-0.9 of the
fastest timed schedule, latency under-predicted by 3.4 % on average, 6 of 21 within budget)?

Stages (python leaf/reproduction/experiments/salt_low_coverage.py <stage>; results in ROOT, nothing overwritten):
  make        (CPU)  GMAC-planned workloads at 0.20, 0.25 and 0.30 x dense GMAC (M3, no-forecast and prune-only variants,
                     leaf.latency.workloads.make_workloads) plus the five lowest-GMAC workloads of the frozen fit as
                     session anchors -> <bb>/workloads.json
  time        (GPU)  bs32 latency of the workloads, one GPU, both GPUs otherwise idle, three interleaved rounds
                     -> <bb>/timings.json
  fit         (CPU)  station_rich SALT refitted on the frozen workloads plus the new ones (same procedure as
                     leaf fit-latency) -> <bb>/salt_ext.json; the frozen workloads alone -> <bb>/salt_refit_check.json
  report      (CPU)  prediction error of the frozen and the extended model on the 21 low-budget schedules against their
                     measured latency -> report.json
  replan      (CPU)  the 21 low budgets re-planned with the extended model -> <bb>/plans_ext.json
  time_replan (GPU)  latency of the re-planned and the original low-budget schedules in one session -> replan_latency.json,
                     and the budget-met counts -> replan_report.json
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
import leaf.reproduction.experiments.main_comparison as main
from leaf.reproduction.experiments.low_budget_and_factorial import probe_curves, ROOT as LOW

ROOT = REPO / 'results/salt_low_coverage'
PLANNING = REPO / 'schedules/leaf/planning'
BBS = A.ALL_BBS
FRACTIONS = (.2, .25, .3)
N_ANCHORS = 5


def load(p): return json.loads(Path(p).read_text())


def save(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(obj, indent=1, sort_keys=True, default=float))


def frozen_rows(bb):
    return load(PLANNING / bb / 'salt_workloads.json')['rows']


def kv(cfg):
    from leaf.planner.alpha import kappa_view
    return kappa_view(cfg)


# ----------------------------------------------------------------------------------------------- make (CPU)
def make(bbs):
    from leaf.evaluation.runner import config_hash
    from leaf.latency.workloads import make_workloads, VARIANTS
    from leaf.planner.gmac import config_cost
    for bb in bbs:
        dest = ROOT / bb / 'workloads.json'
        if dest.exists(): print('exists', dest); continue
        spec, d = A.spec_of(bb), main.dcfg(bb)
        curves = A.Ctx(bb).base['M3']                 # alpha-keyed curves: make_workloads relabels the planner's keys as alpha
        variants = [v for v in VARIANTS if v[0] in ('m3', 'no_forecast', 'prune_only')]
        rows = make_workloads(spec, d, curves, fractions=FRACTIONS, variants=variants, diffrate=False)
        old = frozen_rows(bb); old_hashes = {config_hash(kv(r['config'])) for r in old}
        new = [dict(r, gmac=config_cost(kv(r['config']), spec), role='new') for r in rows if config_hash(kv(r['config'])) not in old_hashes]
        anchors = sorted(old, key=lambda r: config_cost(kv(r['config']), spec))[:N_ANCHORS]
        anchors = [dict(name=f"anchor/{r.get('names', ['?'])[0]}", config=r['config'], gmac=config_cost(kv(r['config']), spec), role='anchor',
                        ms_frozen_session=r['ms']) for r in anchors]
        out = new + anchors
        save(dest, dict(backbone=bb, fractions=list(FRACTIONS), rows=out, time=time.time()))
        print(bb, len(new), 'new workloads', sorted(round(r['gmac'], 2) for r in new), '+', len(anchors), 'anchors', flush=True)


# ----------------------------------------------------------------------------------------------- time (GPU)
def runner(bb):
    from leaf.reproduction.runners.alpha_runner import alpha_runner
    return alpha_runner(bb, ROOT)


def time_rows(bbs, key, rows_of, rounds=3):
    """Time rows_of(bb) -> ROOT/<bb>/<key>.json; one session per backbone; requires both GPUs idle."""
    import os
    import torch
    from leaf.reproduction.runners.exclusive_timing import cast_exclusive
    os.environ['LEAF_EXCLUSIVE_TIMING'] = '1'
    for bb in bbs:
        dest = ROOT / bb / f'{key}.json'
        if dest.exists(): print('exists', dest); continue
        rows = rows_of(bb)
        if not rows: continue
        while not SB.gpus_idle(): time.sleep(60)
        r = runner(bb); raw = ROOT / bb / f'{key}_rounds'; times = {i: [] for i in range(len(rows))}
        for rnd in range(rounds):
            order = list(range(len(rows))) if rnd % 2 == 0 else list(range(len(rows)))[::-1]
            for i in order:
                if not SB.gpus_idle(): raise SystemExit('another GPU process appeared during timing; session discarded')
                p = raw / f'{i:03d}#{rnd}.json'
                if p.exists(): times[i].append(load(p)['ms']); continue
                m = r.inference_model(rows[i]['config'])
                meas = cast_exclusive(m, 32, iters=50, warmup=10)
                save(p, dict(index=i, name=rows[i].get('name'), round=rnd, ms=meas['ms_mean'], measurement=meas,
                             device=torch.cuda.get_device_name(0), time=time.time()))
                times[i].append(meas['ms_mean']); m = None; gc.collect(); torch.cuda.empty_cache()
        out = [dict(rows[i], ms=float(np.mean(times[i])), sd=float(np.std(times[i], ddof=1)), rounds=times[i]) for i in range(len(rows))]
        save(dest, dict(backbone=bb, batch=32, rounds=rounds, device=torch.cuda.get_device_name(0), time=time.time(), rows=out))
        del r; gc.collect(); torch.cuda.empty_cache()
        print(bb, 'timed', len(out), 'rows', flush=True)


def timing(bbs): time_rows(bbs, 'timings', lambda bb: load(ROOT / bb / 'workloads.json')['rows'])


# ----------------------------------------------------------------------------------------------- fit (CPU)
def fit_rows(bb, rows):
    from leaf.evaluation.runner import config_hash
    from leaf.latency.salt import features, fit_salt, FORMULAS
    spec = A.spec_of(bb); data = []
    for row in rows:
        cfg = kv(row['config'])
        group = row.get('group') or config_hash({k: v for k, v in cfg.items() if k not in ('kind', 'merge')})
        data.append(dict(config=cfg, ms=row['ms'], group=group, features=features(cfg, spec)))
    model = fit_salt(data, 'station_rich')
    model.update(backbone=bb, n_rows=len(data),
                 training_range={k: [min(r['features'].get(k, 0) for r in data), max(r['features'].get(k, 0) for r in data)] for k in FORMULAS['station_rich']})
    return model


def fit(bbs):
    for bb in bbs:
        dest = ROOT / bb / 'salt_ext.json'
        if dest.exists(): print('exists', dest); continue
        old = frozen_rows(bb); new = [r for r in load(ROOT / bb / 'timings.json')['rows'] if r['role'] == 'new']
        anchors = [r for r in load(ROOT / bb / 'timings.json')['rows'] if r['role'] == 'anchor']
        drift = [100 * (r['ms'] / r['ms_frozen_session'] - 1) for r in anchors]
        check = fit_rows(bb, old); ext = fit_rows(bb, old + new)
        frozen = main.design()['models'][bb]['main']
        save(ROOT / bb / 'salt_refit_check.json', dict(check, frozen_coef=frozen['coef'], frozen_intercept=frozen['intercept'],
                                                      max_abs_coef_diff=float(np.max(np.abs(np.array(check['coef']) - np.array(frozen['coef']))))))
        save(dest, dict(ext, anchor_drift_pct=drift, n_new=len(new), n_old=len(old)))
        print(bb, 'refit check max|dcoef|', round(float(np.max(np.abs(np.array(check['coef']) - np.array(frozen['coef'])))), 5),
              'anchor drift %', [round(v, 2) for v in drift], 'ext n', len(old) + len(new), flush=True)


# ----------------------------------------------------------------------------------------------- report (CPU)
def low_schedules(bb):
    """The 21 low-budget schedules: config, budget, frozen prediction and measured latency (low-budget session)."""
    jobs = load(LOW / bb / 'plans.json')['jobs']; out = []
    for name, job in jobs.items():
        if not name.startswith('full/latency/') or 'config' not in job: continue
        ms = [load(p)['measurement']['ms_mean'] for p in sorted((LOW / bb / 'latency').glob(f"{bb}-{job['config_sha']}#*.json"))
              if 'ms_mean' in load(p)['measurement']]
        out.append(dict(name=name, config=job['config'], budget_ms=job['budget']['latency_ms'], predicted_frozen=job['predicted_ms'],
                        measured_ms=float(np.mean(ms)) if ms else None, n_rounds=len(ms)))
    return out


def predictor(bb, model):
    from leaf.planner.plan import OperatorCost
    return OperatorCost(model, A.spec_of(bb), main.dcfg(bb), 'M3')


def report(bbs):
    out, rows = {}, []
    for bb in bbs:
        p = ROOT / bb / 'salt_ext.json'
        if not p.exists(): continue
        ext = predictor(bb, load(p))
        for s in low_schedules(bb):
            if s['measured_ms'] is None: continue
            pe = ext(kv(s['config']))
            rows.append(dict(backbone=bb, name=s['name'], budget_ms=s['budget_ms'], measured_ms=s['measured_ms'],
                             frozen=s['predicted_frozen'], ext=pe,
                             err_frozen_pct=100 * (s['predicted_frozen'] / s['measured_ms'] - 1), err_ext_pct=100 * (pe / s['measured_ms'] - 1),
                             met=s['measured_ms'] <= s['budget_ms']))
    for key in ('frozen', 'ext'):
        e = np.array([r[f'err_{key}_pct'] for r in rows])
        out[key] = dict(mean_signed_pct=float(e.mean()), mape_pct=float(np.abs(e).mean()), max_abs_pct=float(np.abs(e).max()), n=len(rows))
    out['budget_met_frozen_plans'] = int(sum(r['met'] for r in rows))
    save(ROOT / 'report.json', dict(summary=out, rows=rows, time=time.time()))
    print(json.dumps(out, indent=1))
    for r in rows: print(f"{r['backbone']:9s} {r['name']:20s} measured {r['measured_ms']:7.2f} frozen {r['err_frozen_pct']:+6.2f}%  ext {r['err_ext_pct']:+6.2f}%")


# ----------------------------------------------------------------------------------------------- replan (CPU) and its timing
def replan(bbs):
    from leaf.evaluation.runner import config_hash
    from leaf.planner.plan import plan_latency
    for bb in bbs:
        dest = ROOT / bb / 'plans_ext.json'
        if dest.exists(): print('exists', dest); continue
        p = ROOT / bb / 'salt_ext.json'
        if not p.exists(): continue
        spec, d = A.spec_of(bb), main.dcfg(bb); curves = probe_curves(bb, d, 'M3'); model = load(p)
        jobs = {}
        for s in low_schedules(bb):
            res = plan_latency(curves, spec, d, model, s['budget_ms'])
            name = s['name'].replace('full/latency/', 'ext/')
            jobs[name] = dict(res, budget_ms=s['budget_ms'], original=s['name'], original_config_sha=config_hash(s['config']))
            if 'config' in res: jobs[name]['config_sha'] = config_hash(res['config'])
            jobs[s['name']] = dict(config=s['config'], config_sha=config_hash(s['config']), budget_ms=s['budget_ms'], group='original',
                                   predicted_frozen=s['predicted_frozen'], measured_ms_low_session=s['measured_ms'])
        save(dest, dict(backbone=bb, jobs=jobs, time=time.time()))
        print(bb, 'replanned', sum(1 for n, j in jobs.items() if n.startswith('ext/') and 'config' in j), 'of', sum(1 for n in jobs if n.startswith('ext/')), flush=True)


def replan_rows(bb):
    p = ROOT / bb / 'plans_ext.json'
    if not p.exists(): return []
    jobs = load(p)['jobs']
    return [dict(name=n, config=j['config'], budget_ms=j['budget_ms'], predicted_ms=j.get('predicted_ms', j.get('predicted_frozen')))
            for n, j in jobs.items() if 'config' in j] + [dict(name='dense', config=dict(kind='dense'), budget_ms=None, predicted_ms=None)]


def time_replan(bbs):
    time_rows(bbs, 'replan_latency', replan_rows)
    rows = []
    for bb in bbs:
        p = ROOT / bb / 'replan_latency.json'
        if not p.exists(): continue
        for r in load(p)['rows']:
            if r['name'] == 'dense': continue
            rows.append(dict(backbone=bb, name=r['name'], budget_ms=r['budget_ms'], predicted_ms=r['predicted_ms'], ms=r['ms'],
                             met=r['ms'] <= r['budget_ms'], err_pct=None if not r['predicted_ms'] else 100 * (r['predicted_ms'] / r['ms'] - 1)))
    summary = {}
    for key, pref in (('ext', 'ext/'), ('original', 'full/latency/')):
        sel = [r for r in rows if r['name'].startswith(pref)]
        if sel:
            e = np.array([r['err_pct'] for r in sel if r['err_pct'] is not None])
            summary[key] = dict(n=len(sel), budget_met=int(sum(r['met'] for r in sel)), mean_signed_err_pct=float(e.mean()) if len(e) else None,
                                mape_pct=float(np.abs(e).mean()) if len(e) else None)
    save(ROOT / 'replan_report.json', dict(summary=summary, rows=rows, time=time.time()))
    print(json.dumps(summary, indent=1))


# ----------------------------------------------------------------------------------------------- accuracy of the re-planned schedules
def eval_replan(bbs):
    """ImageNet val and V2 of the re-planned schedules (the originals are reused from the low-budget campaign)."""
    import torch
    from leaf.reproduction.runners.alpha_runner import alpha_runner
    from leaf.reproduction.runners.exclusive_timing import guard_foreign
    guard_foreign()
    for bb in bbs:
        p = ROOT / bb / 'plans_ext.json'
        if not p.exists(): continue
        r = alpha_runner(bb, ROOT, reuse_roots=[LOW / bb])
        for name, job in load(p)['jobs'].items():
            if 'config' not in job: continue
            for split in ('val', 'v2'):
                arr = r.run(job['config'], split)
                print(bb, name, split, None if arr is None else round(100 * float(arr['correct'].mean()), 2), flush=True)
        del r; gc.collect(); torch.cuda.empty_cache()


def report_replan_accuracy(bbs):
    from leaf.evaluation.data import mcnemar, holm
    from leaf.evaluation.runner import config_hash
    def correct(bb, cfg, split):
        for base in (ROOT / bb, LOW / bb):
            q = base / split / 'runs' / f'{config_hash(cfg)}.npz'
            if q.exists(): return np.load(q)['correct'].astype(bool)
        return None
    rows = []
    for bb in bbs:
        p = ROOT / bb / 'plans_ext.json'
        if not p.exists(): continue
        jobs = load(p)['jobs']
        for name, job in jobs.items():
            if not name.startswith('ext/') or 'config' not in job: continue
            ref = jobs[job['original']]
            t = dict(backbone=bb, budget=name.split('/')[-1], same=config_hash(job['config']) == config_hash(ref['config']))
            for split in ('val', 'v2'):
                a, b = correct(bb, job['config'], split), correct(bb, ref['config'], split)
                if a is None or b is None: t[split] = None; continue
                pv, _, _ = mcnemar(a, b)
                t[split] = dict(delta=float(100 * (a.mean() - b.mean())), p=pv, top1=float(100 * a.mean()), top1_ref=float(100 * b.mean()))
            rows.append(t)
    for split in ('val', 'v2'): holm([t[split] for t in rows if t[split] is not None and not t['same']])
    summary = {s: dict(mean_delta=float(np.mean([t[s]['delta'] for t in rows if t[s]])),
                       wins=sum(t[s] is not None and t[s].get('holm', 1) < .05 and t[s]['delta'] > 0 for t in rows),
                       losses=sum(t[s] is not None and t[s].get('holm', 1) < .05 and t[s]['delta'] < 0 for t in rows)) for s in ('val', 'v2')}
    save(ROOT / 'replan_accuracy.json', dict(summary=summary, rows=rows, time=time.time()))
    print(json.dumps(summary, indent=1))
    for t in rows:
        c = lambda s: 'n/a' if t[s] is None else f"{t[s]['delta']:+.2f}" + ('*' if t[s].get('holm', 1) < .05 else ' ')
        print(f"{t['backbone']:9s} {t['budget']:6s} val {c('val')} ({t['val']['top1_ref']:.2f}->{t['val']['top1']:.2f})  v2 {c('v2')}")


def main_cli():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=('make', 'time', 'fit', 'report', 'replan', 'time_replan', 'eval_replan', 'report_replan_accuracy'))
    ap.add_argument('--backbones', default=','.join(BBS))
    a = ap.parse_args(); bbs = a.backbones.split(',')
    {'make': make, 'time': timing, 'fit': fit, 'report': report, 'replan': replan, 'time_replan': time_replan, 'eval_replan': eval_replan, 'report_replan_accuracy': report_replan_accuracy}[a.stage](bbs)


if __name__ == '__main__':
    main_cli()
