"""Export the paper-v3 schedules of LEAF and the inputs needed to re-plan them (CPU only, reads sealed campaigns).

    python experiments/export_leaf_schedules.py [--backbones deit_b ...]

Writes, per backbone,
  schedules/leaf/<bb>.json                  released schedules with measured accuracy and latency:
      latency/<budget>   the main method (alpha grid {0, 1/4, 1/3, 1/2, 2/3, 3/4, 1}, station-rich SALT budget),
                         results/ablation_and_alpha_grid/<bb>/plans/leaf.json; val / V2 / latency as in the
                         paper (results/main_comparison_analysis/report_leaf.json, latency transferred to the
                         main session by leaf/reproduction/experiments/main_comparison_analysis.py)
      gmac/<budget>      planned under a GMAC budget (alpha grid {0, 1/2, 2/3, 3/4}; budget = DiffRate's actual matrix
                         MACs, results/gmac_planning), the 'ref/M3/gmac' arms of
                         results/main_comparison written in alpha form (same token order; the
                         measured values are those of the evaluated kappa config, recorded as evaluated_config_sha)
  schedules/leaf/planning/<bb>/             what `python -m leaf plan` needs to re-plan any budget:
      probes.json        single-station probe means on the 5,000 calibration images, in planning order
                         (M3 probes relabelled kappa -> alpha, then alpha = 1, then alpha in {1/4, 1/3})
      groups.json        head groups of the forecast D (1,024 calibration images)
      salt.json          the frozen station-rich SALT model (bs32, RTX 5060 Ti, BF16, CUDA graph)
      salt_workloads.json  the timed workloads the SALT model was fitted on (config, ms, sd, CV group)
      budgets.json       the latency budgets of the paper
"""
import argparse
import json
from pathlib import Path
import shutil
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
import numpy as np

from leaf.backbones import SPECS
from leaf.model.alpha_score import config_to_alpha, kappa_to_alpha
from leaf.api import stations
from leaf.planner.alpha import config_gmac

MAIN_COMPARISON = REPO / 'results/main_comparison'
LOW = REPO / 'results/low_budget_and_factorial'
DP = REPO / 'results/gmac_planning'
ABL = REPO / 'results/ablation_and_alpha_grid'
LEAF_REPORT = REPO / 'results/main_comparison_analysis/report_leaf.json'
SALT = REPO / 'results/salt_fit'
OUT = REPO / 'schedules/leaf'
BBS = ('deit_t', 'deit_s', 'deit_b', 'augreg_s', 'augreg_b', 'mae_b', 'mae_l')
ALPHA_GRID = (0., .25, 1 / 3, .5, 2 / 3, .75, 1.)


def load(p): return json.loads(Path(p).read_text())


def save(p, obj):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, default=float) + '\n')


def round_means(folder, bs=32):
    rows = {}
    for f in Path(folder).glob('*.json'):
        d = load(f); m = d['measurements'].get(str(bs)) if isinstance(d.get('measurements'), dict) else None
        if isinstance(m, dict) and 'ms_mean' in m: rows.setdefault(d['id'], []).append(m['ms_mean'])
    return {k: (float(np.mean(v)), float(np.std(v, ddof=1))) for k, v in rows.items() if len(v) == 3}


def top1(root, sha, split):
    p = Path(root) / split / 'runs' / f'{sha}.npz'
    return round(100 * float(np.load(p)['correct'].mean()), 3) if p.exists() else None


def export(bb):
    design = load(MAIN_COMPARISON / 'design.json'); binding = design['backbones'][bb]
    spec = SPECS[bb]
    ms = round_means(MAIN_COMPARISON / bb / 'latency')
    jobs = load(MAIN_COMPARISON / bb / 'plans.json')['jobs']
    r1_rows = {a['name']: a for a in load(LEAF_REPORT)['arms'][bb] if a['name'].startswith('leaf/')}
    r1_plans = load(ABL / bb / 'plans' / 'leaf.json')['jobs']
    gmac_budget = {n.split('/')[-1]: j['meta']['budget'] for n, j in load(DP / bb / 'frozen.json')['jobs'].items()
                   if n.startswith('M3/gmac/')}    # the budgets the GMAC planner was run with

    dense = jobs['ref/dense']; dsha = dense['config_sha']
    doc = dict(
        backbone=bb, timm_id=binding['timm_id'],
        method='LEAF: stations scored by (1 - alpha) z(log P) + alpha z(D), prune -> M3 merge, '
               'planned by the Lagrangian DP under the station-rich SALT latency estimate',
        forecast=dict(G=16, rank=32), operator='M3', alpha_grid=list(ALPHA_GRID), theta_grid=[0., .5, 1.],
        hardware='NVIDIA RTX 5060 Ti, BF16, CUDA graph, batch 32; latency = mean of three interleaved rounds',
        dense=dict(val=top1(MAIN_COMPARISON / bb, dsha, 'val'), v2=top1(MAIN_COMPARISON / bb, dsha, 'v2'), ms_bs32=round(ms[f'{bb}-{dsha}'][0], 3)),
        sources=dict(latency=str(LEAF_REPORT.relative_to(REPO)), latency_plans=str((ABL / bb / 'plans' / 'leaf.json').relative_to(REPO)),
                     gmac=str((MAIN_COMPARISON / bb / 'plans.json').relative_to(REPO))),
        schedules={})
    for b in binding['budgets']:
        bid = b['id']
        plan = r1_plans.get(f'leaf/alpha7/{bid}'); row = r1_rows.get(f'leaf/{bid}')
        if plan and 'config' in plan and row:
            cfg = plan['config']
            doc['schedules'][f'latency/{bid}'] = dict(
                budget=dict(kind='latency_ms_bs32', value=b['latency_ms'], anchor=b.get('anchor'), id=bid),
                stations=stations(cfg, spec), config=cfg, config_sha=plan['config_sha'],
                planner=dict(fhat=plan['fhat'], predicted_ms=plan['predicted_ms'], gmac=config_gmac(cfg, spec)),
                measured=dict(val=round(row['val'], 3), v2=round(row['v2'], 3), ms_bs32=round(row['ms'], 3),
                              sd=round(row['sd'], 3), latency_transfer=row.get('transfer'), stations=row.get('K')))
        g = jobs.get(f'ref/M3/gmac/{bid}')
        if g and 'config' in g:
            cfg = config_to_alpha(g['config']); sha = g['config_sha']
            m = ms.get(f'{bb}-{sha}')
            doc['schedules'][f'gmac/{bid}'] = dict(
                budget=dict(kind='gmac', value=gmac_budget[bid]['gmac'], id=bid, rule=gmac_budget[bid]['source']),
                stations=stations(cfg, spec), config=cfg, evaluated_config_sha=sha,
                planner=dict(gmac=config_gmac(cfg, spec)),
                measured=dict(val=top1(MAIN_COMPARISON / bb, sha, 'val'), v2=top1(MAIN_COMPARISON / bb, sha, 'v2'),
                              ms_bs32=None if m is None else round(m[0], 3), sd=None if m is None else round(m[1], 3)))
    save(OUT / f'{bb}.json', doc)

    # planning inputs
    plan_dir = OUT / 'planning' / bb
    plan_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(binding['root']) / 'groups.json', plan_dir / 'groups.json')
    records = [dict(block=r['block'], count=r['count'], theta=r['theta'], alpha=kappa_to_alpha(r['kappa']), kl=r['kl'], set='M3')
               for r in load(LOW / bb / 'probe_means_M3.json')]
    for name in ('alpha_ext', 'alpha_fine'):
        records += [dict(block=r['block'], count=r['count'], theta=r['theta'], alpha=r['alpha'], kl=r['kl'], set=name)
                    for r in load(ABL / bb / 'probes' / f'{name}.json')]
    save(plan_dir / 'probes.json', dict(backbone=bb, images='calib_bal even indices (5,000 training images)',
                                        batch=32, operator='M3 (pure pruning: plain operator)', forecast=dict(G=16, rank=32),
                                        records=records))
    model = design['models'][bb]['main']
    save(plan_dir / 'salt.json', dict(model, backbone=bb, timm_id=binding['timm_id'], batch=32,
                                      device='NVIDIA RTX 5060 Ti, BF16, CUDA graph',
                                      source=str(Path(design['models'][bb]['source']).relative_to(REPO))
                                      if str(design['models'][bb]['source']).startswith(str(REPO)) else design['models'][bb]['source']))
    data = load(SALT / 'dataset.json')
    amended = {}
    am = SALT / 'workload_cv' / 'group_amendment.json'
    if (SALT / 'workload_cv' / bb).exists() and am.exists():
        amended = {r['key']: r['group'] for r in load(am)['rows'] if r['backbone'] == bb}
    rows = [dict(config=r['config'], ms=r['ms'], sd=r['sd'], group=amended.get(r['key'], r['group']), names=r['names'])
            for r in data['rows'] if r['backbone'] == bb]
    save(plan_dir / 'salt_workloads.json', dict(backbone=bb, batch=32, rounds=3, device='NVIDIA RTX 5060 Ti, BF16, CUDA graph',
                                                note='ms = mean of three interleaved rounds; group = CV group (identical workload)',
                                                rows=rows))
    save(plan_dir / 'budgets.json', binding['budgets'])
    return len(doc['schedules'])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--backbones', nargs='+', default=list(BBS), choices=BBS)
    args = ap.parse_args()
    for bb in args.backbones:
        print(bb, export(bb), 'schedules', flush=True)


if __name__ == '__main__':
    main()
