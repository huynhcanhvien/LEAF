"""ViT-B/16 at 384 px (577 tokens): LEAF and the re-run training-free families on `augreg_b384`
(timm vit_base_patch16_384.augreg_in21k_ft_in1k), ImageNet val and V2, batch 32, with the protocol of clip_comparison.py
(every arm sealed before evaluation, BF16 batch 32, one exclusive timing session with three interleaved rounds, latency
pairing with tau = max(0.05 ms, 2 SD), McNemar + Holm). Launched 2026-10-25 for the KBS supplement.

Differences from the 224-px campaigns: the backbone is registered through LEAF_BACKBONES (backbones_vitb384.json), the
image caches live in data/cache384 (Resize 384 -> CenterCrop 384, timm crop 1.0), the probes use the same fractions
of the patch tokens as the paper's grid (cli.default_deltas), the token-count grid has step 8, budgets are
0.40-0.80 x the measured dense batch-32 latency, and the baseline grids are the main campaign's grids with their
per-layer counts scaled by 576/196 (ratio and keep-rate grids unchanged). DiffRate has no released 384-px schedules
and is not run; MCTF is timed at three points.

Stages (python leaf/reproduction/experiments/vitb384_comparison.py <stage>; results/vitb384_comparison/augreg_b384/):
  search  (GPUs)  `leaf search`: head groups, 1,917 probes on the 5,000 even calibration images (sharded over --gpus),
                  dense budgets, SALT workloads + timing + fit, the latency-budgeted DP -> search/schedules.json
  plan    (CPU)   LEAF schedules, dense and the baseline grids, sealed -> plans.json
  worker  (GPU)   val and v2 of every arm (one process per card, claims)
  time    (GPU0)  bs8/bs32 latency of every arm, three interleaved rounds, nothing else on the card
  report  (CPU)   latency pairs -> report.json, table.tex, frontier.pdf
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
REGISTRY = Path(__file__).resolve().parent / 'backbones_vitb384.json'
os.environ['LEAF_BACKBONES'] = ':'.join([p for p in os.environ.get('LEAF_BACKBONES', '').split(':') if p and p != str(REGISTRY)] + [str(REGISTRY)])
sys.path.insert(0, str(REPO))

from leaf.reproduction.experiments import clip_comparison as C

BB = 'augreg_b384'
ROOT = REPO / 'results/vitb384_comparison'
DATA_CACHE = REPO / 'data/cache384'
DENSE_FRACTIONS = (.40, .48, .56, .64, .72, .80)
SCALE = 576 / 196


def scaled(values): return tuple(sorted({int(round(v * SCALE)) for v in values}))


GRIDS = dict(tome=dict(r=scaled(C.GRIDS['tome']['r'])), tome_m3=dict(r=scaled(C.GRIDS['tome_m3']['r'])),
             tofu=dict(r=scaled(C.GRIDS['tofu']['r'])), rapid=dict(r=scaled(C.GRIDS['rapid']['r'])),
             pitome=dict(ratio=C.GRIDS['pitome']['ratio']), evit=dict(keep_rate=C.GRIDS['evit']['keep_rate']),
             ppt=dict(r=scaled(C.GRIDS['ppt']['r'])), mctf=dict(r=scaled((4, 8, 16))), onemore=dict(n=scaled(C.GRIDS['onemore']['n'])))

# The CLIP driver's protocol, re-targeted: one backbone, classification only, the 384-px caches. DiffRate has no released
# 384-px schedules; as on CLIP, the DeiT-B rates are transferred (same kept fraction of patch tokens per block,
# diffrate_key in backbones_vitb384.json) and sealed separately in plans_dr_transfer.json (plan-dr-transfer stage, added after
# plans.json was sealed), together with GMAC-planned LEAF at the GMAC of each transferred schedule.
C.ROOT, C.BBS, C.RETRIEVAL, C.GRIDS = ROOT, (BB,), (), GRIDS
C.DR_FAMILIES = {BB: 'DiffRate (DeiT-B rates)'}
C.EXTRA_PLANS = C.EXTRA_PLANS + ('plans_dr_transfer.json',)
C.gmac_schedules = lambda bb: {}                                  # plans.json (sealed 08:12) has no GMAC-planned arms


DR_SEARCH = REPO / 'results/diffrate_vitb384_search/rates.json'
DR_SEARCH_FAMILY = 'DiffRate (searched on ViT-B/384)'
C.DR_SEARCH_FAMILY = DR_SEARCH_FAMILY


def plan_dr_search():
    """DiffRate schedules searched on this backbone (diffrate_vitb384_search.py) and GMAC-planned LEAF at their GMAC,
    sealed in plans_dr_search.json before any evaluation of them (the CLIP driver's plan_dr_search, re-targeted)."""
    import json
    import time
    from leaf.backbones import SPECS
    from leaf.planner.configs import diffrate_config_from
    from leaf.planner.gmac import config_cost
    from leaf.planner.alpha import config_gmac
    from leaf.reproduction.experiments.gmac_planning import sha
    dest = ROOT / BB / 'plans_dr_search.json'
    if dest.exists(): return C.load(dest)
    spec, ws = SPECS[BB], C.workspace(BB)
    doc = C.load(DR_SEARCH); rates = doc['rates']['ViT-B384-searched']
    if not rates: raise SystemExit('no searched rates yet')
    jobs = {}
    for b, r in rates.items():
        cfg = diffrate_config_from(spec, json.loads(r['prune_kept_num']), json.loads(r['merge_kept_num']))
        jobs[f'ref/diffrate_search/{b}'] = dict(config=cfg, group='baseline', source=doc['source'], gmac=config_cost(cfg, spec), gflops_diffrate=r['gflops'])
    gm = ws / 'schedules_gmac_dr_search.json'
    if not gm.exists():
        subprocess.run([sys.executable, '-m', 'leaf', 'plan', '--backbone', BB, '--probes', str(ws / 'probes' / 'probes.json'),
                        '--groups', str(ws / 'groups.json'), '--gmac', *[f"{row['gmac']:.4f}" for row in jobs.values()], '--out', str(gm)], check=True, cwd=REPO)
    for name, row in C.load(gm)['schedules'].items():
        if 'config' in row:
            jobs[f'leaf/{name}'] = dict(config=row['config'], group='leaf_gmac', budget=row['budget'], planner=row['planner'],
                                        gmac=config_gmac(row['config'], spec) if row['config']['kind'] == 'pda' else config_cost(row['config'], spec))
    for row in jobs.values(): row['config_sha'] = C.key(row['config'])
    out = dict(backbone=BB, jobs=jobs, sha=sha(jobs), time=time.time(), rates=str(DR_SEARCH.relative_to(REPO)),
               rules=dict(diffrate=DR_SEARCH_FAMILY, note='sealed after plans.json, before any evaluation of these arms'))
    C.immutable(dest, out)
    return out


def plan_dr_transfer():
    """Transferred DiffRate schedules and GMAC-planned LEAF at their GMAC, sealed before any evaluation of them."""
    import json
    import time
    from leaf.backbones import SPECS
    from leaf.planner.gmac import config_cost
    from leaf.planner.alpha import config_gmac
    from leaf.reproduction.experiments.gmac_planning import sha
    dest = ROOT / BB / 'plans_dr_transfer.json'
    if dest.exists(): return C.load(dest)
    spec, ws = SPECS[BB], C.workspace(BB)
    points = C.diffrate_points(BB)
    if not points: raise SystemExit('no DiffRate rates to transfer (diffrate_key missing)')
    jobs = {f'ref/diffrate_transfer/{b}': dict(config=cfg, group='baseline', source=f'DiffRate rates of {spec.diffrate_key}, scaled to {spec.n_tokens} tokens', gmac=g)
            for b, (cfg, g) in points.items()}
    gm = ws / 'schedules_gmac_dr_transfer.json'
    if not gm.exists():
        subprocess.run([sys.executable, '-m', 'leaf', 'plan', '--backbone', BB, '--probes', str(ws / 'probes' / 'probes.json'),
                        '--groups', str(ws / 'groups.json'), '--gmac', *[f'{g:.4f}' for _, g in points.values()], '--out', str(gm)], check=True, cwd=REPO)
    for name, row in C.load(gm)['schedules'].items():
        if 'config' in row:
            jobs[f'leaf/{name}'] = dict(config=row['config'], group='leaf_gmac', budget=row['budget'], planner=row['planner'],
                                        gmac=config_gmac(row['config'], spec) if row['config']['kind'] == 'pda' else config_cost(row['config'], spec))
    for row in jobs.values(): row['config_sha'] = C.key(row['config'])
    out = dict(backbone=BB, jobs=jobs, sha=sha(jobs), time=time.time(), rules=dict(diffrate=C.DR_FAMILIES[BB], note='sealed after plans.json, before any evaluation of these arms'))
    C.immutable(dest, out)
    return out


def runner(bb):
    from leaf.api import LeafRunner
    return LeafRunner(bb, root=ROOT / bb, batch=32, data_cache=str(DATA_CACHE))


C.runner = runner


def search(gpus):
    cmd = [sys.executable, '-m', 'leaf', 'search', '--backbone', BB, '--workspace', str(C.workspace(BB)), '--gpus', gpus,
           '--anchors', 'dense', '--dense-fractions', *[f'{f:g}' for f in DENSE_FRACTIONS], '--data-cache', str(DATA_CACHE)]
    print(' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=REPO)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('stage', choices=['search', 'plan', 'worker', 'time', 'report', 'densify', 'retime', 'plan-dr-transfer', 'plan-dr-search'])
    ap.add_argument('--gpus', default='0,1')
    a = ap.parse_args(argv)
    if a.stage == 'search': search(a.gpus)
    elif a.stage == 'plan':
        p = C.plan(BB)
        print('PLANNED', BB, len(p['jobs']), 'arms', sum(k.startswith('leaf/') for k in p['jobs']), 'LEAF', 'sha', p['sha'][:12], flush=True)
    elif a.stage == 'worker': C.worker((BB,))
    elif a.stage == 'time': C.timing((BB,))
    elif a.stage == 'densify':      # cost-only extra baseline points near every LEAF latency (clip_comparison.densify), sealed before accuracy
        p = C.densify(BB); print('DENSIFIED', BB, len(p['jobs']), 'points', 'sha', p['sha'][:12], flush=True)
    elif a.stage == 'retime': C.retime((BB,))
    elif a.stage == 'plan-dr-transfer':
        p = plan_dr_transfer(); print('PLANNED DiffRate transfer arms', len(p['jobs']), 'sha', p['sha'][:12], flush=True)
    elif a.stage == 'plan-dr-search':
        p = plan_dr_search(); print('PLANNED DiffRate search arms', len(p['jobs']), 'sha', p['sha'][:12], flush=True)
    else: C.report((BB,))


if __name__ == '__main__':
    main()
