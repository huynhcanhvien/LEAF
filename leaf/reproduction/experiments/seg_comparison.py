"""Segmentation pilot: LEAF on Segmenter ViT-B/16 (ADE20K, 512x512 windows), against ToMe with unmerging.

The planner is unchanged; only its objective changes: the probes measure the per-pixel KL of the decoder's class
distribution to the dense model on 2,000 unlabelled training windows (leaf/evaluation/segmentation.py). Tokens leave
the network as in leaf/model/dense.py: a merged token carries its sources to the decoder, a pruned token exits early
with its feature at the station. The token score is the paper's (present and forecast class-token attention).

Arms (sealed by SHA before any evaluation, results/seg_comparison/<backbone>/plans.json):
  LEAF          latency-planned at fractions of the measured dense batch-32 latency (BUDGET_FRACTIONS)
  LEAF (GMAC)   GMAC-planned at the GMAC of every LEAF schedule (equal GMAC: what latency pricing changes)
  ToMe          constant r per block, and ToMe at blocks 2 and 5 only (the insertion points of the training-free
                segmentation mergers MPM and SCMerge), both with unmerging before the decoder
  dense         the reference

Stages (GPU stages take the card in CUDA_VISIBLE_DEVICES and refuse a card used by another process):
  search  (GPU)   `leaf search` in <backbone>/search: head groups, probes, budgets, SALT workloads and fit, latency DP
  plan    (CPU)   GMAC-planned LEAF at the LEAF GMACs, baseline grids; sealed plans.json
  worker  (GPU)   ADE20K val (sliding windows, single scale): mIoU, per-image areas; per-window KL to dense
  time    (GPU)   bs32 latency of every arm, three interleaved rounds (middle reversed), nothing else on the card
  report  (CPU)   latency pairs (tau = max(0.05 ms, 2 SD)), paired bootstrap of mIoU over images with Holm, report.json,
                  table.tex, accuracy-latency figure
  release (CPU)   the searched schedules with their measurements -> schedules/leaf/<backbone>.json, planning inputs ->
                  schedules/leaf/planning/<backbone>/ (both tracked; plans.json and the report are tracked too)

Prerequisite (CPU): `leaf data ade20k --download --split all`.
"""
import argparse
from fractions import Fraction
import gc
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np

from leaf.reproduction.experiments.latency_pricing import save, load, immutable, key, card_idle, ROUNDS

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / 'results/seg_comparison'
BB = 'seg_b_ade'
BUDGET_FRACTIONS = (.5, .6, .7, .8, .9)
# Forecast weights: the paper's reduced grid A_4. One probe takes ~20 s on 2,000 windows at 1,025 tokens (RTX 5060 Ti),
# so the full 7-value grid (1,917 probes) would take ~11 GPU-hours; A_4 needs 1,107 probes.
ALPHAS = ('0', '1/2', '2/3', '3/4')
NEAR = .03          # report: nearest baseline point within 3% of a LEAF schedule's latency
GRIDS = {'tome': [dict(r=r) for r in (8, 16, 24, 32, 40, 48, 56, 64, 80)],
         'tome_sparse': [dict(r_blocks=[0, 0, r, 0, 0, round(.75 * r)]) for r in (64, 128, 192, 256, 320, 384, 448, 512)]}


def workspace(bb=BB): return ROOT / bb / 'search'


def runner(bb=BB):
    from leaf.api import LeafRunner
    return LeafRunner(bb, root=ROOT / bb, batch=32)


def search(bb=BB, gpus='0'):
    """The whole schedule search of the method (leaf search), resumable."""
    cmd = [sys.executable, '-m', 'leaf', 'search', '--backbone', bb, '--workspace', str(workspace(bb)), '--gpus', gpus,
           '--anchors', 'dense', '--dense-fractions', *map(str, BUDGET_FRACTIONS), '--alphas', *ALPHAS]
    print(' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=REPO)


def gmac_schedules(bb, points):
    ws = workspace(bb); dest = ws / 'schedules_gmac.json'
    if not dest.exists():
        subprocess.run([sys.executable, '-m', 'leaf', 'plan', '--backbone', bb, '--probes', str(ws / 'probes' / 'probes.json'),
                        '--groups', str(ws / 'groups.json'), '--alphas', *ALPHAS, '--gmac', *(f'{g:.4f}' for g in points), '--out', str(dest)],
                       check=True, cwd=REPO)
    return load(dest)['schedules']


def plan(bb=BB):
    """Every arm, sealed by SHA before any evaluation."""
    dest = ROOT / bb / 'plans.json'
    if dest.exists(): return load(dest)
    from leaf.backbones import SPECS
    from leaf.planner.alpha import config_gmac
    from leaf.planner.gmac import config_cost
    from leaf.reproduction.experiments.gmac_planning import sha
    spec, ws = SPECS[bb], workspace(bb)
    if not (ws / 'schedules.json').exists(): raise SystemExit(f'{ws}/schedules.json missing: run the search stage first')
    jobs = {'ref/dense': dict(config=dict(kind='dense'), group='reference')}
    leaf = {n: r for n, r in load(ws / 'schedules.json')['schedules'].items() if 'config' in r}
    for name, row in leaf.items():
        jobs[f'leaf/{name}'] = dict(config=row['config'], group='leaf', budget=row['budget'], planner=row['planner'])
    for name, row in gmac_schedules(bb, sorted(r['planner']['gmac'] for r in leaf.values())).items():
        if 'config' in row: jobs[f'leaf/{name}'] = dict(config=row['config'], group='leaf_gmac', budget=row['budget'], planner=row['planner'])
    for family, grid in GRIDS.items():
        for params in grid:
            label = f"r{params['r']}" if 'r' in params else 'r' + '-'.join(str(v) for v in params['r_blocks'] if v)
            jobs[f'ref/baseline/{family}/{label}'] = dict(config=dict(kind='tome', **params), group='baseline')
    for row in jobs.values():
        row['config_sha'] = key(row['config'])
        kind = row['config']['kind']
        if kind in ('pd', 'pdm', 'pda'): row['gmac'] = config_gmac(row['config'], spec) if kind == 'pda' else config_cost(row['config'], spec)
    out = dict(backbone=bb, weights=spec.weights, jobs=jobs, sha=sha(jobs), time=time.time(),
               rules=dict(objective='per-pixel KL to the dense decoder on 2,000 ADE20K training windows',
                          tokens='merged tokens carry their sources to the decoder; pruned tokens exit with their feature',
                          accuracy='ADE20K val, single scale, slide 512/480, BF16 batch 32',
                          timing='one session on an otherwise idle card, three interleaved rounds, bs32 windows'))
    immutable(dest, out)
    return out


def tasks(bb=BB):
    out = {}
    for name, row in load(ROOT / bb / 'plans.json')['jobs'].items():
        t = out.setdefault(row['config_sha'], dict(id=row['config_sha'], config=row['config'], names=[]))
        t['names'].append(name)
    return sorted(out.values(), key=lambda t: (0 if any(n.startswith(('leaf/', 'ref/dense')) for n in t['names']) else 1, t['id']))


def worker(bb=BB):
    import torch
    from leaf.evaluation.runner import require_idle_gpu
    from leaf.evaluation.segmentation import summary
    require_idle_gpu(idle=True)
    r = runner(bb)
    for t in tasks(bb):
        done = ROOT / bb / 'completed' / 'val' / f"{t['id']}.json"
        if done.exists(): continue
        try:
            o = r.run(t['config'], 'val')
            save(done, dict(status='ok' if o is not None else 'infeasible', names=t['names'], time=time.time(),
                            **(summary(o) if o is not None else {})))
        except torch.cuda.OutOfMemoryError:
            save(done, dict(status='OOM', names=t['names'])); torch.cuda.empty_cache()
        print('val', t['id'], t['names'][0], flush=True)


def timing(bb=BB):
    import torch
    from leaf.reproduction.runners.exclusive_timing import cast_exclusive
    done = ROOT / bb / 'latency_done.json'
    if done.exists(): return
    if not card_idle(): raise SystemExit('the card is used by another process')
    r = runner(bb); ts = tasks(bb); dest = ROOT / bb / 'latency'
    if dest.exists(): dest.rename(ROOT / bb / f'latency_interrupted_{time.time_ns()}')
    for rnd in range(ROUNDS):
        for t in (ts if rnd % 2 == 0 else ts[::-1]):
            if not card_idle(): raise RuntimeError('timed card shared during timing; session discarded')
            meas = {}
            try:
                m = r.inference_model(t['config'])
                try: meas['32'] = cast_exclusive(m, 32, iters=50, warmup=10)
                except torch.cuda.OutOfMemoryError: meas['32'] = 'OOM'; torch.cuda.empty_cache()
            except Exception as e:
                meas = dict(status='error', error=repr(e))
            save(dest / f"{t['id']}#{rnd}.json", dict(id=t['id'], names=t['names'], round=rnd, measurements=meas,
                                                      device=torch.cuda.get_device_name(0), time=time.time()))
            m = None; gc.collect(); torch.cuda.empty_cache()
        print('round', rnd, flush=True)
    save(done, dict(rounds=ROUNDS, n=len(ts), time=time.time()))


# ---------------------------------------------------------------- report
def family(name):
    if name.startswith('leaf/latency/'): return 'LEAF'
    if name.startswith('leaf/gmac/'): return 'LEAF (GMAC)'
    if name.startswith('ref/baseline/tome_sparse/'): return 'ToMe (blocks 2, 5)'
    if name.startswith('ref/baseline/tome/'): return 'ToMe'
    if name == 'ref/dense': return 'dense'


def arms(bb=BB, bs=32):
    import leaf.reproduction.experiments.campaign_reports as q
    jobs = load(ROOT / bb / 'plans.json')['jobs']; ms = q.round_means(ROOT / bb / 'latency', bs); out = []
    for name, row in jobs.items():
        p = ROOT / bb / 'val' / 'runs' / f"{row['config_sha']}.npz"
        if row['config_sha'] not in ms or not p.exists(): continue
        a = np.load(p)
        out.append(dict(name=name, family=family(name), sha=row['config_sha'], ms=ms[row['config_sha']][0],
                        sd=ms[row['config_sha']][1], gmac=row.get('gmac'), areas=dict(intersect=a['intersect'], union=a['union']),
                        kl=float(a['kl'].mean())))
    return out


def analyse(bb=BB, bs=32, resamples=10_000):
    import leaf.reproduction.experiments.campaign_reports as q
    from leaf.evaluation.data import holm
    from leaf.evaluation.segmentation import miou, paired_bootstrap
    A = arms(bb, bs); ours = [a for a in A if a['family'] == 'LEAF']
    pairs = []
    for o in ours:
        for fam in sorted({a['family'] for a in A} - {'LEAF', 'dense'}):
            best = min((a for a in A if a['family'] == fam), key=lambda a: abs(a['ms'] - o['ms']))
            t = q.tau((o['ms'], o['sd']), (best['ms'], best['sd']))
            if abs(best['ms'] - o['ms']) <= t:
                pairs.append(dict(ours=o['name'], family=fam, ref=best['name'], ours_ms=o['ms'], ref_ms=best['ms'], tau=t,
                                  **paired_bootstrap(o['areas'], best['areas'], resamples)))
    holm(pairs)
    # The baseline grids are coarse next to the timing SD (pilot: no densify stage), so few pairs fall within tau. Also
    # report, per LEAF schedule and family, the nearest point within NEAR of its latency (bootstrap, Holm over these
    # pairs; the latency gap is part of the record) and the family's frontier linearly interpolated at LEAF's latency.
    near = []
    for o in ours:
        for fam in sorted({a['family'] for a in A} - {'LEAF', 'dense'}):
            cand = [a for a in A if a['family'] == fam]
            best = min(cand, key=lambda a: abs(a['ms'] - o['ms']))
            row = dict(ours=o['name'], family=fam, ours_ms=o['ms'], ours_miou=miou(o['areas']))
            if abs(best['ms'] - o['ms']) <= NEAR * o['ms']:
                row.update(ref=best['name'], ref_ms=best['ms'], gap_ms=best['ms'] - o['ms'],
                           **paired_bootstrap(o['areas'], best['areas'], resamples))
            curve = sorted((a['ms'], miou(a['areas'])) for a in cand)
            lo = [c for c in curve if c[0] <= o['ms']]; hi = [c for c in curve if c[0] >= o['ms']]
            if lo and hi:
                (x0, y0), (x1, y1) = lo[-1], hi[0]
                row['frontier_miou'] = y0 if x1 == x0 else y0 + (o['ms'] - x0) / (x1 - x0) * (y1 - y0)
                row['frontier_delta'] = row['ours_miou'] - row['frontier_miou']
            near.append(row)
    holm([r for r in near if 'p' in r])
    points = [dict(name=a['name'], family=a['family'], ms=a['ms'], sd=a['sd'], gmac=a['gmac'], miou=miou(a['areas']), kl=a['kl'])
              for a in A]
    return dict(backbone=bb, batch=bs, points=sorted(points, key=lambda p: (p['family'], p['ms'])), latency_pairs=pairs,
                near_pairs=near, near=NEAR)


def table(rep):
    lines = ['\\begin{tabular}{l l r r r}\\toprule', 'Family & Point & ms $\\downarrow$ & GMAC & mIoU $\\uparrow$ \\\\ \\midrule']
    for p in rep['points']:
        g = f"{p['gmac']:.1f}" if p['gmac'] else '--'
        lines.append(f"{p['family']} & {p['name'].split('/')[-1]} & {p['ms']:.2f} & {g} & {p['miou']:.2f} \\\\")
    lines.append('\\bottomrule\\end{tabular}')
    return '\n'.join(lines) + '\n'


def figure(rep, dest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for fam in sorted({p['family'] for p in rep['points']}):
        pts = sorted((p for p in rep['points'] if p['family'] == fam), key=lambda p: p['ms'])
        ax.plot([p['ms'] for p in pts], [p['miou'] for p in pts], 'o-' if fam != 'dense' else '*', label=fam, ms=4)
    ax.set_xlabel(f"latency, batch {rep['batch']} windows (ms)"); ax.set_ylabel('ADE20K mIoU (%)'); ax.grid(alpha=.3); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(dest); fig.savefig(Path(dest).with_suffix('.png'), dpi=160); plt.close(fig)


def report(bb=BB):
    rep = analyse(bb)
    save(ROOT / bb / 'report.json', rep)
    (ROOT / bb / 'table.tex').write_text(table(rep))
    figure(rep, ROOT / bb / 'accuracy_latency.pdf')
    for p in rep['latency_pairs']:
        print(f"{p['ours']:28s} vs {p['family']:20s} {p['ref']:34s} {p['ours_ms']:7.2f}/{p['ref_ms']:7.2f} ms  "
              f"dmIoU {p['delta']:+.2f} [{p['lo']:+.2f}, {p['hi']:+.2f}] Holm p {p['holm']:.3g}")
    print(f"{len(rep['latency_pairs'])} pairs within tau; nearest points within {100 * NEAR:g}% latency and frontiers:")
    for p in rep['near_pairs']:
        pair = (f"{p['ref'].split('/')[-1]:10s} {p['gap_ms']:+6.2f} ms  dmIoU {p['delta']:+.2f} [{p['lo']:+.2f}, {p['hi']:+.2f}] "
                f"Holm p {p['holm']:.3g}") if 'ref' in p else f"{'--':10s} (none within {100 * NEAR:g}%)"
        front = f"  frontier d {p['frontier_delta']:+.2f}" if 'frontier_delta' in p else '  frontier --'
        print(f"{p['ours'].split('/')[-1]:9s} {p['ours_ms']:7.2f} ms {p['ours_miou']:.2f} vs {p['family']:20s} {pair}{front}")


PLANNING_FILES = ('groups.json', 'probes/probes.json', 'budgets.json', 'salt_workloads.json', 'salt_timings.json', 'salt.json')
SUMMARY_FILES = ('plans.json', 'report.json', 'table.tex', 'accuracy_latency.pdf', 'accuracy_latency.png', 'REPORT.md')


def release(bb=BB):
    """Tracked outputs: schedules/leaf/<bb>.json (the searched schedules with their measured mIoU and latency, the format
    of schedules/README.md), the planning inputs in schedules/leaf/planning/<bb>/, and the pilot's sealed plans and
    report in results/seg_comparison/<bb>/ (the only files of results/ in git besides README and SUMMARY)."""
    import shutil
    from leaf.api import stations
    from leaf.backbones import SPECS
    from leaf.evaluation.segmentation import miou
    spec, ws, root = SPECS[bb], workspace(bb), ROOT / bb
    rep = load(root / 'report.json'); points = {p['name']: p for p in rep['points']}
    jobs = load(root / 'plans.json')['jobs']
    plan_docs = {'latency': load(ws / 'schedules.json'), 'gmac': load(ws / 'schedules_gmac.json')}
    out = dict(backbone=bb, weights=spec.weights, task='semantic segmentation, ADE20K (150 classes), 512x512 windows',
               method='LEAF: stations scored by (1 - alpha) z(log P) + alpha z(D), prune -> M3 merge, planned by the '
                      'Lagrangian DP under SALT; merged tokens carry their sources to the decoder, pruned tokens exit early',
               objective=load(root / 'plans.json')['rules']['objective'],
               forecast=dict(G=16, rank=32), operator='M3', alpha_grid=[float(Fraction(a)) for a in ALPHAS], theta_grid=[0., .5, 1.],
               hardware='NVIDIA RTX 5060 Ti, BF16, CUDA graph, batch 32 windows; latency = mean of three interleaved rounds',
               dense=dict(val_miou=points['ref/dense']['miou'], ms_bs32=points['ref/dense']['ms'], sd=points['ref/dense']['sd']),
               sources=dict(plans=str((root / 'plans.json').relative_to(REPO)), report=str((root / 'report.json').relative_to(REPO))),
               schedules={})
    for kind, doc in plan_docs.items():
        for key, row in doc['schedules'].items():
            if 'config' not in row: continue
            name = f'leaf/{key}'; p = points.get(name)
            out['schedules'][key] = dict(budget=row['budget'], stations=stations(row['config'], spec), config=row['config'],
                                         planner=row['planner'], config_sha=jobs[name]['config_sha'],
                                         measured=None if p is None else dict(val_miou=p['miou'], ms_bs32=p['ms'], sd=p['sd'], kl=p['kl']))
    dest = REPO / 'schedules/leaf' / f'{bb}.json'
    save(dest, out)
    planning = REPO / 'schedules/leaf/planning' / bb; planning.mkdir(parents=True, exist_ok=True)
    for f in PLANNING_FILES:
        if (ws / f).exists(): shutil.copyfile(ws / f, planning / Path(f).name)
    missing = [f for f in SUMMARY_FILES if not (root / f).exists()]
    print(f'wrote {dest.relative_to(REPO)} ({len(out["schedules"])} schedules), {planning.relative_to(REPO)}/'
          + (f'; missing summaries: {missing}' if missing else ''), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(prog='seg_comparison')
    ap.add_argument('stage', choices=('search', 'plan', 'worker', 'time', 'report', 'release'))
    ap.add_argument('--backbone', default=BB)
    ap.add_argument('--gpus', default='0', help='search: GPU indices for `leaf search` (one probe shard per GPU)')
    args = ap.parse_args(argv)
    if args.stage == 'search': search(args.backbone, args.gpus)
    elif args.stage == 'plan': print(plan(args.backbone)['sha'])
    elif args.stage == 'worker': worker(args.backbone)
    elif args.stage == 'time': timing(args.backbone)
    elif args.stage == 'report': report(args.backbone)
    else: release(args.backbone)


if __name__ == '__main__':
    main()
