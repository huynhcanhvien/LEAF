"""CLIP comparison: LEAF and the re-run families on CLIP ViT-B/16 and ViT-L/14 (OpenAI, 224 px), zero-shot ImageNet
and image-text retrieval.

Same protocol as the main comparison (main_comparison.py): every arm planned and sealed by SHA before any evaluation,
BF16 batch 32, one exclusive timing session per backbone with three interleaved rounds, latency pairing with
tau = max(0.05 ms, 2 SD), McNemar with Holm per split and paired 95% CIs. LEAF is planned once, on the KL of the
zero-shot ImageNet distribution over the 5,000 probe images; the same schedules are then evaluated on retrieval
without re-planning. DiffRate has no released CLIP schedules: the rates of the DiffRate backbone of the same depth are
transferred (`diffrate_key` in leaf/backbones.json: DeiT-B for clip_b16, MAE-L for clip_l14, whose 257 tokens get the
same kept fraction of patch tokens, leaf/planner/configs.py) and reported as 'DiffRate (DeiT-B rates)' / '(MAE-L rates)'.

Stages (results/clip_comparison/<backbone>/):
  search  (GPUs)     `leaf search` in <backbone>/search: head groups, probes, budgets anchored at 0.97x the transferred
                     DiffRate latencies (+ extra1/extra2), SALT workloads and fit, the latency-budgeted DP
  plan    (CPU)      GMAC-planned LEAF at the GMAC of every transferred DiffRate schedule; baseline grids; sealed plans.json
  worker  (GPU)      val, v2 (zero-shot top-1, KL), flickr_test, coco_test (recall@k) for every arm; one process per card
  time    (GPU0)     bs8/bs32 latency of every arm, three interleaved rounds, nothing else on the card
  report  (CPU)      latency pairs and equal-GMAC pairs -> report.json, table.tex, frontier figure (bs32; *_bs8 at batch 8)
  densify (GPU0)     cost-only extra baseline points near the latency of every LEAF schedule (pilot timing only, no
                     accuracy), sealed in plans_dense.json; then worker, retime (a new session of every arm) and report
  plan-dr-search (CPU) DiffRate's own search on CLIP ViT-B/16 (diffrate_clip_search.py, mentor comment L737): the searched
                     schedules and GMAC-planned LEAF at their GMAC, sealed in plans_dr_search.json (budgets searched
                     later: plans_dr_search2.json, plans_dr_search3.json); then worker, retime
                     and report ('DiffRate (searched on CLIP)')

Prerequisites (CPU): `leaf data clip-text --backbone <backbone>`, `leaf data prepare --split flickr_test|coco_test`.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np

from leaf.reproduction.experiments.latency_pricing import save, load, immutable, key, card_idle, ROUNDS

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / 'results/clip_comparison'
BBS = ('clip_b16', 'clip_l14')
CLASSIFICATION, RETRIEVAL = ('val', 'v2'), ('flickr_test', 'coco_test')
BATCHES = (8, 32)
GMAC_TOL = .03
# The main campaign's grids (main_comparison.py + its historical and low-budget extension grids), unchanged.
GRIDS = dict(tome=dict(r=(4, 6, 8, 10, 12, 13, 14, 16, 20, 24, 32, 48)), tome_m3=dict(r=(4, 6, 8, 10, 12, 13, 14, 16, 20, 24, 32, 48)),
             tofu=dict(r=(4, 6, 8, 10, 12, 14, 16, 20, 24, 32, 48)), rapid=dict(r=(4, 6, 8, 10, 12, 14, 16, 20, 24, 32, 48)),
             pitome=dict(ratio=(.8, .825, .85, .875, .9, .925, .95)), evit=dict(keep_rate=(.15, .25, .35, .4, .45, .5, .6, .7, .8, .9)),
             ppt=dict(r=(20, 30, 40, 50, 60)), mctf=dict(r=(4, 6, 8, 10, 12, 14, 16)), onemore=dict(n=(40, 70, 100, 130, 160, 190)))
DR_FAMILIES = {'clip_b16': 'DiffRate (DeiT-B rates)', 'clip_l14': 'DiffRate (MAE-L rates)'}
DR_FAMILY = DR_FAMILIES['clip_b16']
DR_SEARCH = REPO / 'results/diffrate_clip_search/rates.json'
DR_SEARCH_FAMILY = 'DiffRate (searched on CLIP)'
EXTRA_PLANS = ('plans_dense.json', 'plans_dr_search.json', 'plans_dr_search2.json', 'plans_dr_search3.json')


def workspace(bb): return ROOT / bb / 'search'


def runner(bb):
    from leaf.api import LeafRunner
    return LeafRunner(bb, root=ROOT / bb, batch=32)


# ---------------------------------------------------------------- search and plan
def search(bb, gpus):
    """The whole schedule search of the method (leaf search), resumable."""
    cmd = [sys.executable, '-m', 'leaf', 'search', '--backbone', bb, '--workspace', str(workspace(bb)), '--gpus', gpus]
    print(' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=REPO)


def diffrate_points(bb):
    """{DR budget: (config, GMAC)} of the transferred DiffRate schedules."""
    from leaf.backbones import SPECS
    from leaf.planner.configs import diffrate_config, diffrate_schedules
    from leaf.planner.gmac import config_cost
    spec = SPECS[bb]
    return {b: (diffrate_config(spec, b), config_cost(diffrate_config(spec, b), spec)) for b in diffrate_schedules(spec)}


def gmac_schedules(bb):
    """LEAF under a GMAC budget at the GMAC of each transferred DiffRate schedule (CPU, same probes and grid as search)."""
    ws = workspace(bb); dest = ws / 'schedules_gmac.json'
    if not dest.exists():
        points = [f'{g:.4f}' for _, g in diffrate_points(bb).values()]
        subprocess.run([sys.executable, '-m', 'leaf', 'plan', '--backbone', bb, '--probes', str(ws / 'probes' / 'probes.json'),
                        '--groups', str(ws / 'groups.json'), '--gmac', *points, '--out', str(dest)], check=True, cwd=REPO)
    return load(dest)['schedules']


def plan(bb):
    """Every arm of the comparison, sealed by SHA before any evaluation."""
    dest = ROOT / bb / 'plans.json'
    if dest.exists(): return load(dest)
    from leaf.backbones import SPECS
    from leaf.planner.gmac import config_cost
    from leaf.reproduction.experiments.gmac_planning import sha
    spec, ws = SPECS[bb], workspace(bb)
    if not (ws / 'schedules.json').exists(): raise SystemExit(f'{ws}/schedules.json missing: run the search stage first')
    jobs = {'ref/dense': dict(config=dict(kind='dense'), group='reference')}
    for name, row in load(ws / 'schedules.json')['schedules'].items():
        if 'config' in row: jobs[f'leaf/{name}'] = dict(config=row['config'], group='leaf', budget=row['budget'], planner=row['planner'])
    for name, row in gmac_schedules(bb).items():
        if 'config' in row: jobs[f'leaf/{name}'] = dict(config=row['config'], group='leaf_gmac', budget=row['budget'], planner=row['planner'])
    for b, (cfg, _) in diffrate_points(bb).items():
        jobs[f'ref/diffrate_transfer/{b}'] = dict(config=cfg, group='baseline', source=f'DiffRate rates of {spec.diffrate_key}')
    for kind, grid in GRIDS.items():
        (param, values), = grid.items()
        for v in values: jobs[f'ref/baseline/{kind}/{param}{v}'] = dict(config={'kind': kind, param: v}, group='baseline')
    for row in jobs.values():
        row['config_sha'] = key(row['config'])
        if row['config']['kind'] in ('pd', 'pdm', 'pda'):
            from leaf.planner.alpha import config_gmac
            row['gmac'] = config_gmac(row['config'], spec) if row['config']['kind'] == 'pda' else config_cost(row['config'], spec)
    out = dict(backbone=bb, timm_id=spec.timm_id, jobs=jobs, sha=sha(jobs), time=time.time(),
               rules=dict(objective='KL to the dense zero-shot ImageNet distribution on the 5,000 probe images',
                          retrieval='the same sealed schedules, no re-planning', diffrate=DR_FAMILIES[bb],
                          accuracy='BF16 deterministic batch 32', timing='one exclusive session, three interleaved rounds, bs8/bs32'))
    immutable(dest, out)
    return out


def jobs_of(bb):
    """Every sealed arm: plans.json and, after the densify and plan-dr-search stages, their plans."""
    jobs = dict(load(ROOT / bb / 'plans.json')['jobs'])
    for extra in EXTRA_PLANS:
        if (ROOT / bb / extra).exists(): jobs.update(load(ROOT / bb / extra)['jobs'])
    return jobs


def plan_dr_search(bb='clip_b16'):
    """DiffRate schedules searched on this backbone and LEAF under a GMAC budget at their GMAC, sealed before any
    evaluation of them. A later search round (budgets searched after an earlier seal) is sealed in the next
    plans_dr_search<k>.json with only its new budgets; earlier rounds are never rewritten."""
    rounds = [f for f in EXTRA_PLANS if f.startswith('plans_dr_search')]
    done = [ROOT / bb / f for f in rounds if (ROOT / bb / f).exists()]
    sealed = {name for p in done for name in load(p)['jobs'] if name.startswith('ref/diffrate_search/')}
    rates = load(DR_SEARCH)['rates']['CLIP-B16-searched']
    new = {b: r for b, r in rates.items() if f'ref/diffrate_search/{b}' not in sealed}
    if not new: return load(done[-1])
    dest = ROOT / bb / rounds[len(done)]; suffix = dest.stem[len('plans_dr_search'):]
    from leaf.backbones import SPECS
    from leaf.planner.configs import diffrate_config_from
    from leaf.planner.gmac import config_cost
    from leaf.planner.alpha import config_gmac
    from leaf.reproduction.experiments.gmac_planning import sha
    spec, ws = SPECS[bb], workspace(bb)
    jobs = {}
    for b, r in new.items():
        cfg = diffrate_config_from(spec, json.loads(r['prune_kept_num']), json.loads(r['merge_kept_num']))
        jobs[f'ref/diffrate_search/{b}'] = dict(config=cfg, group='baseline', source=load(DR_SEARCH)['source'], gmac=config_cost(cfg, spec))
    gm = ws / f'schedules_gmac_dr_search{suffix}.json'
    if not gm.exists():
        points = [f"{row['gmac']:.4f}" for row in jobs.values()]
        subprocess.run([sys.executable, '-m', 'leaf', 'plan', '--backbone', bb, '--probes', str(ws / 'probes' / 'probes.json'),
                        '--groups', str(ws / 'groups.json'), '--gmac', *points, '--out', str(gm)], check=True, cwd=REPO)
    for name, row in load(gm)['schedules'].items():
        if 'config' in row:
            jobs[f'leaf/{name}'] = dict(config=row['config'], group='leaf_gmac', budget=row['budget'], planner=row['planner'],
                                        gmac=config_gmac(row['config'], spec) if row['config']['kind'] == 'pda' else config_cost(row['config'], spec))
    for row in jobs.values(): row['config_sha'] = key(row['config'])
    out = dict(backbone=bb, jobs=jobs, sha=sha(jobs), time=time.time(), rates=str(DR_SEARCH.relative_to(REPO)))
    immutable(dest, out)
    return out


def tasks(bbs=BBS):
    out = {}
    for bb in bbs:
        if not (ROOT / bb / 'plans.json').exists(): continue
        for name, row in jobs_of(bb).items():
            t = out.setdefault(f"{bb}-{row['config_sha']}", dict(id=f"{bb}-{row['config_sha']}", backbone=bb, config=row['config'], names=[]))
            t['names'].append(name)
    # The method, DiffRate and dense first, so the core rows complete early.
    return sorted(out.values(), key=lambda t: (0 if any(n.startswith(('leaf/', 'ref/dense', 'ref/diffrate')) for n in t['names']) else 1, t['id']))


# ---------------------------------------------------------------- accuracy and retrieval
def worker(bbs=BBS):
    import torch
    from leaf.evaluation.retrieval import RetrievalRunner, recall
    from leaf.reproduction.runners.exclusive_timing import claim, guard_foreign
    guard_foreign()
    rs = {}
    for split in CLASSIFICATION + RETRIEVAL:
        for t in tasks(bbs):
            done = ROOT / 'completed' / split / f"{t['id']}.json"
            if done.exists() or not claim(ROOT, split, t['id']): continue
            bb = t['backbone']
            if bb not in rs:
                rs.clear(); gc.collect(); torch.cuda.empty_cache(); rs[bb] = runner(bb)
            try:
                if split in CLASSIFICATION:
                    o = rs[bb].run(t['config'], split)
                    row = dict(top1=float(o['correct'].mean() * 100), kl=float(o['kl'].mean())) if o is not None else {}
                else:
                    o = RetrievalRunner(rs[bb]).run(t['config'], split)
                    row = recall(o['i2t'], o['t2i']) if o is not None else {}
                save(done, dict(status='ok' if o is not None else 'infeasible', names=t['names'], time=time.time(), **row))
            except torch.cuda.OutOfMemoryError:
                save(done, dict(status='OOM', names=t['names']))
            print(split, t['id'], t['names'][0], flush=True)


# ---------------------------------------------------------------- latency
def timing(bbs=BBS):
    import torch
    from leaf.reproduction.runners.exclusive_timing import cast_exclusive
    for bb in bbs:
        done = ROOT / bb / 'latency_done.json'
        if done.exists(): continue
        while not card_idle(): time.sleep(30)
        r = runner(bb); ts = tasks((bb,)); dest = ROOT / bb / 'latency'
        if dest.exists(): dest.rename(ROOT / bb / f'latency_interrupted_{time.time_ns()}')
        for rnd in range(ROUNDS):
            for t in (ts if rnd % 2 == 0 else ts[::-1]):
                if not card_idle(): raise RuntimeError('timed card shared during timing; session discarded')
                meas = {}
                try:
                    m = r.inference_model(t['config'])
                    for b in BATCHES:
                        try: meas[str(b)] = cast_exclusive(m, b, iters=50, warmup=10)
                        except torch.cuda.OutOfMemoryError: meas[str(b)] = 'OOM'; torch.cuda.empty_cache()
                except Exception as e:
                    meas = dict(status='error', error=repr(e))
                save(dest / f"{t['id']}#{rnd}.json", dict(id=t['id'], names=t['names'], round=rnd, measurements=meas,
                                                          device=torch.cuda.get_device_name(0), time=time.time()))
                m = None; gc.collect(); torch.cuda.empty_cache()
        save(done, dict(rounds=ROUNDS, n=len(ts), time=time.time()))
        r = None; gc.collect(); torch.cuda.empty_cache()
        print('timed', bb, len(ts), flush=True)


# ---------------------------------------------------------------- report
def family(name, cfg, bb='clip_b16'):
    import leaf.reproduction.experiments.campaign_reports as q
    if name.startswith('leaf/latency/'): return 'LEAF'
    if name.startswith('leaf/gmac/'): return 'LEAF (GMAC)'
    if name.startswith('ref/diffrate_transfer/'): return DR_FAMILIES[bb]
    if name.startswith('ref/diffrate_search/'): return DR_SEARCH_FAMILY
    if name == 'ref/dense': return 'dense'
    return q.KIND.get(cfg['kind'])


def hits(bb, sha, split):
    """Per-query correctness: top-1 per image (classification); R@1 per image (i2t) and per caption (t2i) (retrieval)."""
    p = ROOT / bb / split / 'runs' / f'{sha}.npz'
    if not p.exists(): return None
    a = np.load(p)
    return {split: a['correct']} if split in CLASSIFICATION else {f'{split}_i2t': a['i2t'] == 0, f'{split}_t2i': a['t2i'] == 0}


def arms(bb, bs=32):
    import leaf.reproduction.experiments.campaign_reports as q
    jobs = jobs_of(bb); ms = q.round_means(ROOT / bb / 'latency', bs); out = []
    for name, row in jobs.items():
        tid = f"{bb}-{row['config_sha']}"
        if tid not in ms: continue
        a = dict(name=name, family=family(name, row['config'], bb), sha=row['config_sha'], ms=ms[tid][0], sd=ms[tid][1], gmac=row.get('gmac'), hits={})
        for split in CLASSIFICATION + RETRIEVAL:
            h = hits(bb, row['config_sha'], split)
            if h: a['hits'].update(h)
        if a['family'] and a['hits']: out.append(a)
    return out


def metric(a, split):
    return 100 * float(np.mean(a['hits'][split]))


def compare(o, ref, extra):
    from leaf.reproduction.experiments.main_comparison_analysis import paired_ci
    from leaf.evaluation.data import mcnemar
    rec = dict(ours=o['name'], family=ref['family'], ref=ref['name'], ours_ms=o['ms'], ref_ms=ref['ms'], **extra)
    for s in sorted(set(o['hits']) & set(ref['hits'])):
        p, _, _ = mcnemar(o['hits'][s], ref['hits'][s])
        rec[s] = dict(delta=metric(o, s) - metric(ref, s), p=p, **{f'ci_{k}': v for k, v in paired_ci(o['hits'][s], ref['hits'][s]).items()})
    return rec


def analyse(bb, bs=32):
    import leaf.reproduction.experiments.campaign_reports as q
    from leaf.evaluation.data import holm
    A = arms(bb, bs); ours = [a for a in A if a['family'] == 'LEAF']; ours_g = [a for a in A if a['family'] == 'LEAF (GMAC)']
    latency_pairs, gmac_pairs = [], []
    for o in ours:
        for fam in sorted({a['family'] for a in A} - {'LEAF', 'LEAF (GMAC)', 'dense'}):
            best = min((a for a in A if a['family'] == fam), key=lambda a: abs(a['ms'] - o['ms']))
            t = q.tau((o['ms'], o['sd']), (best['ms'], best['sd']))
            if abs(best['ms'] - o['ms']) <= t: latency_pairs.append(compare(o, best, dict(tau=t)))
    for o in ours_g:   # equal GMAC with the transferred (and searched) DiffRate schedules (same GMAC convention, config_cost)
        for fam in (DR_FAMILIES[bb], DR_SEARCH_FAMILY):
            cands = [a for a in A if a['family'] == fam and a['gmac']]
            if not cands: continue
            best = min(cands, key=lambda a: abs(a['gmac'] - o['gmac']))
            if abs(best['gmac'] - o['gmac']) <= GMAC_TOL * best['gmac']:
                gmac_pairs.append(compare(o, best, dict(ours_gmac=o['gmac'], ref_gmac=best['gmac'])))
    splits = sorted({k for r in latency_pairs + gmac_pairs for k in r if isinstance(r[k], dict)})
    for pairs in (latency_pairs, gmac_pairs):
        for s in splits: holm([r[s] for r in pairs if s in r])
    summary = {}
    for r in latency_pairs:
        f = summary.setdefault(r['family'], dict(pairs=0, **{s: dict(win=0, loss=0, deltas=[]) for s in splits}))
        f['pairs'] += 1
        for s in splits:
            if s not in r: continue
            f[s]['deltas'].append(r[s]['delta'])
            if r[s]['holm'] < .05: f[s]['win' if r[s]['delta'] > 0 else 'loss'] += 1
    for f in summary.values():
        for s in splits: f[s]['mean'] = float(np.mean(f[s]['deltas'])) if f[s]['deltas'] else None
    points = [dict(name=a['name'], family=a['family'], ms=a['ms'], sd=a['sd'], gmac=a['gmac'],
                   **{s: metric(a, s) for s in a['hits']}) for a in A]
    return dict(backbone=bb, batch=bs, splits=splits, points=points, latency_pairs=latency_pairs, gmac_pairs=gmac_pairs, summary=summary)


def table(rep):
    cols = [s for s in ('val', 'v2', 'flickr_test_i2t', 'flickr_test_t2i', 'coco_test_i2t', 'coco_test_t2i') if s in rep['splits']]
    head = dict(val='Val', v2='V2', flickr_test_i2t='F30k I$\\to$T', flickr_test_t2i='F30k T$\\to$I',
                coco_test_i2t='COCO I$\\to$T', coco_test_t2i='COCO T$\\to$I')
    lines = ['\\begin{tabular}{l r ' + ' '.join('rr' for _ in cols) + '}\\toprule',
             'Family & Pairs & ' + ' & '.join(f'\\multicolumn{{2}}{{c}}{{{head[c]}}}' for c in cols) + ' \\\\',
             ' & & ' + ' & '.join('W/L & $\\bar\\Delta$$\\uparrow$' for _ in cols) + ' \\\\ \\midrule']
    for fam, f in sorted(rep['summary'].items()):
        cells = [f"{f[c]['win']}/{f[c]['loss']} & {f[c]['mean']:+.2f}" if f[c]['mean'] is not None else '-- & --' for c in cols]
        lines.append(f"{fam} & {f['pairs']} & " + ' & '.join(cells) + ' \\\\')
    lines.append('\\bottomrule\\end{tabular}')
    return '\n'.join(lines) + '\n'


def figure(rep, dest):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import leaf.reproduction.experiments.campaign_reports as q
    panels = [s for s in ('val', 'coco_test_i2t', 'coco_test_t2i') if s in rep['splits']]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.4), squeeze=False)
    for ax, s in zip(axes[0], panels):
        for fam in sorted({p['family'] for p in rep['points']}):
            pts = sorted((p['ms'], p[s]) for p in rep['points'] if p['family'] == fam and s in p)
            if not pts: continue
            color = q.COLORS.get(fam, q.COLORS['DiffRate'] if fam.startswith('DiffRate') else ('black' if fam == 'dense' else q.GRAY))
            front = [pts[i] for i in sorted(q.pareto(pts))]
            ax.plot(*zip(*front), color=color, lw=1.2 if fam.startswith(('LEAF', 'DiffRate')) else .7)
            ax.scatter(*zip(*pts), s=14, color=color, marker='*' if fam == 'dense' else q.MARKERS.get(fam.split(' ')[0], 'o'), label=fam)
        ax.set_xlabel(f"ms / batch {rep.get('batch', 32)}"); ax.set_ylabel({'val': 'zero-shot top-1 (%)'}.get(s, s.replace('_', ' ') + ' R@1 (%)'))
    axes[0][0].legend(fontsize=6, ncol=2)
    fig.tight_layout(); fig.savefig(dest); plt.close(fig)


def report(bbs=BBS):
    for bb in bbs:
        for bs in BATCHES:
            rep = analyse(bb, bs); sfx = '' if bs == 32 else f'_bs{bs}'
            save(ROOT / bb / f'report{sfx}.json', rep)
            (ROOT / bb / f'table{sfx}.tex').write_text(table(rep))
            if rep['points']: figure(rep, ROOT / bb / f'frontier{sfx}.pdf')
            print('report', bb, f'bs{bs}', len(rep['latency_pairs']), 'latency pairs', len(rep['gmac_pairs']), 'equal-GMAC pairs', flush=True)


# ---------------------------------------------------------------- extra baseline points near the LEAF latencies
# Cost-only and fixed before any accuracy of the new points is seen: candidates are predicted from the latency of the
# family's grid (session 1), timed once (pilot), and for every LEAF schedule the nearest pilot point on each side is kept.
# ToMe, ToMe+M3, ToFu and RAPID take ToMe's (r, inflection) form for latency steps finer than one r. MCTF gets points
# only if one of its grid points is faster than the dense model (not on ViT-B/16; decided from latency alone).
DENSE = ('tome', 'tome_m3', 'tofu', 'rapid', 'pitome', 'evit', 'ppt', 'onemore')
R_KINDS = ('tome', 'tome_m3', 'tofu', 'rapid')
PILOT_PER_SIDE = 3


def candidates(kind, spec):
    """{config_sha: config} of the fine parameter range of a family."""
    if kind in R_KINDS:
        from baselines.methods import parse_r
        out, seen = {}, set()
        for r in range(2, 41):
            for inflect in np.round(np.arange(-.6, .601, .02), 2):
                sched = tuple(parse_r(spec.depth, (r, float(inflect))))
                if min(sched) < 0 or sched in seen: continue
                seen.add(sched)
                cfg = {'kind': kind, 'r': r} if inflect == 0 else {'kind': kind, 'r': [r, float(inflect)]}
                out[key(cfg)] = cfg
        return out
    grid = dict(pitome=('ratio', np.round(np.arange(.75, .9601, .001), 3)), evit=('keep_rate', np.round(np.arange(.1, .9501, .005), 3)),
                ppt=('r', range(10, (spec.n_tokens + 1) // 2)), mctf=('r', range(1, 41)), onemore=('n', range(3, spec.n_tokens + 1)))[kind]
    return {key(c): c for c in ({'kind': kind, grid[0]: v.item() if hasattr(v, 'item') else v} for v in grid[1])}


def scalar(cfg, spec):
    """Monotone cost proxy of a config of its family: tokens summed over blocks (r kinds) or the parameter itself."""
    if cfg['kind'] in R_KINDS:
        from baselines.methods import parse_r
        r = cfg['r']; total = 0; n = spec.n_tokens
        for rl in parse_r(spec.depth, (r[0], r[1]) if isinstance(r, list) else r):
            n -= min(rl, (n - 1) // 2); total += n
        return total
    return cfg[dict(pitome='ratio', evit='keep_rate', ppt='r', mctf='r', onemore='n')[cfg['kind']]]


def densify(bb):
    """Pilot-time the candidates nearest every LEAF latency and seal the chosen points (plans_dense.json)."""
    import torch
    import leaf.reproduction.experiments.campaign_reports as q
    from leaf.reproduction.runners.exclusive_timing import cast_exclusive
    from leaf.reproduction.experiments.gmac_planning import sha
    from leaf.backbones import SPECS
    dest = ROOT / bb / 'plans_dense.json'
    if dest.exists(): return load(dest)
    spec = SPECS[bb]
    jobs = load(ROOT / bb / 'plans.json')['jobs']; ms = q.round_means(ROOT / bb / 'latency', 32)
    ms_of = {name: ms[f"{bb}-{row['config_sha']}"][0] for name, row in jobs.items() if f"{bb}-{row['config_sha']}" in ms}
    targets = sorted(ms_of[n] for n in jobs if n.startswith('leaf/latency/'))
    have = {row['config_sha'] for row in jobs.values()}
    pilot_dir = ROOT / bb / 'pilot'; r = None
    chosen = {}
    faster = [ms_of[n] for n, row in jobs.items() if row['config']['kind'] == 'mctf' and n in ms_of and ms_of[n] < ms_of['ref/dense']]
    for kind in DENSE + (('mctf',) if faster else ()):
        grid = sorted((scalar(row['config'], spec), ms_of[n]) for n, row in jobs.items() if row['config']['kind'] == kind and n in ms_of)
        xs, ys = np.array([g[0] for g in grid], float), np.array([g[1] for g in grid])
        cands = candidates(kind, spec); pred = {c: float(np.interp(scalar(cfg, spec), xs, ys)) for c, cfg in cands.items()}
        todo = set()
        for t in targets:
            below = sorted((c for c in cands if pred[c] <= t), key=lambda c: t - pred[c])[:PILOT_PER_SIDE]
            above = sorted((c for c in cands if pred[c] > t), key=lambda c: pred[c] - t)[:PILOT_PER_SIDE]
            todo.update(below + above)
        pilot = {}
        for c in sorted(todo):
            p = pilot_dir / f'{c}.json'
            if not p.exists():
                if r is None:
                    while not card_idle(): time.sleep(30)
                    r = runner(bb)
                m = r.inference_model(cands[c])
                save(p, dict(config=cands[c], predicted=pred[c], measurement=cast_exclusive(m, 32, iters=50, warmup=10)))
                m = None; gc.collect(); torch.cuda.empty_cache()
            pilot[c] = load(p)['measurement']['ms_mean']
        for t in targets:
            below = [c for c in pilot if pilot[c] <= t]; above = [c for c in pilot if pilot[c] > t]
            for side in (max(below, key=lambda c: pilot[c], default=None), min(above, key=lambda c: pilot[c], default=None)):
                if side and side not in have:
                    cfg = cands[side]; param = 'r' if kind in R_KINDS or kind == 'ppt' else next(k for k in cfg if k != 'kind')
                    v = cfg[param]; label = f"r{v[0]}i{v[1]:+.2f}" if isinstance(v, list) else f'{param}{v}'
                    chosen[f'ref/baseline_dense/{kind}/{label}'] = dict(config=cfg, group='baseline', pilot_ms=pilot[side], config_sha=side)
        print('densify', bb, kind, len(todo), 'pilot points', flush=True)
    out = dict(backbone=bb, jobs=chosen, sha=sha(chosen), time=time.time(), targets_ms=targets,
               rule='cost only: candidates nearest each LEAF latency by interpolated session-1 latency, one pilot timing (bs32), '
                    'nearest pilot point on each side kept; sealed before any accuracy of these points')
    immutable(dest, out)
    return out


def retime(bbs=BBS):
    """A new exclusive session of every arm (grids, dense points, LEAF, DiffRate); the earlier session is archived."""
    for bb in bbs:
        d = ROOT / bb
        if (d / 'latency_done.json').exists():
            k = 1
            while (d / f'latency_session{k}').exists(): k += 1
            (d / 'latency').rename(d / f'latency_session{k}'); (d / 'latency_done.json').rename(d / f'latency_done_session{k}.json')
    timing(bbs)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('stage', choices=['search', 'plan', 'worker', 'time', 'report', 'densify', 'retime', 'plan-dr-search'])
    ap.add_argument('--backbones', default=','.join(BBS))
    ap.add_argument('--gpus', default='0,1', help='search: GPUs for the probes (timing steps use the first)')
    a = ap.parse_args(argv); bbs = a.backbones.split(',')
    if a.stage == 'search':
        for bb in bbs: search(bb, a.gpus)
    elif a.stage == 'plan':
        for bb in bbs:
            p = plan(bb)
            print('PLANNED', bb, len(p['jobs']), 'arms', sum(k.startswith('leaf/') for k in p['jobs']), 'LEAF', 'sha', p['sha'][:12], flush=True)
    elif a.stage == 'worker':
        worker(bbs)
    elif a.stage == 'time':
        timing(bbs)
    elif a.stage == 'densify':
        for bb in bbs:
            p = densify(bb); print('DENSIFIED', bb, len(p['jobs']), 'points', 'sha', p['sha'][:12], flush=True)
    elif a.stage == 'retime':
        retime(bbs)
    elif a.stage == 'plan-dr-search':
        p = plan_dr_search(); print('PLANNED DiffRate search arms', len(p['jobs']), 'sha', p['sha'][:12], flush=True)
    else:
        report(bbs)


if __name__ == '__main__':
    main()
