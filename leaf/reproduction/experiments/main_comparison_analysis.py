"""Main comparison of the paper with the LEAF schedules (alpha grid of seven values) as the method (CPU only, no new inference).

python leaf/reproduction/experiments/main_comparison_analysis.py check   # same analysis with the sealed leaf_alpha4 schedules: must reproduce the paper-v2 numbers
python leaf/reproduction/experiments/main_comparison_analysis.py         # LEAF (alpha grid {0, 1/4, 1/3, 1/2, 2/3, 3/4, 1}) as the method

LEAF = the latency-budget schedules of results/ablation_and_alpha_grid (group leaf, arms leaf/alpha7/<budget>), planned from the same probes,
SALT and DP as leaf_alpha4 with the finer alpha grid, sealed before evaluation. Baselines, DiffRate and the M0 variant are the arms of
the main campaign (results/main_comparison), unchanged.

Latency transfer. LEAF was timed in the ablation session together with leaf_alpha4, not in the main session. Where LEAF equals leaf_alpha4 the
main-session timing is used as is; otherwise ms(LEAF) = ms_main(leaf_alpha4) x ms_abl(LEAF) / ms_abl(leaf_alpha4) (same budget), and the SD is
the main-session SD of leaf_alpha4 at the same budget, so that tau uses the noise of the session the baselines were timed in
(the ablation-session SD is kept as sd_abl). Accuracy: per-image arrays of the ablation run (val, V2; ReaL from
the val predictions as in leaf/reproduction/experiments/imagenet_real.py).

Outputs: results/main_comparison_analysis/{report.json, numbers.json}; with --tables also the paper-v3 tables
results_all_budgets.tex and latency_paired_comparison.tex (rows regenerated, headers kept) and Figures 2/6.
"""
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
import numpy as np

import leaf.reproduction.experiments.campaign_reports as q
import leaf.reproduction.experiments.ablation_and_alpha_grid as ab
from leaf.evaluation.data import mcnemar, holm
from leaf.evaluation.runner import config_hash

OUT = REPO / 'results/main_comparison_analysis'
V3 = REPO / 'paper'
M0 = 'LEAF-M0'
_REAL = {}


def real_correct(pred):
    if not _REAL:
        from leaf.reproduction.experiments.imagenet_real import real_sets
        _REAL['sets'], _REAL['mask'] = real_sets()
    return np.array([int(p) in s for p, s in zip(pred, _REAL['sets'])])[_REAL['mask']]


def abl_rounds(bb, cfg):
    return [q.load(p)['measurement']['ms_mean'] for p in sorted((ab.ROOT / bb / 'latency').glob(f'{bb}-{config_hash(cfg)}#*.json'))]


def arms(bb, use_leaf):
    """All arms of a backbone: baselines from the main campaign, ours = leaf_alpha4 (check) or LEAF."""
    jobs = q.load(q.MAIN_COMPARISON / bb / 'plans.json')['jobs']; ms = q.round_means(q.MAIN_COMPARISON / bb / 'latency', 32)
    base = [dict(a, src='main') for a in q.main_arms(bb) if a['family'] != 'LEAF']
    sealed = ab.sealed_jobs(bb); ours = []
    for name, row in jobs.items():
        if not name.startswith('leaf_alpha4/') or 'config' not in row: continue
        bid = name.split('/')[1]; tid = f"{bb}-{row['config_sha']}"
        main = dict(name=name, family='LEAF', sha=row['config_sha'], ms=ms[tid][0], sd=ms[tid][1], gmac=row['gmac'],
                    budget=row['budget_ms'], budget_id=bid, K=row['details']['K'], src='main', same=True)
        for s in ('val', 'v2'): main[s] = float(q.correct(q.MAIN_COMPARISON / bb, row['config_sha'], s).mean() * 100)
        if not use_leaf: ours.append(main); continue
        r1, ref = sealed[f'leaf/alpha7/{bid}'], sealed[f'ref/leaf_alpha4/{bid}']
        if config_hash(r1['config']) == config_hash(ref['config']): ours.append(dict(main, name=f'leaf/{bid}')); continue
        rr, lr = abl_rounds(bb, r1['config']), abl_rounds(bb, ref['config'])
        f = main['ms'] / np.mean(lr)
        o = dict(name=f'leaf/{bid}', family='LEAF', sha=config_hash(r1['config']), cfg=r1['config'], ms=float(np.mean(rr) * f),
                 sd=main['sd'], sd_abl=float(np.std(rr, ddof=1) * f), gmac=r1['gmac'], budget=r1['budget_ms'], budget_id=bid,
                 K=len(r1['state']['sites']), src='abl', same=False, transfer=float(np.mean(rr) / np.mean(lr)))
        for s in ('val', 'v2'): o[s] = float(ab.arrays(bb, r1['config'], s)['correct'].mean() * 100)
        ours.append(o)
    return base + ours


def correct_of(bb, a, split):
    if a['src'] == 'main' or a.get('same'):
        if split == 'real': return real_correct(np.load(q.MAIN_COMPARISON / bb / 'val' / 'runs' / f"{a['sha']}.npz")['pred'])
        return q.correct(q.MAIN_COMPARISON / bb, a['sha'], split)
    if split == 'real': return real_correct(ab.arrays(bb, a['cfg'], 'val')['pred'])
    return ab.arrays(bb, a['cfg'], split)['correct']


def paired_ci(a, b, z=1.96):
    d = a.astype(float) - b.astype(float); m, se = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
    return dict(delta=float(m * 100), lo=float((m - z * se) * 100), hi=float((m + z * se) * 100))


def analyse(use_leaf):
    rows, tests = {}, []
    for bb in q.BBS:
        A = arms(bb, use_leaf); rows[bb] = A
        for o in [a for a in A if a['family'] == 'LEAF']:
            for fam in sorted({a['family'] for a in A} - {'LEAF'}):
                best = min((a for a in A if a['family'] == fam), key=lambda a: abs(a['ms'] - o['ms']))
                t = q.tau((o['ms'], o['sd']), (best['ms'], best['sd']))
                if abs(best['ms'] - o['ms']) > t: continue
                rec = dict(backbone=bb, ours=o['name'], family=fam, ref=best['name'], ours_ms=o['ms'], ref_ms=best['ms'], tau=t)
                for s in ('val', 'v2', 'real'):
                    ca, cb = correct_of(bb, o, s), correct_of(bb, best, s); p, _, _ = mcnemar(ca, cb)
                    rec[s] = dict(delta=float((ca.mean() - cb.mean()) * 100), p=p, **{f'ci_{k}': v for k, v in paired_ci(ca, cb).items()})
                tests.append(rec)
    for s in ('val', 'v2', 'real'): holm([t[s] for t in tests])
    # released DiffRate schedule of the same budget (results_all_budgets)
    anchors = []
    for bb in q.BBS:
        A = rows[bb]; byname = {a['name']: a for a in A}
        for o in [a for a in A if a['family'] == 'LEAF']:
            r = dict(bb=bb, o=o); dr = byname.get(f"ref/history/baseline/{o['budget_id']}")
            if dr:
                r['dr'] = dr
                for s in ('val', 'v2'):
                    p, _, _ = mcnemar(correct_of(bb, o, s), correct_of(bb, dr, s)); r[f'p_{s}'] = dict(p=p)
                r['faster'] = dr['ms'] - o['ms'] >= q.tau((o['ms'], o['sd']), (dr['ms'], dr['sd']))
            anchors.append(r)
    for s in ('val', 'v2'): holm([r[f'p_{s}'] for r in anchors if f'p_{s}' in r])
    return rows, tests, anchors


def frontier_stats(rows, split):
    """Our schedules within DiffRate's latency range vs DiffRate's Pareto frontier interpolated at the same latency."""
    out = dict(n=0, above=0, deltas=[], below=[], dr_n=0, dr_below_ours=0, faster_than_all=0)
    for bb in q.BBS:
        A = rows[bb]; ours = [a for a in A if a['family'] == 'LEAF']; dr = [a for a in A if a['family'] == 'DiffRate']
        if not dr: continue
        def frontier(pts):
            pts = [(a['ms'], a[split]) for a in pts]; f = sorted(pts[i] for i in q.pareto(pts))
            return [p[0] for p in f], [p[1] for p in f]
        dx, dy = frontier(dr); ox, oy = frontier(ours)
        for o in ours:
            if o['ms'] < min(a['ms'] for a in dr): out['faster_than_all'] += 1; continue
            if o['ms'] > max(a['ms'] for a in dr): continue
            d = o[split] - float(np.interp(o['ms'], dx, dy)); out['n'] += 1; out['deltas'].append(d)
            if d > 0: out['above'] += 1
            else: out['below'].append((bb, round(o['ms'], 1), round(d, 2)))
        for a in dr:
            if min(ox) <= a['ms'] <= max(ox):
                out['dr_n'] += 1; out['dr_below_ours'] += a[split] < float(np.interp(a['ms'], ox, oy))
    d = out.pop('deltas'); out.update(mean=float(np.mean(d)), lo=float(min(d)), hi=float(max(d)))
    return out


def numbers(rows, tests, anchors):
    T = [t for t in tests if t['family'] != M0]; fam = {}
    for t in T:
        s = fam.setdefault(t['family'], dict(n=0, bbs=set(), **{f'{k}_{x}': 0 for k in ('val', 'v2', 'real') for x in ('w', 'l')},
                                              **{f'd_{k}': 0. for k in ('val', 'v2', 'real')}))
        s['n'] += 1; s['bbs'].add(t['backbone'])
        for k in ('val', 'v2', 'real'):
            s[f'd_{k}'] += t[k]['delta']
            if t[k]['holm'] < .05: s[f"{k}_{'w' if t[k]['delta'] > 0 else 'l'}"] += 1
    for s in fam.values():
        s['bbs'] = len(s['bbs'])
        for k in ('val', 'v2', 'real'): s[f'd_{k}'] /= s['n']
    lows = {k: sorted(T, key=lambda t: t[k]['ci_lo'])[0] for k in ('val', 'v2')}
    ci = dict(val_min_lo=lows['val']['val']['ci_lo'], val_min_pair=(lows['val']['backbone'], round(lows['val']['ours_ms'], 1), lows['val']['family']),
              val_n_above_m013=sum(t['val']['ci_lo'] > -.13 for t in T),
              v2_n_above_m05=sum(t['v2']['ci_lo'] > -.5 for t in T), v2_min_lo=lows['v2']['v2']['ci_lo'],
              v2_min_pair=(lows['v2']['backbone'], round(lows['v2']['ours_ms'], 1), lows['v2']['family']))
    A = [r for r in anchors if 'dr' in r]
    dom = {s: sum(r[f'p_{s}']['holm'] < .05 and r['o'][s] > r['dr'][s] and r['faster'] for r in A) for s in ('val', 'v2')}
    worse = {s: sum(r[f'p_{s}']['holm'] < .05 and r['o'][s] < r['dr'][s] for r in A) for s in ('val', 'v2')}
    deitb = [dict(ms=round(o['ms'], 2), gmac=round(o['gmac'], 1), val=round(o['val'], 2), v2=round(o['v2'], 2))
             for o in sorted((a for a in rows['deit_b'] if a['family'] == 'LEAF'), key=lambda a: a['ms'])]
    deitb_dr = [dict(ms=round(a['ms'], 2), gmac=a.get('gmac'), val=round(a['val'], 2), v2=round(a['v2'], 2))
                for a in sorted((a for a in rows['deit_b'] if a['family'] == 'DiffRate'), key=lambda a: a['ms'])]
    dr_pairs_deitb = [dict(ours_ms=round(t['ours_ms'], 2), ref_ms=round(t['ref_ms'], 2), dval=round(t['val']['delta'], 2), dv2=round(t['v2']['delta'], 2))
                      for t in T if t['backbone'] == 'deit_b' and t['family'] == 'DiffRate']
    n_any_loss = {k: sum(t[k]['holm'] < .05 and t[k]['delta'] < 0 for t in T) for k in ('val', 'v2', 'real')}
    return dict(n_pairs=len(T), n_all=len(tests), families=fam, ci=ci, released_dr=dict(n=len(A), dominate=dom, worse=worse),
                frontier=dict(val=frontier_stats(rows, 'val'), v2=frontier_stats(rows, 'v2')), any_loss=n_any_loss,
                deit_b=deitb, deit_b_diffrate=deitb_dr, deit_b_dr_pairs=dr_pairs_deitb,
                changed=sum(not a['same'] for bb in rows for a in rows[bb] if a['family'] == 'LEAF'),
                transfer=[a['transfer'] for bb in rows for a in rows[bb] if a.get('transfer')])


def results_table(anchors):
    """Rows of tables/results_all_budgets.tex (header and footer of the current v3 file are kept)."""
    order = lambda r: (q.BBS.index(r['bb']), r['o']['ms']); fmt = lambda x: f'{x:.2f}'; body = []; cur = None
    from leaf.reproduction.report.tables import dense_ref, all_budgets_group
    for r in sorted(anchors, key=order):
        o = r['o']
        if r['bb'] != cur:
            body += all_budgets_group(r['bb'], dense_ref(r['bb']), first=cur is None); cur = r['bb']
        mark = {}
        for s in ('val', 'v2'):
            mark[s] = ''
            if f'p_{s}' in r and r[f'p_{s}']['holm'] < .05:
                mark[s] = r'$^{\dagger}$' if (o[s] > r['dr'][s] and r['faster']) else (r'$^{\ddagger}$' if o[s] < r['dr'][s] else '')
        dr = r.get('dr')
        drc = f"{fmt(dr['gmac']) if dr.get('gmac') is not None else '--'} & {fmt(dr['ms'])} & {fmt(dr['val'])} & {fmt(dr['v2'])}" if dr else '-- & -- & -- & --'
        body.append(f"{drc} & {fmt(o['budget'])} & {o['K']} &{fmt(o['gmac'])} & {fmt(o['ms'])} & {fmt(o['val'])}{mark['val']} & {fmt(o['v2'])}{mark['v2']} \\\\")
    return body


def pairs_rows(nb):
    order = ['DiffRate', 'PPT', 'ToFu', 'RAPID', 'ToMe', 'ToMe+M3', 'One-More-Token', 'EViT', 'PiToMe']; F = nb['families']
    return [f"{f} & {F[f]['n']} & {F[f]['bbs']} & {F[f]['val_w']}/{F[f]['val_l']} & {F[f]['d_val']:+.2f} & {F[f]['real_w']}/{F[f]['real_l']} & "
            f"{F[f]['d_real']:+.2f} & {F[f]['v2_w']}/{F[f]['v2_l']} & {F[f]['d_v2']:+.2f} \\\\" for f in order if f in F]


def splice(path, start_marker, end_marker, body):
    s = path.read_text(); i = s.index(start_marker) + len(start_marker); j = s.index(end_marker, i)
    return s[:i] + '\n' + '\n'.join(body) + '\n' + s[j:]


def write_tables():
    """Splice the rows of the paper's method into paper/tables/results_all_budgets.tex and latency_paired_comparison.tex
    (their headers come from leaf.reproduction.report.tables)."""
    import re
    rows, tests, anchors = analyse(True); nb = numbers(rows, tests, anchors)
    p = V3 / 'tables/results_all_budgets.tex'
    p.write_text(splice(p, r'\midrule\endhead', r'\bottomrule', results_table(anchors)))
    p = V3 / 'tables/latency_paired_comparison.tex'; s = splice(p, r'\midrule', r'\bottomrule', pairs_rows(nb))
    s = re.sub(r'\(\d+~pairs, seven backbones\)', f"({nb['n_pairs']}~pairs, seven backbones)", s)
    s = re.sub(r'over all \d+~pairs of a split', f"over all {nb['n_all']}~pairs of a split", s)
    p.write_text(s); print('wrote', p)


def write_figures():
    """Accuracy-latency frontiers of every family, ImageNet val and V2 (paper Figures 2 and S-V2)."""
    from leaf.reproduction.experiments.accuracy_latency_figures import frontier_figure_single_large
    rows, tests, anchors = analyse(True)
    dense = {}
    for bb in rows:
        sha = q.load(q.MAIN_COMPARISON / bb / 'plans.json')['jobs']['ref/dense']['config_sha']
        dense[bb] = dict(ms=q.round_means(q.MAIN_COMPARISON / bb / 'latency', 32)[f'{bb}-{sha}'][0],
                         **{sp: float(q.correct(q.MAIN_COMPARISON / bb, sha, sp).mean() * 100) for sp in ('val', 'v2')})
    for split in ('val', 'v2'): frontier_figure_single_large(rows, split, 'main_frontier_leaf_large', dense)
    for split in ('val', 'v2'): frontier_figure_single_large(rows, split, 'main_frontier_leaf_3col', dense, cols=3)   # paper layout


def main():
    use_leaf = 'check' not in sys.argv
    rows, tests, anchors = analyse(use_leaf); nb = numbers(rows, tests, anchors)
    tag = 'leaf' if use_leaf else 'leaf_alpha4_check'
    OUT.mkdir(parents=True, exist_ok=True)
    q.save(OUT / f'numbers_{tag}.json', nb)
    q.save(OUT / f'report_{tag}.json', dict(arms={bb: [{k: v for k, v in a.items() if k != 'cfg'} for a in A] for bb, A in rows.items()}, tests=tests))
    print(json.dumps({k: v for k, v in nb.items() if k not in ('transfer',)}, indent=1, default=str))
    if '--tables' in sys.argv and use_leaf:
        write_tables(); write_figures()


if __name__ == '__main__':
    main()
