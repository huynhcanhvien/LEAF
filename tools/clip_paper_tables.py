"""Write the CLIP ViT-B/16 tables of the paper from results/clip_comparison/clip_b16 (report.json and the sealed
plans_dr_search*.json rounds).

    python tools/clip_paper_tables.py            # -> paper/tables/clip_latency_pairs.tex, paper/tables/clip_diffrate_search.tex

clip_latency_pairs.tex: latency pairs of LEAF with every re-run family except the transferred DiffRate rates (Holm over
the pairs shown, per metric). clip_diffrate_search.tex: DiffRate searched on CLIP against LEAF planned under a GMAC
budget at the GMAC of each searched schedule, and the fastest LEAF latency-planned schedule that is faster than it.
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from leaf.evaluation.data import holm, mcnemar
from leaf.evaluation.runner import config_hash

D = REPO / 'results/clip_comparison/clip_b16'
OUT = REPO / 'paper/tables'
METRICS = ('val', 'v2', 'flickr_test_i2t', 'flickr_test_t2i', 'coco_test_i2t', 'coco_test_t2i')
NAMES = {'ToMe+M3': 'ToMe+MLERP', 'One-More-Token': 'Pruning One More Token', 'DiffRate (searched on CLIP)': 'DiffRate (searched)'}
EXCLUDE = ('DiffRate (DeiT-B rates)',)


def load(p): return json.loads(Path(p).read_text())


def pm(v): return f'${v:+.2f}$'


def pairs_table(rep):
    pairs = [dict(p) for p in rep['latency_pairs'] if p['family'] not in EXCLUDE]
    for m in METRICS:
        holm([p[m] for p in pairs if m in p])
    fams = sorted({p['family'] for p in pairs}, key=lambda f: NAMES.get(f, f).lower())
    dense = next(p for p in rep['points'] if p['family'] == 'dense')
    wl = lambda ps, m: f"{sum(p[m]['holm'] < .05 and p[m]['delta'] > 0 for p in ps)}/{sum(p[m]['holm'] < .05 and p[m]['delta'] < 0 for p in ps)}"
    rows = []
    for f in fams:
        ps = [p for p in pairs if p['family'] == f]; d = [p['val']['delta'] for p in ps]
        rng = pm(min(d)) if len(d) == 1 else f'{pm(min(d))}\\,--\\,{pm(max(d))}'
        rows.append(f"{NAMES.get(f, f)} & {len(ps)} & " + ' & '.join(wl(ps, m) for m in METRICS) + f" & {rng} & {pm(min(p['val']['ci_lo'] for p in ps))} \\\\")
    rows[-1] += ' \\midrule'
    rows.append(f"All & {len(pairs)} & " + ' & '.join(wl(pairs, m) for m in METRICS) + f" & & {pm(min(p['val']['ci_lo'] for p in pairs))} \\\\")
    n = len(pairs)
    return '\n'.join([
        '\\begin{table}[t]\\centering\\small\\setlength{\\tabcolsep}{2.5pt}',
        '\\caption{CLIP \\mbox{ViT-B/16}: latency-paired comparisons at batch~32 in one session (pairing as in \\cref{sec:setup}). '
        'Metrics: zero-shot ImageNet top-1 on val and V2 and Recall@1 on Flickr30k and MS-COCO in both directions (\\cref{sec:setup}). '
        'W/L: number of pairs in which \\ours\\ is significantly more/less accurate (McNemar on per-image or per-query correctness, '
        f'Holm over all {n}~pairs of a metric). Val difference: \\ours\\ minus the family on val top-1, in points, as the range over '
        "the family's pairs and the lowest paired 95\\% confidence bound. DiffRate (searched): its own search on CLIP (\\cref{tab:clip-diffrate-search}). Dense model: "
        f"{dense['val']:.2f}/{dense['v2']:.2f} top-1, {dense['flickr_test_i2t']:.2f}/{dense['flickr_test_t2i']:.2f} (Flickr30k) and "
        f"{dense['coco_test_i2t']:.2f}/{dense['coco_test_t2i']:.2f} (COCO) R@1.}}",
        '\\label{tab:clip-pairs}',
        '\\begin{tabular}{l r cc cc cc rr}\\toprule',
        '& & \\multicolumn{6}{c}{W/L (significant wins/losses of \\ours)} & \\multicolumn{2}{c}{Val difference} \\\\',
        '\\cmidrule(lr){3-8}\\cmidrule(lr){9-10}',
        '& & \\multicolumn{2}{c}{ImageNet top-1} & \\multicolumn{2}{c}{Flickr30k R@1} & \\multicolumn{2}{c}{COCO R@1} & & \\\\',
        '\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}',
        'Family & Pairs & Val & V2 & I$\\to$T & T$\\to$I & I$\\to$T & T$\\to$I & range & lo \\\\ \\midrule',
        *rows, '\\bottomrule\\end{tabular}\\end{table}']) + '\n', n


def correct(job, split):
    p = D / split / 'runs' / f"{config_hash(job['config'])}.npz"
    return np.load(p)['correct'].astype(bool)


def diffrate_table(rep):
    pts = {p['name']: p for p in rep['points']}
    jobs, rows, tests = {}, [], []
    for f in ('plans_dr_search.json', 'plans_dr_search2.json', 'plans_dr_search3.json'):
        if (D / f).exists(): jobs.update(load(D / f)['jobs'])
    plans = load(D / 'plans.json')['jobs']; leaf_lat = sorted((n for n in pts if pts[n]['family'] == 'LEAF'), key=lambda n: pts[n]['ms'])
    for d in sorted((n for n in jobs if n.startswith('ref/diffrate_search/')), key=lambda n: jobs[n]['gmac']):
        g = jobs[d]['gmac']
        l = min((n for n in jobs if n.startswith('leaf/gmac/')), key=lambda n: abs(jobs[n]['gmac'] - g))
        fast = [n for n in leaf_lat if pts[n]['ms'] < pts[d]['ms']]; o = max(fast, key=lambda n: pts[n]['val']) if fast else None
        row = dict(d=d, l=l, o=o, g=g)
        for m in METRICS:
            row[m] = pts[l][m] - pts[d][m]
            if m in ('val', 'v2'):
                t = dict(p=mcnemar(correct(jobs[l], m), correct(jobs[d], m))[0]); tests.append(t); row['t_' + m] = t
                if o is not None:
                    t2 = dict(p=mcnemar(correct(plans[o], m), correct(jobs[d], m))[0]); tests.append(t2); row['o_' + m] = t2
        rows.append(row)
    holm(tests)
    sig = sum(t['holm'] < .05 for t in tests)
    dense = next(p for p in rep['points'] if p['family'] == 'dense')
    cell = lambda v, dlt: f'{v:.2f}\\,{{\\footnotesize(${dlt:+.2f}$)}}'
    body = []
    for r in rows:
        L, Dr = pts[r['l']], pts[r['d']]
        body.append(f"{r['g']:.2f} & {L['ms']:.2f} & {Dr['ms']:.2f} & " + ' & '.join(cell(L[m], r[m]) for m in METRICS) + ' \\\\')
    lat = []
    for r in rows:
        if r['o'] is None: continue
        O, Dr = pts[r['o']], pts[r['d']]
        lat.append(f"{r['g']:.2f} & {O['ms']:.2f} & {Dr['ms']:.2f} & " + ' & '.join(cell(O[m], O[m] - Dr[m]) for m in METRICS) + ' \\\\')
    tex = '\n'.join([
        '\\begin{table*}[t]\\centering\\small\\setlength{\\tabcolsep}{4pt}',
        '\\caption{CLIP \\mbox{ViT-B/16}: DiffRate~\\cite{chen2023diffrate} searched on CLIP with its official code and settings '
        '(three epochs on the labelled ImageNet training set per budget, zero-shot head) against \\ours, all re-run and timed in one session. '
        'Top: \\ours\\ planned under a GMAC budget at the GMAC of each searched schedule. Bottom: the most accurate latency-planned '
        '\\ours\\ schedule that is faster than the searched one. Each cell: \\ours\'s score, with \\ours\\ minus DiffRate in parentheses '
        f"(points). {sig} of the {len(tests)} val and V2 differences are significant (McNemar, Holm over the {len(tests)} tests). "
        'GMAC: of the searched DiffRate schedule (in the bottom block the \\ours\\ schedule has fewer). ms: measured latency at batch~32.}',
        '\\label{tab:clip-diffrate-search}',
        '\\begin{tabular}{r rr rr rr rr}\\toprule',
        '& \\multicolumn{2}{c}{ms$\\downarrow$} & \\multicolumn{2}{c}{ImageNet top-1$\\uparrow$} & \\multicolumn{2}{c}{Flickr30k R@1$\\uparrow$} & \\multicolumn{2}{c}{COCO R@1$\\uparrow$} \\\\',
        '\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}',
        'GMAC & \\ours & DiffRate & Val & V2 & I$\\to$T & T$\\to$I & I$\\to$T & T$\\to$I \\\\ \\midrule',
        f"Dense & \\multicolumn{{2}}{{c}}{{{dense['ms']:.2f}}} & " + ' & '.join(f'{dense[m]:.2f}' for m in METRICS) + ' \\\\ \\midrule',
        '\\multicolumn{9}{l}{\\emph{Equal GMAC}} \\\\', *body, '\\midrule',
        '\\multicolumn{9}{l}{\\emph{Faster \\ours\\ schedule (latency-planned)}} \\\\', *lat,
        '\\bottomrule\\end{tabular}\\end{table*}']) + '\n'
    return tex, rows, sig


def main():
    rep = load(D / 'report.json')
    tex, n = pairs_table(rep); (OUT / 'clip_latency_pairs.tex').write_text(tex)
    tex2, rows, sig = diffrate_table(rep); (OUT / 'clip_diffrate_search.tex').write_text(tex2)
    print(f'clip_latency_pairs.tex: {n} pairs; clip_diffrate_search.tex: {len(rows)} budgets, {sig} significant tests')


if __name__ == '__main__':
    main()
