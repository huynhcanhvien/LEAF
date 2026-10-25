"""Write paper/tables/search_baselines.tex from results/search_baselines (leaf/reproduction/experiments/search_baselines.py).

    python tools/search_paper_table.py

Rows: the six budgets of the surrogate check (DeiT-B, ViT-B (MAE)). Columns: the measured objective (LEAF on the
four-value alpha grid against the regularised-evolution schedule with the lowest calibration KL, val and V2 top-1,
McNemar with Holm over the 12 tests), the best of the 30 random schedules of the surrogate check (val), and the
surrogate (median over five seeds of the best F-hat found divided by the DP's F-hat).
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from leaf.evaluation.data import holm

S = REPO / 'results/search_baselines'
NAMES = {'deit_b': '\\mbox{DeiT-B}', 'mae_b': '\\mbox{ViT-B} (MAE)'}


def load(p): return json.loads(Path(p).read_text())


def stations(cfg, n=197):
    k = 0
    for p, m in zip(cfg['prune_keep'], cfg['merge_keep']):
        if min(n, p, m) < n: k += 1; n = min(n, p, m)
    return k


def main():
    rep, sealed, sur = load(S / 'report.json'), load(S / 'sealed.json')['jobs'], load(S / 'surrogate.json')
    lat = load(S / 'latency.json')['latency']
    r30 = {(r['backbone'], r['budget']): r for r in load(S / 'random30_vs_leaf.json')['rows']}
    rows, tests = [], []
    for key in sur:
        bb, b = key.split('/')
        ref = load(REPO / f'results/ablation_and_alpha_grid/{bb}/plans/reference.json')['jobs'][f'ref/leaf_alpha4/{b}']
        L = ref['budget']['latency_ms'] if isinstance(ref['budget'], dict) else ref['budget']
        re_, lf = rep[f'{bb}/RE/{b}'], rep[f'{bb}/LEAF/{b}']
        row = dict(ms=(lat[f'{bb}/LEAF/{b}']['ms'], lat[f'{bb}/RE/{b}']['ms']), bb=bb, L=L, k=(stations(sealed[f'{bb}/LEAF/{b}']['config']), stations(sealed[f'{bb}/RE/{b}']['config'])), lf=lf, re=re_,
                   rand=r30[(bb, b)]['random_top1_best'])
        med = lambda m, cp: float(np.median([r['best'][cp] for r in sur[key]['runs'][m]])) / sur[key]['dp']['fhat']
        row['sur'] = (med('RE', '1000'), med('RE', '10000'), med('RS', '10000'))
        for s in ('val', 'v2'):
            t = dict(p=re_[f'{s}_vs_leaf']['p']); row['t_' + s] = t; tests.append(t)
        rows.append(row)
    holm(tests)
    cell = lambda r, s: (f"{r['re'][s]['top1']:.2f}\\,{{\\footnotesize(${r['re'][s]['top1'] - r['lf'][s]['top1']:+.2f}$)}}"
                         + ('$^{*}$' if r['t_' + s]['holm'] < .05 else ''))
    body = [f"{NAMES[r['bb']]} & {r['L']:.1f} & {r['ms'][0]:.2f} & {r['ms'][1]:.2f} & {r['k'][0]}/{r['k'][1]} & {r['lf']['half']['kl']:.4f} & {r['re']['half']['kl']:.4f} & "
            f"{r['lf']['val']['top1']:.2f} & {cell(r, 'val')} & {r['lf']['v2']['top1']:.2f} & {cell(r, 'v2')} & {r['rand']:.2f} & "
            f"{r['sur'][0]:.2f} & {r['sur'][1]:.2f} & {r['sur'][2]:.2f} \\\\" for r in rows]
    tex = '\n'.join([
        '\\begin{table*}[t]\\centering\\small\\setlength{\\tabcolsep}{3pt}',
        '\\caption{Black-box search over station layouts against the planner (\\cref{app:search}), at the six budgets of the surrogate check, '
        'with the probes, \\cost\\ model and grid $\\mathcal A_4$ of that check; \\ours\\ is the DP schedule of the same inputs. $L$: latency '
        'budget (batch~32), met by every schedule according to \\cost; ms: measured latency at batch~32 (one session, three interleaved rounds on one GPU with the other idle). $K$: stations of \\ours/RE. $F$: calibration KL. '
        'RE: regularised evolution on the measured $F$, 400 evaluations per budget; its schedule with the lowest $F$ is sealed and evaluated, '
        'with RE minus \\ours\\ in parentheses ($^{*}$: significant, McNemar, Holm over the 12 tests). Random: best val top-1 of the 30 random '
        'schedules of \\cref{app:evidence}. Surrogate: best $\\hat F$ found divided by the DP\'s $\\hat F$ (median over five seeds) after '
        '$10^3$ and $10^4$ evaluations of $\\hat F$ (RE) and $10^4$ (random search, RS).}',
        '\\label{tab:search-baselines}',
        '\\fitbox{\\begin{tabular}{l r rr c rr rr rr r rrr}\\toprule',
        '& & \\multicolumn{2}{c}{ms$\\downarrow$} & & \\multicolumn{2}{c}{$F\\downarrow$} & \\multicolumn{2}{c}{Val top-1$\\uparrow$} & \\multicolumn{2}{c}{V2 top-1$\\uparrow$} & Random & '
        '\\multicolumn{3}{c}{Surrogate: $\\hat F/\\hat F_{\\mathrm{DP}}$} \\\\',
        '\\cmidrule(lr){3-4}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}\\cmidrule(lr){10-11}\\cmidrule(lr){12-12}\\cmidrule(lr){13-15}',
        'Backbone & $L$ (ms) & \\ours & RE & $K$ & \\ours & RE & \\ours & RE & \\ours & RE & val & RE $10^3$ & RE $10^4$ & RS $10^4$ \\\\ \\midrule',
        *body, '\\bottomrule\\end{tabular}}\\end{table*}']) + '\n'
    (REPO / 'paper/tables/search_baselines.tex').write_text(tex)
    print(f'search_baselines.tex: {len(rows)} budgets, {sum(t["holm"] < .05 for t in tests)} significant of {len(tests)}')


if __name__ == '__main__':
    main()
