"""Paper v3 tables from the ablation_v3 campaign (2026-10-20): schedule-structure ablation and sensitivity on DeiT-B and
MAE ViT-B, and the grid study on all seven backbones.

python leaf/reproduction/experiments/ablation_tables.py
  -> paper/tables/ablation_structure.tex, paper/tables/grid_sensitivity.tex
     (numbers read from results/ablation_and_alpha_grid/summary.json; prints the numbers quoted in the text)
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SUMMARY = ROOT / 'results/ablation_and_alpha_grid/summary.json'
OUT = ROOT / 'paper/tables'
CORE = (('deit_b', ('extra2', 'DR10.0', 'DR12.5'), ('7.0', '10.3', '13.1')),
        ('mae_b', ('extra2', 'DR10.0', 'DR11.5'), ('6.9', '10.0', '11.8')))
NAMES = dict(deit_t='DeiT-T', deit_s='DeiT-S', deit_b='DeiT-B', augreg_s='ViT-S (AugReg)', augreg_b='ViT-B (AugReg)',
             mae_b='ViT-B (MAE)', mae_l='ViT-L (MAE)')
ROWS = [
    ('Full method', None),
    ('\\ours\\ on $\\mathcal A$ (seven values)', 'leaf/alpha7'),
    ('Schedule structure (re-planned)', None),
    ('single station', 'schedule_structure/single/PD/M3'),
    ('DiffRate schedule, our score and M3', 'schedule_structure/diffrate/PD/M3'),
    ('one $\\alpha$ for all stations', 'shared_settings/shared_alpha'),
    ('one $\\theta$ for all stations', 'shared_settings/shared_theta'),
    ('one $(\\theta,\\alpha)$ for all stations', 'shared_settings/shared_both'),
    ('$K=2$ stations', 'fixed_station_count/fixed2'), ('$K=3$', 'fixed_station_count/fixed3'), ('$K=4$', 'fixed_station_count/fixed4'), ('$K=6$', 'fixed_station_count/fixed6'), ('$K=8$', 'fixed_station_count/fixed8'),
    ('Score (re-planned)', None),
    ('$D$ only (\\mbox{$\\alpha_t=1$} wherever \\mbox{$\\alpha_t>0$})', 'forecast_only/d_only'),
    ('Setting grids (re-planned)', None),
    ('$\\{0,\\frac12\\}$', 'alpha_grid/a2'),
    ('$\\{0,\\frac12,\\frac34\\}$', 'alpha_grid/a3'),
    ('$\\mathcal A_4\\cup\\{1\\}$', 'alpha_grid/alpha_ext'),
    ('$\\mathcal A_4\\cup\\{\\frac14,\\frac13\\}$', 'alpha_grid/alpha_fine'),
    ('$\\Theta=\\{0,1\\}$', 'theta_grid/theta01'),
    ('$\\Theta=\\{0,\\frac14,\\frac12,\\frac34,1\\}$', 'theta_grid/theta5'),
    ('Forecast compression (same stations)', None),
    ('$G=4$, $r=32$', 'forecast_compression/G4r32_fixed'), ('$G=8$, $r=32$', 'forecast_compression/G8r32_fixed'), ('$G=32$, $r=32$', 'forecast_compression/G32r32_fixed'),
    ('$G=16$, $r=8$', 'forecast_compression/G16r8_fixed'), ('$G=16$, $r=16$', 'forecast_compression/G16r16_fixed'), ('$G=16$, $r=64$', 'forecast_compression/G16r64_fixed'),
    ('$G=16$, full rank', 'forecast_compression/G16rfull_fixed'),
    ('Probe sample (re-planned)', None),
    ('$|\\mathcal C|=2{,}500$, draw 1', 'probe_sample_size/n2500_s0'), ('$|\\mathcal C|=2{,}500$, draw 2', 'probe_sample_size/n2500_s1'),
    ('$|\\mathcal C|=1{,}250$, draw 1', 'probe_sample_size/n1250_s0'), ('$|\\mathcal C|=1{,}250$, draw 2', 'probe_sample_size/n1250_s1'),
]


def cell(r):
    if r is None: return '--'
    if r['same_config']: return '='
    s = f"{r['delta']:+.2f}"
    if r['holm'] < .05: s = f"\\underline{{{s}}}" if r['delta'] > 0 else f"\\textbf{{{s}}}"
    return s


def core_table(S):
    def get(bb, arm, b, sp):
        for r in S[bb]:
            if r['name'] == f'{arm}/{b}' and r['split'] == sp: return r
    lines = [r'\begin{table*}[t]\centering\footnotesize\setlength{\tabcolsep}{4pt}',
             r'\caption{Schedule-structure ablation and sensitivity of \ours\ on \mbox{DeiT-B} and \mbox{ViT-B} (MAE), columns labelled by '
             r'GMAC (variant minus reference, ImageNet~val points). The reference is the planner on the four-value grid '
             r'\mbox{$\mathcal A_4=\{0,\frac12,\frac23,\frac34\}$}, on which every variant is re-planned, so each row changes one component; '
             r'the first row is the full method on $\mathcal A$. Re-planned variants run the same planner at the same '
             r'latency budget; `same stations\' keeps the reference\'s stations, counts and settings and only changes the forecast. '
             r'\underline{Underlined}/\textbf{bold}: significantly better/worse (McNemar, Holm over all contrasts per backbone and split). '
             r'`=\': identical schedule; `--\': not run. $\bar\Delta_{\rm V2}$: mean V2 difference; $\Delta t$: mean latency change, '
             r'all arms timed in one session.}\label{tab:abl-structure}',
             r'\begin{tabular}{l rrr rrr rr r}\toprule',
             r' & \multicolumn{3}{c}{DeiT-B} & \multicolumn{3}{c}{ViT-B (MAE)} & \multicolumn{2}{c}{$\bar\Delta_{\rm V2}\uparrow$} & \\ '
             r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-9}',
             'Variant (GMAC) & ' + ' & '.join(g for _, _, gs in CORE for g in gs) + r' & \mbox{DeiT-B} & MAE-B & $\Delta t$ (\%) \\ \midrule']
    for name, arm in ROWS:
        if arm is None:
            lines.append(f'\\multicolumn{{10}}{{l}}{{\\emph{{{name}}}}} \\\\'); continue
        cells, v2m, dt = [], [], []
        for bb, bs, _ in CORE:
            vv = []
            for b in bs:
                r = get(bb, arm, b, 'val'); w = get(bb, arm, b, 'v2'); cells.append(cell(r))
                if w: vv.append(w['delta'])
                if r and r['ms'] and r['ref_ms']: dt.append(100 * (r['ms'] / r['ref_ms'] - 1))
            v2m.append(f'{sum(vv) / len(vv):+.2f}' if vv else '--')
        lines.append(f"{name} & {' & '.join(cells)} & {v2m[0]} & {v2m[1]} & {sum(dt) / len(dt):+.1f} \\\\")
    lines += [r'\bottomrule\end{tabular}\end{table*}']
    return '\n'.join(lines).replace("\\'", "'") + '\n'   # raw strings above keep \' (not a LaTeX accent)


def grid_table(S):
    groups = (('leaf', '$\\mathcal A$ with $\\frac14,\\frac13,1$'), ('grid_theta', '$\\Theta$ with $\\frac14,\\frac34$'), ('grid_theta_alpha', 'both'))
    lines = [r'\begin{table}[t]\centering\small',
             r'\caption{Finer setting grids on all seven backbones and 48~latency budgets, re-planned from the same probes '
             r'(variant minus full method, mean over budgets; ImageNet~val and V2 points, latency in \%). Changed: schedules that differ from the '
             r'full method. Holm: significant wins/losses over both splits.}\label{tab:grid-sensitivity}',
             r'\setlength{\tabcolsep}{3pt}\begin{tabular}{l rrrr rrrr rrrr}\toprule',
             ' & ' + ' & '.join(f'\\multicolumn{{4}}{{c}}{{{t}}}' for _, t in groups) + r' \\ '
             r'\cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(lr){10-13}',
             'Backbone & ' + ' & '.join(['chg. & Val & V2 & $\\Delta t$'] * 3) + r' \\ \midrule']
    tot = {g: [] for g, _ in groups}
    for bb in NAMES:
        cells = []
        for g, _ in groups:
            v = [r for r in S[bb] if r['group'] == g and r['split'] == 'val']
            w = {r['budget']: r for r in S[bb] if r['group'] == g and r['split'] == 'v2'}
            ch = sum(not r['same_config'] for r in v)
            dv = sum(r['delta'] for r in v) / len(v); d2 = sum(w[r['budget']]['delta'] for r in v) / len(v)
            dt = sum(100 * (r['ms'] / r['ref_ms'] - 1) for r in v) / len(v)
            tot[g] += [(r, w[r['budget']]) for r in v]
            cells += [f'{ch}/{len(v)}', f'{dv:+.2f}', f'{d2:+.2f}', f'{dt:+.1f}']
        lines.append(f"\\mbox{{{NAMES[bb]}}} & {' & '.join(cells)} \\\\")
    cells, holm = [], []
    for g, _ in groups:
        rs = tot[g]; n = len(rs)
        cells += [f"{sum(not a['same_config'] for a, _ in rs)}/{n}", f"{sum(a['delta'] for a, _ in rs) / n:+.2f}",
                  f"{sum(b['delta'] for _, b in rs) / n:+.2f}", f"{sum(100 * (a['ms'] / a['ref_ms'] - 1) for a, _ in rs) / n:+.1f}"]
        x = [r for p in rs for r in p if r['holm'] < .05]
        holm.append(f"{sum(r['delta'] > 0 for r in x)}/{sum(r['delta'] < 0 for r in x)}")
    lines.append(r'\midrule all & ' + ' & '.join(cells) + r' \\')
    lines.append('Holm & ' + ' & '.join(f'\\multicolumn{{4}}{{c}}{{{h}}}' for h in holm) + r' \\')
    lines += [r'\bottomrule\end{tabular}\end{table}']
    return '\n'.join(lines) + '\n'


def grid_table_leaf():
    """tab:grid-sensitivity with LEAF (A of seven values) as the full method: the four-value grid (leaf_alpha4) and
    Theta with 1/4, 3/4 (grid_theta_alpha) against LEAF, per-image McNemar, Holm over all contrasts of the table per split; latency ratio within
    the ablation session (all three timed together)."""
    import sys
    import numpy as np
    sys.path.insert(0, str(ROOT))
    import leaf.reproduction.experiments.ablation_and_alpha_grid as ab
    from leaf.evaluation.data import mcnemar, holm
    groups = (('ref/leaf_alpha4/', '$\\mathcal A_4=\\{0,\\frac12,\\frac23,\\frac34\\}$'), ('grid_theta_alpha/theta5_alpha7/', '$\\Theta$ with $\\frac14,\\frac34$'))
    recs = []
    for bb in NAMES:
        J = ab.sealed_jobs(bb)
        for name, row in J.items():
            if not name.startswith('leaf/alpha7/'): continue
            bid = name.split('/')[-1]; ref = row['config']
            for gi, (pre, _) in enumerate(groups):
                cfg = J[pre + bid]['config']; rec = dict(bb=bb, g=gi, same=ab.config_hash(cfg) == ab.config_hash(ref) if hasattr(ab, 'config_hash') else None)
                from leaf.evaluation.runner import config_hash
                rec['same'] = config_hash(cfg) == config_hash(ref)
                for s in ('val', 'v2'):
                    a, b = ab.arrays(bb, cfg, s)['correct'], ab.arrays(bb, ref, s)['correct']
                    p, _, _ = mcnemar(a, b); rec[s] = dict(delta=float((a.mean() - b.mean()) * 100), p=p)
                rec['dt'] = 100 * (ab.latency(bb, cfg) / ab.latency(bb, ref) - 1)
                recs.append(rec)
    for s in ('val', 'v2'): holm([r[s] for r in recs])
    lines = [r'\begin{table}[t]\centering\small',
             r'\caption{Setting grids on all seven backbones and 48~latency budgets: the full method ($\mathcal A$ of seven values, '
             r'$\Theta=\{0,\frac12,1\}$) against the coarser four-value $\mathcal A_4$ and against a finer $\Theta$, all re-planned from the same '
             r'probes and timed in one session (variant minus full method, mean over budgets; ImageNet~val and V2 points, latency in \%). '
             r'chg.: schedules that differ from the full method. Holm: significant wins/losses over both splits.}\label{tab:grid-sensitivity}',
             r'\setlength{\tabcolsep}{3pt}\begin{tabular}{l rrrr rrrr}\toprule',
             ' & ' + ' & '.join(f'\\multicolumn{{4}}{{c}}{{{t}}}' for _, t in groups) + r' \\ \cmidrule(lr){2-5}\cmidrule(lr){6-9}',
             'Backbone & ' + ' & '.join(['chg. & Val & V2 & $\\Delta t$'] * 2) + r' \\ \midrule']
    def cells(rs):
        n = len(rs)
        return [f"{sum(not r['same'] for r in rs)}/{n}", f"{sum(r['val']['delta'] for r in rs) / n:+.2f}",
                f"{sum(r['v2']['delta'] for r in rs) / n:+.2f}", f"{sum(r['dt'] for r in rs) / n:+.1f}"]
    for bb in NAMES:
        c = [x for gi in range(len(groups)) for x in cells([r for r in recs if r['bb'] == bb and r['g'] == gi])]
        lines.append(f"\\mbox{{{NAMES[bb]}}} & {' & '.join(c)} \\\\")
    c = [x for gi in range(len(groups)) for x in cells([r for r in recs if r['g'] == gi])]
    lines.append(r'\midrule all & ' + ' & '.join(c) + r' \\')
    hl = []
    for gi in range(len(groups)):
        x = [r[s] for r in recs if r['g'] == gi for s in ('val', 'v2') if r[s]['holm'] < .05]
        hl.append(f"{sum(d['delta'] > 0 for d in x)}/{sum(d['delta'] < 0 for d in x)}")
    lines.append('Holm & ' + ' & '.join(f'\\multicolumn{{4}}{{c}}{{{h}}}' for h in hl) + r' \\')
    lines += [r'\bottomrule\end{tabular}\end{table}']
    for gi in range(len(groups)):
        sig = [(r['bb'], s, round(r[s]['delta'], 2)) for r in recs if r['g'] == gi for s in ('val', 'v2') if r[s]['holm'] < .05]
        print('grid group', gi, 'significant', sig)
    return '\n'.join(lines) + '\n'


def text_numbers(S):
    """Numbers quoted in the ablation text (final-block station)."""
    for src, refname in (('leaf_alpha4', None), ('gmac', 'final_station_all/gmac_ref')):
        d, a = [], []
        for bb in NAMES:
            for r in S[bb]:
                if not (r['name'].startswith(f'final_station_all/{src}_no_final/') and r['split'] == 'val'): continue
                ref = r if refname is None else next(x for x in S[bb] if x['name'] == f"{refname}/{r['budget']}" and x['split'] == 'val')
                rms, rt = (r['ref_ms'], r['ref_top1']) if refname is None else (ref['ms'], ref['top1'])
                d.append(100 * (r['ms'] / rms - 1)); a.append((bb, r['top1'] - rt))
        cls = [abs(x) for bb, x in a if not bb.startswith('mae')]
        print(f"A7all {src}: n={len(d)} slower without final station {sum(d) / len(d):+.2f}% ({min(d):+.2f}..{max(d):+.2f}); "
              f"class-token max |dtop1| {max(cls):.3f}; MAE-L {[round(x, 2) for bb, x in a if bb == 'mae_l']}")


def main():
    S = json.loads(SUMMARY.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'ablation_structure.tex').write_text(core_table(S))
    (OUT / 'grid_sensitivity.tex').write_text(grid_table_leaf())   # LEAF (seven alpha values) is the full method (grid_table kept)
    print('wrote', OUT / 'ablation_structure.tex', OUT / 'grid_sensitivity.tex')
    text_numbers(S)


if __name__ == '__main__':
    main()
