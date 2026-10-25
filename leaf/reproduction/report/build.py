"""Build the paper's tables and figures from the stored results:  python -m leaf report tables | figures | all

Every generated table and figure is written to paper/tables/<name>.tex and paper/figures/<name>.pdf. Inputs are the
raw results under results/ (and the analysis outputs the experiment modules derive from them, `leaf report analyze`).

Tables (generator -> file):
  latency_paired_comparison, results_all_budgets   leaf.reproduction.report.tables (header) + main_comparison_analysis (rows)
  equal_gmac_methods, equal_gmac_pairs, equal_gmac_vs_diffrate, latency_pricing, leaf_vs_m0, published_methods,
  ablation, search_cost, negative_variants          leaf.reproduction.report.tables
  ablation_structure, grid_sensitivity               leaf.reproduction.experiments.ablation_tables
  forecast_quality                                   forecast_quality_table() below (LEAF-M0 campaign results)
  forecast_lowrank_approx                            leaf.reproduction.experiments.forecast_rank.write_table
  m0_vs_diffrate                                     leaf.reproduction.report.leaf_m0_tables (LEAF-M0 campaign data)
Hand-written: figures/method_overview.tex (TikZ); the notation table and Tables S4-S6 text in main.tex / supp.tex.
Figures (generator -> file):
  accuracy_latency_imagenet_{val,v2}                 leaf.reproduction.experiments.main_comparison_analysis --tables
  equal_gmac_latency_residual_vs_stations, salt_vs_gmac_planning, m0_latency_residual_vs_stations
                                                     leaf.reproduction.experiments.leaf_m0_figures
  token_schedule_deit_b_10gmac                       leaf.reproduction.experiments.token_schedule (needs data/cache val)
"""
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[3]
PAPER = REPO / 'paper'
BUILD = REPO / 'results/paper_build'
FIG_WORK = REPO / 'results/figures'

# tables written by leaf.reproduction.report.tables (results/paper_build/tables) and copied to paper/tables
TABLES = ('ablation.tex', 'search_cost.tex', 'equal_gmac_pairs.tex', 'equal_gmac_methods.tex', 'equal_gmac_vs_diffrate.tex',
          'leaf_vs_m0.tex', 'results_all_budgets.tex', 'negative_variants.tex', 'latency_paired_comparison.tex',
          'latency_pricing.tex', 'published_methods.tex')
# LaTeX labels: names used by the earlier paper build -> paper names (main.tex / supp.tex and hand-written tables)
LABELS = {
    'fig:v2-tokens': 'fig:token-schedule', 'fig:v2-frontier-v2': 'fig:frontier-imagenet-v2', 'fig:v2-frontier': 'fig:frontier-val',
    'fig:v2-eqg-residual': 'fig:eqgmac-residual', 'fig:v2-pricing': 'fig:salt-vs-gmac',
    'tab:v2-pairs': 'tab:latency-pairs', 'tab:v2-main': 'tab:all-budgets', 'tab:v2-eqg-main': 'tab:eqgmac-methods',
    'tab:v2-eqg-dr': 'tab:eqgmac-diffrate', 'tab:v2-eqg': 'tab:eqgmac-pairs', 'tab:v2-pricing': 'tab:latency-pricing',
    'tab:v2-ablation': 'tab:ablation', 'tab:v2-cost': 'tab:search-cost', 'tab:v2-m0': 'tab:leaf-vs-m0',
    'tab:v2-negative': 'tab:negative-variants', 'tab:main-dr': 'tab:m0-vs-diffrate', 'fig:residual': 'fig:m0-residual',
    'tab:latmodel': 'tab:m0-salt-profile',
}
# figure work files -> names in the paper
FIGURE_NAMES = {
    'main_frontier_leaf_3col_val.pdf': 'accuracy_latency_imagenet_val.pdf',
    'main_frontier_leaf_3col_v2.pdf': 'accuracy_latency_imagenet_v2.pdf',
    'equal_gmac_residual_K_large.pdf': 'equal_gmac_latency_residual_vs_stations.pdf',
    'pricing_salt_vs_gmac_large.pdf': 'salt_vs_gmac_planning.pdf',
    'm0_latency_residual_K_large.pdf': 'm0_latency_residual_vs_stations.pdf',
    'tokens_deit_b_ref_M3_gmac_DR10.0.pdf': 'token_schedule_deit_b_10gmac.pdf',
}


def _rename_labels(text):
    for old, new in sorted(LABELS.items(), key=lambda kv: -len(kv[0])):
        text = re.sub(r'(?<=[{,])' + re.escape(old) + r'(?=[},])', new, text)
    return text


# ------------------------------------------------------------------------------------------------ tables
def forecast_quality_table(root=REPO / 'results/leaf_m0_forecast_quality'):
    """Spearman of P, D (G16/r32) and the combined score (alpha = 1/2) with the true future attention, blocks 1-10."""
    cols = (('deit_s', 'DeiT-S'), ('deit_b', 'DeiT-B'), ('mae_b', 'ViT-B (MAE)'))
    data = {bb: json.loads((root / bb / 'results.json').read_text())['blocks'] for bb, _ in cols}
    lines = [r'\begin{table}[h]\centering\small\setlength{\tabcolsep}{4pt}',
             r'\caption{Forecast quality: Spearman correlation between a token score and the true future class-token attention $\Phi_j$ (\cref{eq:target}) on the patch tokens of the dense model, averaged over 1,024 held-out training images (95\% bootstrap intervals are below $\pm0.02$); higher is better. $P$: present attention; $D$: deployed forecast (\mbox{$G{=}16$}, \mbox{$r{=}32$}); $S$: combined score with \mbox{$\alpha{=}\frac12$}.}',
             r'\label{tab:forecast-quality}',
             r'\begin{tabular}{l ccc ccc ccc}\toprule',
             '& ' + ' & '.join(rf'\multicolumn{{3}}{{c}}{{{name}}}' for _, name in cols) + r'\\ \cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}',
             r'Block & $P$ & $D$ & $S$ & $P$ & $D$ & $S$ & $P$ & $D$ & $S$\\\midrule']
    for l in range(1, 11):
        cells = []
        for bb, _ in cols:
            # k = 1 <=> alpha = 1/2; values as reported in the campaign's table.md (4 decimals), then to 2 decimals
            v = [float(f"{data[bb][f'{l}/{s}/spearman']['mean']:.4f}") for s in ('P', 'D_G16_r32', 'PD_G16_r32_k1')]
            best = max(range(3), key=lambda i: v[i])
            cells += [rf'\textbf{{{x:.2f}}}' if i == best else f'{x:.2f}' for i, x in enumerate(v)]
        lines.append(f'{l} & ' + ' & '.join(cells) + r'\\')
    lines += [r'\bottomrule', r'\end{tabular}\end{table}']
    return '\n'.join(lines) + '\n'


def m0_vs_diffrate_table():
    """LEAF-M0 versus DiffRate (Supplementary): the earlier campaign's table relabelled for the appendix."""
    from leaf.reproduction.report.leaf_m0_tables import main as leaf_m0_tables
    from leaf.reproduction.report.typesetting import arrow_audit, fix_breaks, metric_arrows
    out = BUILD / 'tables_leaf_m0'
    leaf_m0_tables(str(REPO / 'results/leaf_m0_summaries/paper_final_data.json'), str(REPO / 'results/leaf_m0_summaries/reports'), str(out))
    t = ((out / 'main_dr.tex').read_text()
         .replace('Ours versus DiffRate', '\\ours-M0 (earlier variant) versus DiffRate').replace('Ours, Lat.', '\\ours-M0, Lat.')
         .replace('Model & Anchor &', 'Model & &').replace('& extra1 &', '& &').replace('& extra2 &', '& &'))
    p = out / 'main_dr_m0.tex'; p.write_text(fix_breaks(metric_arrows(t)))
    arrow_audit(out)
    return re.sub(r'(?m)^& [0-9]+\.[0-9] & ', '& & ', p.read_text())


def tables():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import leaf.reproduction.report.tables as T
    from leaf.reproduction.report.typesetting import arrow_audit
    from leaf.reproduction.experiments import ablation_tables, forecast_rank, main_comparison_analysis
    from leaf.reproduction.report.typesetting import ADD
    T.anchors(by_gmac=True); T.pairs_single(); T.pairs_single_real(); T.m0_variant(by_gmac=True)
    T.published_single(positive_drop=True, with_dense=True); T.equal_gmac_summary(); T.pricing(); T.cost()
    T.ablation(by_gmac=True, dense=True); T.negative(); T.equal_gmac_main_stacked(bold='best', with_dense=True); T.equal_gmac_dr(with_dense=True)
    arrow_audit(T.OUT)
    (PAPER / 'tables').mkdir(parents=True, exist_ok=True)
    for name in TABLES: shutil.copyfile(T.OUT / name, PAPER / 'tables' / name)
    main_comparison_analysis.write_tables()                     # rows of the paper's method (spliced into the two tables)
    ablation_tables.main()                                      # ablation_structure.tex, grid_sensitivity.tex
    (PAPER / 'tables/forecast_quality.tex').write_text(forecast_quality_table())
    rank = json.loads((REPO / 'results/forecast_rank/audit.json').read_text())
    forecast_rank.write_table(rank, PAPER / 'tables/forecast_lowrank_approx.tex')
    t = (PAPER / 'tables/forecast_lowrank_approx.tex').read_text()
    for a, b in ADD['forecast_rank.tex']:
        if b not in t: t = t.replace(a, b)
    (PAPER / 'tables/forecast_lowrank_approx.tex').write_text(t)
    (PAPER / 'tables/m0_vs_diffrate.tex').write_text(m0_vs_diffrate_table())
    finalize_tables()


def supplementary_refs():
    """\\cref to a Supplementary label from the main document -> fixed text (the two documents compile separately)."""
    aux = PAPER / 'supp.aux'
    if not aux.exists(): return {}
    kinds = {'table': 'Table', 'section': 'Section', 'figure': 'Figure', 'equation': 'Equation'}
    out = {}
    for m in re.finditer(r'^\\newlabel\{([^}@]+)\}\{\{([^}]*)\}.*\{(\w+)\.\d+\}\{\}\}$', aux.read_text(), re.M):
        out[m.group(1)] = f'Supplementary {kinds[m.group(3)]}~{m.group(2)}'
    return out


def finalize_tables():
    refs = supplementary_refs() or {'tab:leaf-vs-m0': 'Supplementary Table~S4'}
    for p in sorted((PAPER / 'tables').glob('*.tex')):
        t = _rename_labels(p.read_text())
        if p.name == 'latency_paired_comparison.tex':           # read from main.tex: inline the Supplementary reference
            t = t.replace(r'\cref{tab:leaf-vs-m0}', refs['tab:leaf-vs-m0'])
        p.write_text(t)


# ------------------------------------------------------------------------------------------------ figures
def figures(tokens=True):
    from leaf.reproduction.experiments import main_comparison_analysis, leaf_m0_figures
    main_comparison_analysis.write_figures()
    leaf_m0_figures.main([str(REPO / 'results/leaf_m0_summaries/second_campaign_data.json')])
    if tokens:
        (BUILD / 'data').mkdir(parents=True, exist_ok=True); (BUILD / 'figures').mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, str(REPO / 'leaf/reproduction/experiments/token_schedule.py'), '--backbone', 'deit_b',
                        '--schedule', 'ref/M3/gmac/DR10.0', '--examples', 'object10', '--style', 'tiles_black', '--render-only'],
                       cwd=REPO, check=True)
    (PAPER / 'figures').mkdir(parents=True, exist_ok=True)
    for old, new in FIGURE_NAMES.items():
        src = FIG_WORK / old
        if src.exists(): shutil.copyfile(src, PAPER / 'figures' / new)
        else: print('missing figure', src)


def documents():
    """Rename the labels in main.tex / supp.tex (idempotent)."""
    for name in ('main.tex', 'supp.tex'):
        p = PAPER / name; p.write_text(_rename_labels(p.read_text()))


def main(argv=None):
    what = (argv or sys.argv[1:] or ['all'])[0]
    if what in ('tables', 'all'): tables()
    if what in ('figures', 'all'): figures()
    documents()


if __name__ == '__main__':
    main()
