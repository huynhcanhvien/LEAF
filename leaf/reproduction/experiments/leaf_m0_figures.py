"""Paper v3 figures other than Figure 1 with larger titles, ticks, axis labels and legends (2026-10-20, user request).

python leaf/reproduction/experiments/leaf_m0_figures.py
  -> results/figures/equal_gmac_residual_K_large.{png,pdf}   (main Figure: latency residual vs K, 3 per row)
     results/figures/pricing_salt_vs_gmac_large.{png,pdf}         (main Figure: SALT- vs GMAC-planned DP)
python leaf/reproduction/experiments/leaf_m0_figures.py <bad.json>   (also the supplementary M0 residual figure; bad.json from
  artifacts/paper/v1/tools/collect2.py run in the Codex worktree, CPU only) -> m0_latency_residual_K_large.{png,pdf}

Same data and drawing as equal_gmac.residual_figure and campaign_reports.p1; only layout, font and marker sizes
change. Frontier figures (2 and 6) are drawn by leaf/reproduction/experiments/main_comparison_analysis.py --tables (main_frontier_r1_3col_*).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np   # noqa: E402
import leaf.reproduction.experiments.campaign_reports as q   # noqa: E402

TITLE_FS, LABEL_FS, TICK_FS, LEGEND_FS = 15, 13.5, 12, 18


def residual_figure_large(cols=3):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    import leaf.reproduction.experiments.equal_gmac as eg
    fits = q.load(eg.OUT / 'report.json')['station_fit']
    allrows = {bb: eg.arms(bb) for bb in q.BBS}
    colors = dict(q.COLORS, **{'LEAF-GMAC': q.COLORS['LEAF']}); markers = dict(q.MARKERS, **{'LEAF-GMAC': 'o'})
    label = {'LEAF': f'{q.PAPER_NAME} (latency budget)', 'LEAF-GMAC': f'{q.PAPER_NAME} (GMAC budget)', 'ToMe+M3': 'ToMe+MLERP'}
    bbs = [b for b in q.BBS if b in allrows]; n = len(bbs); r = (n + 1 + cols - 1) // cols
    fig, axes = plt.subplots(r, cols, figsize=(5.6 * cols, 4.4 * r), squeeze=False)
    rng = np.random.default_rng(0)
    for ax, bb in zip(axes.flat, bbs):
        f = fits[bb]
        for fam in ('ToMe', 'ToMe+M3', 'ToFu', 'RAPID', 'PiToMe', 'EViT', 'PPT', 'One-More-Token', 'DiffRate', 'LEAF', 'LEAF-GMAC'):
            pts = [x for x in allrows[bb] if x['family'] == fam]
            if not pts: continue
            res = [x['ms'] - f['c0'] - f['ms_per_gmac'] * x['gmac'] for x in pts]
            ks = [x['K'] + rng.uniform(-.18, .18) for x in pts]
            col = colors.get(fam, q.GRAY); filled = fam in colors and fam != 'LEAF'
            ax.scatter(ks, res, s=48 if fam in colors else 32, marker=markers.get(fam, 'o'), facecolors=col if filled else 'none',
                       edgecolors=col, linewidths=1.2, alpha=.9, zorder=4 if fam in colors else 2, label=label.get(fam, fam))
        kmax = max(x['K'] for x in allrows[bb]); xs = np.array([0, kmax + .5])
        ax.plot(xs, f['ms_per_station'] * xs, '-', color='#4a4a48', lw=1.2, alpha=.7, zorder=1)
        ax.set_title(f"{q.NAMES[bb]}: {1000 * f['ms_per_station']:.0f} $\\mu$s per station", fontsize=TITLE_FS)
        ax.set_xlabel('stations K', fontsize=LABEL_FS); ax.set_ylabel('latency $-$ GMAC term (ms)', fontsize=LABEL_FS)
        ax.grid(alpha=.25, lw=.5); ax.tick_params(labelsize=TICK_FS); [s.set_alpha(.3) for s in ax.spines.values()]
    for ax in list(axes.flat)[n:]: ax.axis('off')
    seen = {}
    for a in axes.flat[:n]:
        for hh, ll in zip(*a.get_legend_handles_labels()): seen.setdefault(ll, hh)
    free = list(axes.flat)[n:]
    fig.tight_layout(); b0, b1 = free[0].get_position(), free[-1].get_position()
    fig.legend(seen.values(), seen.keys(), loc='center', bbox_to_anchor=((b0.x0 + b1.x1) / 2, (b0.y0 + b0.y1) / 2),
               fontsize=LEGEND_FS, markerscale=1.6, frameon=False, ncol=2)
    for ext in ('png', 'pdf'): fig.savefig(q.FIG / f'equal_gmac_residual_K_large.{ext}', dpi=160)
    plt.close(fig); print('wrote', q.FIG / 'equal_gmac_residual_K_large.pdf')


def pricing_large():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from leaf.reproduction.experiments.latency_pricing import final_ms, passed_regimes
    regimes = passed_regimes(); bbs = ('deit_s', 'deit_b', 'mae_b')
    fig, axes = plt.subplots(len(bbs), len(regimes), figsize=(4.6 * len(regimes), 4.0 * len(bbs)), squeeze=False)
    for i, bb in enumerate(bbs):
        jobs = q.load(q.PRICING / bb / 'plans.json')['jobs']
        for j, reg in enumerate(regimes):
            ax = axes[i][j]; ms = final_ms(bb, reg)
            for grp, col, mk, lab in (('gmac', q.GRAY, 's', 'GMAC-planned DP'), ('salt', q.COLORS['LEAF'], 'o', 'SALT-planned DP (this device)')):
                pts = []
                for name, row in jobs.items():
                    if row.get('group') != grp or 'config' not in row or (grp == 'salt' and row.get('regime') != reg): continue
                    k = f"{bb}-{row['config_sha']}"; c = q.correct(q.PRICING / bb, row['config_sha'], 'val')
                    if k in ms and c is not None: pts.append((ms[k][0], float(c.mean() * 100)))
                if not pts: continue
                keep = q.pareto(pts); xs, ys = zip(*sorted(pts[t] for t in keep))
                ax.plot(xs, ys, '-', color=col, lw=2.4 if grp == 'salt' else 1.2, zorder=3 if grp == 'salt' else 2)
                ax.scatter(*zip(*pts), s=48, marker=mk, facecolors=col if grp == 'salt' else 'none', edgecolors=col, zorder=3, label=lab)
            dense = ms.get(f'{bb}-7ae560d3b3fdeea9')
            if dense:
                ax.axvline(dense[0], color='#52514e', lw=1, ls=':')
                ax.text(dense[0], ax.get_ylim()[0], ' dense', fontsize=TICK_FS, color='#52514e', va='bottom')
            ax.set_title(f"{q.NAMES[bb]}, {reg.replace('_', ' ').replace('gpu', 'GPU').replace('cpu', 'CPU')}", fontsize=TITLE_FS)
            ax.grid(alpha=.25, lw=.5); ax.tick_params(labelsize=TICK_FS)
            ax.set_xlabel('measured latency (ms)', fontsize=LABEL_FS); ax.set_ylabel('val top-1 (%)', fontsize=LABEL_FS)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=2, fontsize=LEGEND_FS, markerscale=1.5, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, .955))
    for ext in ('png', 'pdf'): fig.savefig(q.FIG / f'pricing_salt_vs_gmac_large.{ext}', dpi=160)
    plt.close(fig); print('wrote', q.FIG / 'pricing_salt_vs_gmac_large.pdf')


def m0_residual_large(bad_json):
    """Supplementary figure: latency residual over a GMAC fit vs K for the earlier LEAF-M0 campaign (DeiT-S, DeiT-B), redrawn
    from the second-campaign collection (artifacts/paper/v1/tools/collect2.py output) with the logic of
    artifacts/paper/v1/tools/assets.py lat_gmac; only sizes change."""
    import json
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    bad = json.load(open(bad_json))
    fam = lambda n: 'DR' if n.startswith('DR') else 'ablation' if n.startswith('ablation/') else n.split('/')[0]
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for ax, bb in zip(axs, ('deit_s', 'deit_b')):
        pts = []
        for n, r in bad[bb]['rows'].items():
            ms = {b: x[0] for b, x in r['lat'].items()}
            if r['gmac'] is None or '32' not in ms or n == 'dense': continue
            f = fam(n)
            if f not in ('DR', 'latency', 'latency_gmac', 'ablation') or n.startswith('ablation/D4'): continue
            st = r['state']; pts.append((r['gmac'], ms['32'], len(st['sites']) if st else None, f))
        X = np.array([t[0] for t in pts]); Y = np.array([t[1] for t in pts])
        A = np.vstack([np.ones_like(X), X]).T; res = Y - A @ np.linalg.lstsq(A, Y, rcond=None)[0]
        Ks = [t[2] if t[2] is not None else 11 for t in pts]
        for (_, _, _, f), rr, k in zip(pts, res, Ks):
            ax.scatter(k, rr, s=34, c='#1f77b4' if f == 'DR' else ('#d62728' if f.startswith('latency') else '#999999'), alpha=.8,
                       zorder=3 if f != 'ablation' else 2)
        cf = np.polyfit(np.array(Ks, float), res, 1); xs = np.linspace(min(Ks), max(Ks), 10)
        ax.plot(xs, np.polyval(cf, xs), 'k--', lw=1.3)
        ax.set_title(f'{q.NAMES[bb]}: slope {cf[0] * 1000:.0f} $\\mu$s/station', fontsize=TITLE_FS)
        ax.set_xlabel('number of reduction stations $K$', fontsize=LABEL_FS)
        ax.set_ylabel('latency residual vs. GMAC fit (ms)', fontsize=LABEL_FS)
        ax.grid(alpha=.25); ax.tick_params(labelsize=TICK_FS)
        for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
        print(bb, 'slope us/station', round(cf[0] * 1000, 1))
    axs[1].legend([Line2D([], [], marker='o', ls='', ms=9, color=c) for c in ('#d62728', '#1f77b4', '#999999')],
                  [f'early {q.PAPER_NAME}', 'DiffRate', 'ablation schedules'], fontsize=LEGEND_FS - 4, frameon=False)
    fig.tight_layout()
    for ext in ('png', 'pdf'): fig.savefig(q.FIG / f'm0_latency_residual_K_large.{ext}', dpi=160)
    plt.close(fig); print('wrote', q.FIG / 'm0_latency_residual_K_large.pdf')


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    residual_figure_large()
    pricing_large()
    if argv: m0_residual_large(argv[0])   # LEAF-M0 second-campaign data (results/leaf_m0_summaries/second_campaign_data.json)


if __name__ == '__main__':
    main()
