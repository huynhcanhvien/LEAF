"""Paper v3 Figures 2 and 6 with larger ticks, axis labels and panel titles (2026-10-20, user request).

python leaf/reproduction/experiments/accuracy_latency_figures.py   -> results/figures/main_frontier_single_large_{val,v2}.{png,pdf}

Same data and layout as leaf/reproduction/experiments/campaign_reports.py main_single_paper; only the font sizes and marker sizes change.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import leaf.reproduction.experiments.campaign_reports as q   # noqa: E402

TITLE_FS, LABEL_FS, TICK_FS, LEGEND_FS = 15, 13, 12, 17


def frontier_figure_single_large(rows, split, out_stem, dense, cols=4):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    label = {'LEAF': f'{q.PAPER_NAME} (ours)', 'ToMe+M3': 'ToMe+MLERP'}   # display names; the data keys stay
    bbs = [b for b in q.BBS if b in rows]; n = len(bbs); r = (n + 1 + cols - 1) // cols   # one slot for the legend
    fig, axes = plt.subplots(r, cols, figsize=(4.2 * cols, 3.6 * r) if cols == 4 else (5.6 * cols, 4.4 * r), squeeze=False)
    for ax, bb in zip(axes.flat, bbs):
        arms = [a for a in rows[bb] if a['family'] != 'LEAF-M0']; ours = [a for a in arms if a['family'] == 'LEAF']
        lo, hi = min(a['ms'] for a in ours) * .85, max(a['ms'] for a in ours) * 1.12
        ymin = min(a[split] for a in ours) - 1.5
        if bb in dense: hi = max(hi, dense[bb]['ms'] * 1.05)
        for fam in ['ToMe', 'ToMe+M3', 'ToFu', 'RAPID', 'PiToMe', 'EViT', 'PPT', 'One-More-Token', 'MCTF', 'DiffRate', 'LEAF']:
            pts = [(a['ms'], a[split]) for a in arms if a['family'] == fam and lo <= a['ms'] <= hi and a[split] >= ymin]
            if not pts: continue
            col = q.COLORS.get(fam, q.GRAY); lw = 2.2 if fam in q.COLORS else 1.2; z = 5 if fam == 'LEAF' else 3 if fam in q.COLORS else 2
            xs, ys = zip(*sorted(pts[i] for i in q.pareto(pts)))
            ax.plot(xs, ys, '-', color=col, lw=lw, alpha=.9 if fam in q.COLORS else .6, zorder=z)
            ax.scatter(*zip(*pts), s=48 if fam in q.COLORS else 32, marker=q.MARKERS[fam], facecolors=col if fam in q.COLORS else 'none',
                       edgecolors=col, linewidths=1, alpha=.95, zorder=z, label=label.get(fam, fam))
            # 2026-10-22: horizontal bars = between-round SD of the latency (three interleaved rounds)
            err = [(a['ms'], a[split], a['sd']) for a in arms if a['family'] == fam and lo <= a['ms'] <= hi and a[split] >= ymin and 'sd' in a]
            if err: ax.errorbar(*zip(*[(x, y) for x, y, _ in err]), xerr=[e for *_, e in err], fmt='none', ecolor=col, elinewidth=1,
                                capsize=2, alpha=.9, zorder=z)
        if bb in dense:
            ax.scatter([dense[bb]['ms']], [dense[bb][split]], marker='*', s=200, color='#0b0b0b', zorder=6, label='dense')
        ax.set_title(q.NAMES[bb], fontsize=TITLE_FS); ax.set_xlim(lo, hi); ax.grid(alpha=.25, lw=.5)
        ax.set_xlabel('latency, bs32 (ms)', fontsize=LABEL_FS)
        ax.set_ylabel(f'{"ImageNet val" if split == "val" else "ImageNet-V2"} top-1 (%)', fontsize=LABEL_FS)
        ax.tick_params(labelsize=TICK_FS); [s.set_alpha(.3) for s in ax.spines.values()]
    for ax in list(axes.flat)[n:]: ax.axis('off')
    seen = {}
    for a in axes.flat[:n]:
        for hh, ll in zip(*a.get_legend_handles_labels()): seen.setdefault(ll, hh)
    free = list(axes.flat)[n:]
    if len(free) == 1: free[0].legend(seen.values(), seen.keys(), loc='center', fontsize=LEGEND_FS, markerscale=1.5, frameon=False, ncol=1)
    else:   # 3 columns: legend across the free slots of the last row
        fig.tight_layout(); b0, b1 = free[0].get_position(), free[-1].get_position()
        fig.legend(seen.values(), seen.keys(), loc='center', bbox_to_anchor=((b0.x0 + b1.x1) / 2, (b0.y0 + b0.y1) / 2), fontsize=LEGEND_FS + 2,
                   markerscale=1.6, frameon=False, ncol=2)
    fig.tight_layout(); q.FIG.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'): fig.savefig(q.FIG / f'{out_stem}_{split}.{ext}', dpi=160)
    plt.close(fig); print('wrote', q.FIG / f'{out_stem}_{split}.pdf')


def main():
    rows = q.load(q.MAIN_COMPARISON / 'report.json')['arms']
    dense = {}
    for bb in rows:   # dense reference from the same session, as in q1_report.main_single_paper
        sha = q.load(q.MAIN_COMPARISON / bb / 'plans.json')['jobs']['ref/dense']['config_sha']
        dense[bb] = dict(ms=q.round_means(q.MAIN_COMPARISON / bb / 'latency', 32)[f'{bb}-{sha}'][0],
                         **{sp: float(q.correct(q.MAIN_COMPARISON / bb, sha, sp).mean() * 100) for sp in ('val', 'v2')})
    for split in ('val', 'v2'): frontier_figure_single_large(rows, split, 'main_frontier_single_large', dense)


if __name__ == '__main__':
    main()
