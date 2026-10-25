"""Q1 revision reports (CPU only): latency-paired tests, accuracy-latency frontiers, tables and schedule export.

  python leaf/reproduction/experiments/campaign_reports.py p2     # LEAF vs every family, 7 backbones (val/V2), figures + CSV + JSON
  python leaf/reproduction/experiments/campaign_reports.py p1     # SALT vs GMAC on GPU bs4/8 and CPU bs1/8, figures
  python leaf/reproduction/experiments/campaign_reports.py seg    # DINOv2-B ADE20K mIoU vs latency, figure
  python leaf/reproduction/experiments/campaign_reports.py export # schedules/q1_*/2026-10-18/<backbone>.json
Pairing: nearest measured latency within tau = max(0.05 ms, 2 x pooled SD of the three round means), cost-only;
McNemar per image, Holm over every contrast of a split. Frontier lines join each family's Pareto points.
"""
import csv
import glob
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np

REPO = Path(__file__).resolve().parents[3]
FIG = REPO / 'results/figures'
MAIN_COMPARISON = REPO / 'results/main_comparison'
PRICING = REPO / 'results/latency_pricing'
SEG = REPO / 'results/segmentation'
BBS = ('deit_t', 'deit_s', 'deit_b', 'augreg_s', 'augreg_b', 'mae_b', 'mae_l')
NAMES = dict(deit_t='DeiT-T', deit_s='DeiT-S', deit_b='DeiT-B', augreg_s='ViT-S (AugReg)', augreg_b='ViT-B (AugReg)',
             mae_b='ViT-B (MAE)', mae_l='ViT-L (MAE)')
# Reference palette slots 1-3 (validated all-pairs, light mode); every other family is gray with its own marker.
COLORS = {'LEAF': '#2a78d6', 'DiffRate': '#eb6834', 'LEAF-M0': '#1baf7a'}
GRAY = '#8a8984'
PAPER_NAME = 'LEAF'   # method name in paper figures (renamed from LEAF on 2026-10-18)
MARKERS = {'LEAF': 'o', 'DiffRate': 's', 'LEAF-M0': 'D', 'ToMe': 'v', 'ToMe+M3': '^', 'ToFu': '<',
           'RAPID': '>', 'PiToMe': 'p', 'EViT': 'h', 'PPT': 'X', 'One-More-Token': 'P', 'MCTF': '*'}
KIND = dict(tome='ToMe', tome_m3='ToMe+M3', tofu='ToFu', rapid='RAPID', pitome='PiToMe', evit='EViT', ppt='PPT',
            onemore='One-More-Token', mctf='MCTF')


def load(p): return json.loads(Path(p).read_text())


def save(p, obj):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(obj, indent=1, default=float))


def family(name, cfg):
    if name.startswith('leaf_alpha4/'): return 'LEAF'
    if name.startswith('ref/paper/latency/'): return 'LEAF-M0'
    if name.startswith('ref/history/baseline/DR'): return 'DiffRate'
    if name.startswith('ref/') and cfg['kind'] in KIND: return KIND[cfg['kind']]
    return None


def round_means(folder, bs):
    rows = {}
    for f in sorted(glob.glob(str(Path(folder) / '*.json'))):   # fixed order: the mean of the rounds does not depend on the file system
        d = load(f); m = d['measurements'].get(str(bs)) if isinstance(d['measurements'], dict) else None
        if isinstance(m, dict) and 'ms_mean' in m: rows.setdefault(d['id'], []).append(m['ms_mean'])
    return {k: (float(np.mean(v)), float(np.std(v, ddof=1))) for k, v in rows.items() if len(v) == 3}


def correct(root, sha, split):
    p = Path(root) / split / 'runs' / f'{sha}.npz'
    return np.load(p)['correct'] if p.exists() else None


def pareto(points):
    """points: [(ms, acc, ...)] -> indices on the upper-left frontier (faster and more accurate)."""
    order = sorted(range(len(points)), key=lambda i: (points[i][0], -points[i][1])); best = -1e9; keep = []
    for i in order:
        if points[i][1] > best: keep.append(i); best = points[i][1]
    return keep


def tau(a, b): return max(.05, 2 * np.sqrt((a[1] ** 2 + b[1] ** 2) / 2))


# ---------------------------------------------------------------- MAIN_COMPARISON
def main_arms(bb, bs=32):
    jobs = load(MAIN_COMPARISON / bb / 'plans.json')['jobs']; ms = round_means(MAIN_COMPARISON / bb / 'latency', bs); arms = []
    for name, row in jobs.items():
        if 'config' not in row: continue
        fam = family(name, row['config']); tid = f"{bb}-{row['config_sha']}"
        if fam is None or tid not in ms: continue
        val, v2 = correct(MAIN_COMPARISON / bb, row['config_sha'], 'val'), correct(MAIN_COMPARISON / bb, row['config_sha'], 'v2')
        if val is None or v2 is None: continue
        arms.append(dict(name=name, family=fam, sha=row['config_sha'], ms=ms[tid][0], sd=ms[tid][1],
                         val=float(val.mean() * 100), v2=float(v2.mean() * 100), gmac=row.get('gmac'),
                         budget=row.get('budget', {}).get('latency_ms') if isinstance(row.get('budget'), dict) else None))
    return arms


def p2():
    from leaf.evaluation.data import mcnemar, holm
    rows, tests = {}, []
    for bb in BBS:
        if not (MAIN_COMPARISON / bb / 'latency_done.json').exists(): print('skip (not timed)', bb); continue
        arms = main_arms(bb); rows[bb] = arms
        ours = [a for a in arms if a['family'] == 'LEAF']
        for o in ours:
            for fam in sorted({a['family'] for a in arms} - {'LEAF'}):
                cand = [a for a in arms if a['family'] == fam]
                best = min(cand, key=lambda a: abs(a['ms'] - o['ms']))
                t = tau((o['ms'], o['sd']), (best['ms'], best['sd']))
                if abs(best['ms'] - o['ms']) > t: continue
                rec = dict(backbone=bb, ours=o['name'], family=fam, ref=best['name'], ours_ms=o['ms'], ref_ms=best['ms'], tau=t)
                for split in ('val', 'v2'):
                    ca, cb = correct(MAIN_COMPARISON / bb, o['sha'], split), correct(MAIN_COMPARISON / bb, best['sha'], split)
                    p, b_, c_ = mcnemar(ca, cb)
                    rec[split] = dict(delta=float((ca.mean() - cb.mean()) * 100), p=p)
                tests.append(rec)
    for split in ('val', 'v2'): holm([t[split] for t in tests])
    save(REPO / 'results/main_comparison/report.json', dict(arms=rows, tests=tests))
    with open(REPO / 'results/main_comparison/paired_tests.csv', 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['backbone', 'ours', 'family', 'ref', 'ours_ms', 'ref_ms', 'tau', 'dval', 'holm_val', 'dv2', 'holm_v2'])
        for t in tests: w.writerow([t['backbone'], t['ours'], t['family'], t['ref'], f"{t['ours_ms']:.3f}", f"{t['ref_ms']:.3f}", f"{t['tau']:.3f}",
                                    f"{t['val']['delta']:+.3f}", f"{t['val']['holm']:.3g}", f"{t['v2']['delta']:+.3f}", f"{t['v2']['holm']:.3g}"])
    summary = {}
    for t in tests:
        s = summary.setdefault(t['family'], dict(n=0, val_w=0, val_l=0, v2_w=0, v2_l=0, dval=0., dv2=0.))
        s['n'] += 1; s['dval'] += t['val']['delta']; s['dv2'] += t['v2']['delta']
        for split in ('val', 'v2'):
            if t[split]['holm'] < .05: s[f'{split}_w' if t[split]['delta'] > 0 else f'{split}_l'] += 1
    for fam, s in sorted(summary.items()):
        print(f"{fam:18s} pairs {s['n']:3d}  val W/L {s['val_w']}/{s['val_l']}  mean {s['dval']/s['n']:+.2f}   V2 W/L {s['v2_w']}/{s['v2_l']}  mean {s['dv2']/s['n']:+.2f}")
    for split in ('val', 'v2'): frontier_figure(rows, split)
    return summary


def frontier_figure(rows, split):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    bbs = [b for b in BBS if b in rows]; n = len(bbs); cols = 4; r = (n + cols - 1) // cols
    fig, axes = plt.subplots(r, cols, figsize=(4.2 * cols, 3.4 * r), squeeze=False)
    for ax, bb in zip(axes.flat, bbs):
        arms = rows[bb]; ours = [a for a in arms if a['family'] == 'LEAF']
        lo, hi = min(a['ms'] for a in ours) * .85, max(a['ms'] for a in ours) * 1.12
        ymin = min(a[split] for a in ours) - 1.5
        for fam in ['ToMe', 'ToMe+M3', 'ToFu', 'RAPID', 'PiToMe', 'EViT', 'PPT', 'One-More-Token', 'MCTF', 'LEAF-M0', 'DiffRate', 'LEAF']:
            pts = [(a['ms'], a[split]) for a in arms if a['family'] == fam and lo <= a['ms'] <= hi and a[split] >= ymin]
            if not pts: continue
            col = COLORS.get(fam, GRAY); lw = 2 if fam in COLORS else 1; z = 5 if fam == 'LEAF' else 3 if fam in COLORS else 2
            keep = pareto(pts)
            xs, ys = zip(*sorted(pts[i] for i in keep))
            ax.plot(xs, ys, '-', color=col, lw=lw, alpha=.9 if fam in COLORS else .6, zorder=z)
            ax.scatter(*zip(*pts), s=36 if fam in COLORS else 24, marker=MARKERS[fam], facecolors=col if fam in COLORS else 'none',
                       edgecolors=col, linewidths=1, alpha=.95, zorder=z, label=fam)
        ax.set_title(NAMES[bb], fontsize=10); ax.set_xlim(lo, hi); ax.grid(alpha=.25, lw=.5)
        ax.set_xlabel('latency, bs32 (ms)', fontsize=8); ax.set_ylabel(f'{"ImageNet val" if split == "val" else "ImageNet-V2"} top-1 (%)', fontsize=8)
        ax.tick_params(labelsize=7); [s.set_alpha(.3) for s in ax.spines.values()]
    for ax in list(axes.flat)[n:]: ax.axis('off')
    h, l = axes.flat[0].get_legend_handles_labels()
    seen = {}
    for a in axes.flat[:n]:
        for hh, ll in zip(*a.get_legend_handles_labels()): seen.setdefault(ll, hh)
    fig.legend(seen.values(), seen.keys(), loc='lower right', bbox_to_anchor=(.98, .06), fontsize=8, frameon=False, ncol=2)
    fig.suptitle(f'LEAF vs re-run baselines, measured latency (RTX 5060 Ti, BF16, one session per backbone) - {split}', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .96)); FIG.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'): fig.savefig(FIG / f'main_frontier_{split}.{ext}', dpi=160)
    plt.close(fig); print('wrote', FIG / f'main_frontier_{split}.png')


# ---------------------------------------------------------------- PRICING
def p1():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from leaf.reproduction.experiments.latency_pricing import final_ms, passed_regimes, REGIMES
    regimes = passed_regimes(); bbs = ('deit_s', 'deit_b', 'mae_b')
    fig, axes = plt.subplots(len(bbs), len(regimes), figsize=(3.8 * len(regimes), 3.2 * len(bbs)), squeeze=False)
    for i, bb in enumerate(bbs):
        jobs = load(PRICING / bb / 'plans.json')['jobs']
        for j, reg in enumerate(regimes):
            ax = axes[i][j]; ms = final_ms(bb, reg)
            for grp, col, mk, lab in (('gmac', GRAY, 's', 'GMAC-planned DP'), ('salt', COLORS['LEAF'], 'o', 'SALT-planned DP (this device)')):
                pts = []
                for n, row in jobs.items():
                    if row.get('group') != grp or 'config' not in row or (grp == 'salt' and row.get('regime') != reg): continue
                    k = f"{bb}-{row['config_sha']}"; c = correct(PRICING / bb, row['config_sha'], 'val')
                    if k in ms and c is not None: pts.append((ms[k][0], float(c.mean() * 100)))
                if not pts: continue
                keep = pareto(pts); xs, ys = zip(*sorted(pts[q] for q in keep))
                ax.plot(xs, ys, '-', color=col, lw=2 if grp == 'salt' else 1, zorder=3 if grp == 'salt' else 2)
                ax.scatter(*zip(*pts), s=30, marker=mk, facecolors=col if grp == 'salt' else 'none', edgecolors=col, zorder=3, label=lab)
            dense = ms.get(f'{bb}-7ae560d3b3fdeea9')
            if dense: ax.axvline(dense[0], color='#52514e', lw=.8, ls=':'); ax.text(dense[0], ax.get_ylim()[0], ' dense', fontsize=7, color='#52514e', va='bottom')
            ax.set_title(f"{NAMES[bb]} - {reg.replace('_', ' ')}", fontsize=9); ax.grid(alpha=.25, lw=.5); ax.tick_params(labelsize=7)
            ax.set_xlabel('measured latency (ms)', fontsize=8); ax.set_ylabel('val top-1 (%)', fontsize=8)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=2, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, .95)); FIG.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'): fig.savefig(FIG / f'pricing_salt_vs_gmac.{ext}', dpi=160)
    plt.close(fig); print('wrote', FIG / 'pricing_salt_vs_gmac.png')


# ---------------------------------------------------------------- segmentation
def seg():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    jobs = load(SEG / 'dinov2_b' / 'plans.json')['jobs']; out = {}
    for bs in (1, 8):
        ms = round_means(SEG / 'dinov2_b' / 'timing_final', bs)
        for name, row in jobs.items():
            if 'config' not in row: continue
            rec = load(SEG / 'completed' / 'val' / f"dinov2_b-{row['config_sha']}.json")
            out.setdefault(name, dict(name=name, miou=rec['miou']))[f'ms_bs{bs}'] = ms[row['config_sha']][0]
    head = load(SEG / 'dinov2_b' / 'head_latency.json')
    save(SEG / 'report.json', dict(arms=out, head_ms={k: v['ms_mean'] for k, v in head.items()}))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, bs in zip(axes, (1, 8)):
        for fam, col, mk in (('tome', GRAY, 'v'), ('laps', COLORS['LEAF'], 'o')):
            pts = sorted((r[f'ms_bs{bs}'], r['miou'], r['name']) for r in out.values() if r['name'].startswith(fam))
            ax.plot([p[0] for p in pts], [p[1] for p in pts], '-', color=col, lw=2 if fam == 'laps' else 1)
            ax.scatter([p[0] for p in pts], [p[1] for p in pts], marker=mk, s=34, facecolors=col if fam == 'laps' else 'none',
                       edgecolors=col, label='LEAF (SALT + P+D + M3)' if fam == 'laps' else 'ToMe (unmerge)', zorder=3)
            for p in pts: ax.annotate(p[2].split('/')[1], (p[0], p[1]), fontsize=6, xytext=(3, -8), textcoords='offset points', color='#52514e')
        d = out['dense']; ax.scatter([d[f'ms_bs{bs}']], [d['miou']], marker='*', s=90, color='#0b0b0b', label='dense', zorder=4)
        ax.axvline(d[f'ms_bs{bs}'], color='#52514e', lw=.8, ls=':')
        ax.set_xlabel(f'encoder latency incl. restoration, bs{bs} (ms)', fontsize=8); ax.set_ylabel('ADE20K val mIoU (%)', fontsize=8)
        ax.set_title(f'DINOv2 ViT-B/14 + official linear head, bs{bs}', fontsize=9); ax.grid(alpha=.25, lw=.5); ax.tick_params(labelsize=7)
    axes[0].legend(fontsize=8, frameon=False, loc='lower right')
    fig.tight_layout(); FIG.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'): fig.savefig(FIG / f'seg_miou_latency.{ext}', dpi=160)
    plt.close(fig); print('wrote', FIG / 'seg_miou_latency.png')
    for r in sorted(out.values(), key=lambda r: r['ms_bs1']): print(f"{r['name']:10s} mIoU {r['miou']:6.2f}  bs1 {r['ms_bs1']:6.2f} ms  bs8 {r['ms_bs8']:7.2f} ms")
    return out


# ---------------------------------------------------------------- export
def export():
    from leaf.planner.plan import config_details
    from leaf.backbones import SPECS
    for campaign, root, bbs in (('q1_main_v2', MAIN_COMPARISON, BBS), ('q1_latency_pricing', PRICING, ('deit_s', 'deit_b', 'mae_b'))):
        for bb in bbs:
            doc = load(root / bb / 'plans.json'); out = {}
            for name, row in doc['jobs'].items():
                if 'config' not in row or row['config']['kind'] not in ('pd', 'pdm'): continue
                if not (name.startswith(('leaf_alpha4/', 'salt/', 'gmac/', 'ablation/', 'seed'))): continue
                det = row.get('details') or config_details(row['config'], SPECS[bb])
                out[name] = dict(stations=[dict(block=r['block'], tokens_after=r['merge_keep'], pruned=r['pruned'], merged=r['merged'],
                                                theta=r['theta'], kappa=r['kappa']) for r in det['token_path'] if r['removed'] > 0],
                                 config=row['config'], config_sha=row['config_sha'], predicted_ms=row.get('predicted_ms'),
                                 budget=row.get('budget') or row.get('budget_ms'), gmac=row.get('gmac'))
            save(REPO / 'schedules' / campaign / '2026-10-18' / f'{bb}.json',
                 dict(backbone=bb, timm_id=SPECS[bb].timm_id, plans_sha=doc['sha'], schedules=out,
                      usage="model = leaf.evaluation.runner.build(vit, spec, config)  # vit, spec = leaf.backbones.load(backbone)"))
    doc = load(SEG / 'dinov2_b' / 'plans.json')
    save(REPO / 'schedules' / 'q1_seg' / '2026-10-18' / 'dinov2_b.json',
         dict(backbone='dinov2_b (vit_base_patch14_dinov2.lvd142m, 518 px)', schedules={k: dict(config=v['config'], details=v.get('details'), predicted_ms=v.get('predicted_ms'), budget_ms=v.get('budget_ms'))
                                                    for k, v in doc['jobs'].items() if k.startswith('laps/')},
              usage="model = models.seg_backbone.CenterPad(models.seg_backbone.restoring_model(vit, spec, config)); vit, spec = load_dino_encoder()"))
    print('exported schedules to', REPO / 'schedules')


def frontier_figure_single(rows, split, out_stem, legend_fs=None, dense=None, dense_style='line'):
    """Paper version of frontier_figure: one method (LEAF) against the re-run baselines, no earlier variant, no suptitle."""
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    label = {'LEAF': f'{PAPER_NAME} (ours)'}
    bbs = [b for b in BBS if b in rows]; n = len(bbs); cols = 4; r = (n + cols - 1) // cols
    fig, axes = plt.subplots(r, cols, figsize=(4.2 * cols, 3.4 * r), squeeze=False)
    for ax, bb in zip(axes.flat, bbs):
        arms = [a for a in rows[bb] if a['family'] != 'LEAF-M0']; ours = [a for a in arms if a['family'] == 'LEAF']
        lo, hi = min(a['ms'] for a in ours) * .85, max(a['ms'] for a in ours) * 1.12
        ymin = min(a[split] for a in ours) - 1.5
        if dense is not None and bb in dense and dense_style == 'point': hi = max(hi, dense[bb]['ms'] * 1.05)   # room for the dense point
        for fam in ['ToMe', 'ToMe+M3', 'ToFu', 'RAPID', 'PiToMe', 'EViT', 'PPT', 'One-More-Token', 'MCTF', 'DiffRate', 'LEAF']:
            pts = [(a['ms'], a[split]) for a in arms if a['family'] == fam and lo <= a['ms'] <= hi and a[split] >= ymin]
            if not pts: continue
            col = COLORS.get(fam, GRAY); lw = 2 if fam in COLORS else 1; z = 5 if fam == 'LEAF' else 3 if fam in COLORS else 2
            xs, ys = zip(*sorted(pts[i] for i in pareto(pts)))
            ax.plot(xs, ys, '-', color=col, lw=lw, alpha=.9 if fam in COLORS else .6, zorder=z)
            ax.scatter(*zip(*pts), s=36 if fam in COLORS else 24, marker=MARKERS[fam], facecolors=col if fam in COLORS else 'none',
                       edgecolors=col, linewidths=1, alpha=.95, zorder=z, label=label.get(fam, fam))
        if dense is not None and bb in dense and dense_style == 'line':   # 2026-10-19: dense accuracy as a reference line
            dms, dacc = dense[bb]['ms'], dense[bb][split]
            ax.axhline(dacc, color='#52514e', lw=.9, ls=':', zorder=1, label='dense')
            ax.text(lo + .02 * (hi - lo), dacc, f'dense {dacc:.2f} ({dms:.1f} ms)', fontsize=7.5, color='#52514e', va='bottom')
            ax.set_ylim(top=max(ax.get_ylim()[1], dacc + .3 * (ax.get_ylim()[1] - ax.get_ylim()[0]) / 6))
        if dense is not None and bb in dense and dense_style == 'point':  # 2026-10-19: dense model as its own point
            dms, dacc = dense[bb]['ms'], dense[bb][split]
            ax.scatter([dms], [dacc], marker='*', s=150, color='#0b0b0b', zorder=6, label='dense')
            hi = max(hi, dms * 1.05)
        ax.set_title(NAMES[bb], fontsize=10); ax.set_xlim(lo, hi); ax.grid(alpha=.25, lw=.5)
        ax.set_xlabel('latency, bs32 (ms)', fontsize=8); ax.set_ylabel(f'{"ImageNet val" if split == "val" else "ImageNet-V2"} top-1 (%)', fontsize=8)
        ax.tick_params(labelsize=7); [s.set_alpha(.3) for s in ax.spines.values()]
    for ax in list(axes.flat)[n:]: ax.axis('off')
    seen = {}
    for a in axes.flat[:n]:
        for hh, ll in zip(*a.get_legend_handles_labels()): seen.setdefault(ll, hh)
    if legend_fs is None:
        fig.legend(seen.values(), seen.keys(), loc='lower right', bbox_to_anchor=(.98, .06), fontsize=8, frameon=False, ncol=2)
    else:                                   # 2026-10-19: readable legend in the empty panel of the grid
        list(axes.flat)[n].legend(seen.values(), seen.keys(), loc='center', fontsize=legend_fs, markerscale=1.6, frameon=False, ncol=1)
    fig.tight_layout(); FIG.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'): fig.savefig(FIG / f'{out_stem}_{split}.{ext}', dpi=160)
    plt.close(fig); print('wrote', FIG / f'{out_stem}_{split}.png')


def main_single():
    """Frontier figures of the single-method paper from the saved report (no re-timing): figures/q1/main_frontier_single_{val,v2}."""
    rows = load(MAIN_COMPARISON / 'report.json')['arms']
    for split in ('val', 'v2'): frontier_figure_single(rows, split, 'main_frontier_single')


def main_single_paper():
    """As main_single, with the legend enlarged in the empty eighth panel (paper v2 Figures 2 and 6)."""
    rows = load(MAIN_COMPARISON / 'report.json')['arms']
    dense = {}
    for bb in rows:   # 2026-10-19: dense reference from the same session
        sha = load(MAIN_COMPARISON / bb / 'plans.json')['jobs']['ref/dense']['config_sha']
        dense[bb] = dict(ms=round_means(MAIN_COMPARISON / bb / 'latency', 32)[f'{bb}-{sha}'][0],
                         **{sp: float(correct(MAIN_COMPARISON / bb, sha, sp).mean() * 100) for sp in ('val', 'v2')})
    for split in ('val', 'v2'): frontier_figure_single(rows, split, 'main_frontier_single', legend_fs=15, dense=dense, dense_style='point')


def ablation_report(bbs=('deit_b', 'mae_b', 'augreg_s')):
    """Main-campaign ablation and calibration-seed arms against the full method of the same budget (leaf_alpha4):
    val / V2 top-1 difference (points), McNemar p, Holm over all arms whose config differs from the reference (per
    split), bs32 latency difference -> MAIN_COMPARISON/ablation_report.json (read by leaf.reproduction.report.tables.ablation)."""
    from leaf.evaluation.data import mcnemar, holm
    rows = []
    for bb in bbs:
        jobs = load(MAIN_COMPARISON / bb / 'plans.json')['jobs']; ms = round_means(MAIN_COMPARISON / bb / 'latency', 32)
        for name, row in jobs.items():
            if not name.startswith(('ablation/', 'seed')) or 'config' not in row: continue
            parts = name.split('/'); bid = parts[-1]
            arm = '/'.join(parts[1:-1]) if parts[0] == 'ablation' else parts[0]
            base = jobs[f'leaf_alpha4/{bid}']
            r = dict(bb=bb, arm=arm, budget=bid, same=row['config_sha'] == base['config_sha'])
            for split in ('val', 'v2'):
                a, b = correct(MAIN_COMPARISON / bb, row['config_sha'], split), correct(MAIN_COMPARISON / bb, base['config_sha'], split)
                r[split] = dict(delta=100 * float(a.mean() - b.mean()), p=mcnemar(a.astype(bool), b.astype(bool))[0])
            ka, kb = f"{bb}-{row['config_sha']}", f"{bb}-{base['config_sha']}"
            r['dms'] = ms[ka][0] - ms[kb][0] if ka in ms and kb in ms else None
            rows.append(r)
    for split in ('val', 'v2'): holm([r[split] for r in rows if not r['same']])
    save(MAIN_COMPARISON / 'ablation_report.json', rows)
    print('wrote', MAIN_COMPARISON / 'ablation_report.json', len(rows), 'rows')
    return rows


if __name__ == '__main__':
    {'main': p2, 'pricing': p1, 'seg': seg, 'export': export, 'main_single': main_single, 'main_single_paper': main_single_paper,
     'ablation': ablation_report}[sys.argv[1]]()
