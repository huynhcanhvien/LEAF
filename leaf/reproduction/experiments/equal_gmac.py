"""Equal-GMAC comparison on the main campaign: same GMAC, measured latency, accuracy and station count K.

python leaf/reproduction/experiments/equal_gmac.py gmac      # GMAC of every timed baseline arm (reuses gmac_planning gmacs.json; FlopCounter otherwise; GPU, no timing)
python leaf/reproduction/experiments/equal_gmac.py analyze   # pairs, McNemar/Holm, station-cost regression, figure, report JSON (CPU)

Uses only existing artifacts of results/main_comparison (latency of the exclusive bs32 session, per-image
correctness). Ours = LEAF planned under a GMAC budget (`ref/M3/gmac/*`: same probes, score, M3 operator, GMAC instead of SALT).
GMAC conventions are those of the gmac_planning campaign: our configs and DiffRate by leaf.planner.gmac.config_cost
(matrix/conv MACs + grouped D), adapters by torch FlopCounter on the actual module (matrix/conv/SDPA).

Pre-declared pairing (fixed before looking at the pairs): for each GMAC-capped LEAF schedule and each family, the family point with
the nearest GMAC, accepted if |dGMAC| <= 3% of ours. Latency: dms = ours - theirs, tau = max(0.05 ms, 2 pooled SD); 'faster' if
dms <= -tau, 'slower' if dms >= tau. Accuracy: exact McNemar on the same images, Holm over all pairs of a split.
"""
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
import numpy as np

from leaf.reproduction.experiments.campaign_reports import BBS, NAMES, COLORS, GRAY, MARKERS, family, load, save, round_means, correct, tau, MAIN_COMPARISON

OUT = REPO / 'results/equal_gmac'
DPM3 = REPO / 'results/gmac_planning'
FIG = REPO / 'results/figures'
EVERY_BLOCK = ('ToMe', 'ToMe+M3', 'ToFu', 'PiToMe', 'RAPID', 'MCTF')
FAMILIES = ('DiffRate', 'ToMe', 'ToMe+M3', 'ToFu', 'RAPID', 'PiToMe', 'EViT', 'PPT', 'One-More-Token', 'MCTF')
REL_GMAC = 0.03


def stations(fam, row, depth):
    """Number of blocks with a reduction step: planner details for pd/pdm, the method's rule otherwise."""
    d = row.get('details')
    if isinstance(d, dict) and 'K' in d: return int(d['K'])
    cfg = row['config']
    if cfg['kind'] in ('pd', 'pdm'):
        from leaf.planner.plan import config_details
        from leaf.backbones import SPECS
        return int(config_details(cfg, next(s for s in SPECS.values() if s.depth == depth))['K'])
    if fam in EVERY_BLOCK: return depth
    if fam in ('EViT', 'PPT'): return 3
    if fam == 'One-More-Token': return 1
    raise ValueError(fam)


def arms(bb):
    """Every timed arm of the session with val/V2 correctness: baselines, DiffRate, LEAF (latency) and LEAF-GMAC."""
    from leaf.backbones import SPECS
    from leaf.planner.gmac import config_cost
    spec = SPECS[bb]; jobs = load(MAIN_COMPARISON / bb / 'plans.json')['jobs']; ms = round_means(MAIN_COMPARISON / bb / 'latency', 32)
    gm = load(OUT / bb / 'gmacs.json') if (OUT / bb / 'gmacs.json').exists() else {}
    out = []
    for name, row in jobs.items():
        if 'config' not in row: continue
        fam = 'LEAF-GMAC' if name.startswith('ref/M3/gmac/') else family(name, row['config'])
        tid = f"{bb}-{row['config_sha']}"
        if fam is None or fam == 'LEAF-M0' or tid not in ms: continue
        if row['config']['kind'] in ('pd', 'pdm'): g = config_cost(row['config'], spec)
        else: g = gm.get(name, {}).get('gmac')
        out.append(dict(name=name, family=fam, sha=row['config_sha'], ms=ms[tid][0], sd=ms[tid][1], gmac=g,
                        K=stations(fam, row, spec.depth), config=row['config']))
    return out


def gmac_stage():
    import torch
    from leaf.backbones import load as load_bb, SPECS
    from leaf.reproduction.experiments.low_budget_and_factorial import build          # the campaign's builder (ToMe+M3, ToFu, RAPID, One-More-Token)
    from leaf.latency.timing import gmacs
    from leaf.planner.gmac import dense_cost
    for bb in BBS:
        prev = load(DPM3 / bb / 'gmacs.json') if (DPM3 / bb / 'gmacs.json').exists() else {}
        path = OUT / bb / 'gmacs.json'; res = load(path) if path.exists() else {}
        todo = [a for a in arms(bb) if a['config']['kind'] not in ('pd', 'pdm') and a['name'] not in res]
        vit = spec = None
        for a in todo:
            key = a['name'].replace('ref/', '', 1)
            if key in prev and 'FlopCounter' in prev[key].get('scope', ''):
                res[a['name']] = dict(prev[key], source='dp_m3_baselines/gmacs.json'); continue
            if vit is None: vit, spec = load_bb(bb)
            model = build(vit, spec, a['config']).eval()
            res[a['name']] = dict(gmac=gmacs(model), scope='torch FlopCounter matrix/conv/SDPA; actual adapter operations counted', source='computed')
            del model; torch.cuda.empty_cache()
        if vit is None: vit, spec = load_bb(bb)
        res['_check_dense'] = dict(flopcounter=gmacs(vit), config_cost=dense_cost(SPECS[bb]))
        save(path, res); print(bb, len(todo), 'new;', res['_check_dense'])
        del vit; torch.cuda.empty_cache()


def mcnemar_pair(bb, a, b, split):
    from leaf.evaluation.data import mcnemar
    ca, cb = correct(MAIN_COMPARISON / bb, a['sha'], split), correct(MAIN_COMPARISON / bb, b['sha'], split)
    p, _, _ = mcnemar(ca, cb)
    return dict(delta=float((ca.mean() - cb.mean()) * 100), p=p, ours=float(ca.mean() * 100), ref=float(cb.mean() * 100))


def station_fit(rows):
    """OLS ms = c0 + c1*GMAC + c2*K over the timed arms of one backbone (MCTF excluded: adapter slower than dense)."""
    pts = [r for r in rows if r['gmac'] is not None and r['family'] != 'MCTF']
    X = np.array([[1., r['gmac'], r['K']] for r in pts]); y = np.array([r['ms'] for r in pts])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None); res = y - X @ coef
    cov = np.linalg.inv(X.T @ X) * (res @ res) / max(len(y) - 3, 1)
    return dict(c0=float(coef[0]), ms_per_gmac=float(coef[1]), ms_per_station=float(coef[2]), se_station=float(np.sqrt(cov[2, 2])),
                n=len(pts), r2=float(1 - res.var() / y.var()))


def analyze():
    from leaf.evaluation.data import holm
    pairs, fits, allrows = [], {}, {}
    for bb in BBS:
        rows = arms(bb); allrows[bb] = rows
        miss = [r['name'] for r in rows if r['gmac'] is None]
        if miss: raise SystemExit(f'{bb}: {len(miss)} arms without GMAC; run the gmac stage first')
        fits[bb] = station_fit(rows)
        ours = [r for r in rows if r['family'] == 'LEAF-GMAC']
        for o in ours:
            for fam in FAMILIES:
                cand = [r for r in rows if r['family'] == fam and abs(r['gmac'] - o['gmac']) <= REL_GMAC * o['gmac']]
                if not cand: continue
                b = min(cand, key=lambda r: (abs(r['gmac'] - o['gmac']), r['ms']))
                t = tau((o['ms'], o['sd']), (b['ms'], b['sd'])); d = o['ms'] - b['ms']
                pairs.append(dict(backbone=bb, ours=o['name'], family=fam, ref=b['name'], gmac=o['gmac'], ref_gmac=b['gmac'],
                                  K=o['K'], ref_K=b['K'], ms=o['ms'], ref_ms=b['ms'], dms=d, rel_ms=d / b['ms'], tau=t,
                                  speed='faster' if d <= -t else 'slower' if d >= t else 'tie',
                                  val=mcnemar_pair(bb, o, b, 'val'), v2=mcnemar_pair(bb, o, b, 'v2')))
    for s in ('val', 'v2'): holm([p[s] for p in pairs])
    summary = {}
    for p in pairs:
        s = summary.setdefault(p['family'], dict(n=0, K=[], ref_K=[], faster=0, slower=0, rel=[], vw=0, vl=0, ww=0, wl=0, dv=[], dv2=[], bbs=set()))
        s['n'] += 1; s['K'].append(p['K']); s['ref_K'].append(p['ref_K']); s['faster'] += p['speed'] == 'faster'; s['slower'] += p['speed'] == 'slower'
        s['rel'].append(p['rel_ms']); s['dv'].append(p['val']['delta']); s['dv2'].append(p['v2']['delta']); s['bbs'].add(p['backbone'])
        for sp, w, l in (('val', 'vw', 'vl'), ('v2', 'ww', 'wl')):
            if p[sp]['holm'] < .05: s[w if p[sp]['delta'] > 0 else l] += 1
    summary = {k: dict(n=v['n'], backbones=sorted(v['bbs']), K=[min(v['K']), max(v['K'])], ref_K=[min(v['ref_K']), max(v['ref_K'])],
                       faster=v['faster'], slower=v['slower'], rel_ms_mean=float(np.mean(v['rel'])), rel_ms_min=float(min(v['rel'])),
                       rel_ms_max=float(max(v['rel'])), val_w=v['vw'], val_l=v['vl'], v2_w=v['ww'], v2_l=v['wl'],
                       dval_mean=float(np.mean(v['dv'])), dv2_mean=float(np.mean(v['dv2']))) for k, v in summary.items()}
    rule = dict(pairing=f'nearest GMAC per (LEAF-GMAC schedule, family), |dGMAC| <= {REL_GMAC:.0%} of ours; ties -> faster',
                latency='bs32 exclusive session of main_comparison; tau = max(0.05 ms, 2 pooled SD)', accuracy='exact McNemar, Holm over all pairs per split')
    save(OUT / 'report.json', dict(rule=rule, pairs=pairs, summary=summary, station_fit=fits))
    for fam in FAMILIES:
        if fam in summary: print(f"{fam:15s}", {k: summary[fam][k] for k in ('n', 'K', 'ref_K', 'faster', 'slower', 'rel_ms_mean', 'val_w', 'val_l', 'v2_w', 'v2_l', 'dval_mean')})
    for bb, f in fits.items(): print(bb, f"{1000 * f['ms_per_station']:.0f} +- {1000 * f['se_station']:.0f} us/station, n={f['n']}, R2={f['r2']:.4f}")
    residual_figure(allrows, fits)


def residual_figure(allrows, fits, legend_fs=None, dense=None):
    """Latency residual over the GMAC term of the joint fit versus K, every family, one panel per backbone."""
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    colors = dict(COLORS, **{'LEAF-GMAC': COLORS['LEAF']}); markers = dict(MARKERS, **{'LEAF-GMAC': 'o'})
    from leaf.reproduction.experiments.campaign_reports import PAPER_NAME
    label = {'LEAF': f'{PAPER_NAME} (latency budget)', 'LEAF-GMAC': f'{PAPER_NAME} (GMAC budget)'}
    bbs = [b for b in BBS if b in allrows]; cols = 4; r = (len(bbs) + cols - 1) // cols
    fig, axes = plt.subplots(r, cols, figsize=(4.2 * cols, 3.3 * r), squeeze=False)
    rng = np.random.default_rng(0)
    for ax, bb in zip(axes.flat, bbs):
        f = fits[bb]
        for fam in ('ToMe', 'ToMe+M3', 'ToFu', 'RAPID', 'PiToMe', 'EViT', 'PPT', 'One-More-Token', 'DiffRate', 'LEAF', 'LEAF-GMAC'):
            pts = [x for x in allrows[bb] if x['family'] == fam]
            if not pts: continue
            res = [x['ms'] - f['c0'] - f['ms_per_gmac'] * x['gmac'] for x in pts]
            ks = [x['K'] + rng.uniform(-.18, .18) for x in pts]
            col = colors.get(fam, GRAY); filled = fam in colors and fam != 'LEAF'
            ax.scatter(ks, res, s=26 if fam in colors else 18, marker=markers.get(fam, 'o'), facecolors=col if filled else 'none',
                       edgecolors=col, linewidths=1, alpha=.9, zorder=4 if fam in colors else 2, label=label.get(fam, fam))
        if dense is not None and bb in dense:   # 2026-10-19: the dense model (K = 0) as a reference point
            ax.scatter([0], [dense[bb]['ms'] - f['c0'] - f['ms_per_gmac'] * dense[bb]['gmac']], marker='*', s=110, color='#0b0b0b', zorder=6, label='dense')
        kmax = max(x['K'] for x in allrows[bb]); xs = np.array([0, kmax + .5])
        ax.plot(xs, f['ms_per_station'] * xs, '-', color='#4a4a48', lw=1, alpha=.7, zorder=1)
        ax.set_title(f"{NAMES[bb]}: {1000 * f['ms_per_station']:.0f} $\\mu$s per station", fontsize=9.5)
        ax.set_xlabel('stations K (blocks with a reduction step)', fontsize=8); ax.set_ylabel('latency - GMAC term (ms)', fontsize=8)
        ax.grid(alpha=.25, lw=.5); ax.tick_params(labelsize=7); [s.set_alpha(.3) for s in ax.spines.values()]
    for ax in list(axes.flat)[len(bbs):]: ax.axis('off')
    seen = {}
    for a in axes.flat[:len(bbs)]:
        for hh, ll in zip(*a.get_legend_handles_labels()): seen.setdefault(ll, hh)
    if legend_fs is None:
        fig.legend(seen.values(), seen.keys(), loc='lower right', bbox_to_anchor=(.98, .05), fontsize=8, frameon=False, ncol=2)
    else:                                   # 2026-10-19: readable legend in the empty panel of the grid
        list(axes.flat)[len(bbs)].legend(seen.values(), seen.keys(), loc='center', fontsize=legend_fs, markerscale=1.6, frameon=False, ncol=1)
    fig.tight_layout(); FIG.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'): fig.savefig(FIG / f'equal_gmac_residual_K.{ext}', dpi=160)
    plt.close(fig); print('wrote', FIG / 'equal_gmac_residual_K.png')


def figure_paper(with_dense=False):
    """Redraw the residual figure from the saved station fits with an enlarged legend (paper figure: latency residual vs stations); no re-pairing."""
    fits = load(OUT / 'report.json')['station_fit']
    from leaf.planner.gmac import dense_cost
    from leaf.backbones import SPECS
    dense = {}
    for bb in BBS:
        sha = load(MAIN_COMPARISON / bb / 'plans.json')['jobs']['ref/dense']['config_sha']
        dense[bb] = dict(ms=round_means(MAIN_COMPARISON / bb / 'latency', 32)[f'{bb}-{sha}'][0], gmac=dense_cost(SPECS[bb]))
    # dense lies outside the GMAC range of the fit, so its residual is an extrapolation; off by default
    residual_figure({bb: arms(bb) for bb in BBS}, fits, legend_fs=15, dense=dense if with_dense else None)


if __name__ == '__main__':
    {'gmac': gmac_stage, 'analyze': analyze, 'figure_paper': figure_paper}[sys.argv[1]]()
