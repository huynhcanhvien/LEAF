"""CPU-only analyses requested in the paper-v2 self-review (2026-10-19); no new inference, no timing.

python leaf/reproduction/experiments/final_station_and_ci.py lastblock   # final-block station: share of schedules, GMAC it saves, equal-GMAC pairs without it
python leaf/reproduction/experiments/final_station_and_ci.py ci          # paired 95% CIs of the accuracy difference for every latency and equal-GMAC pair
python leaf/reproduction/experiments/final_station_and_ci.py stationfit  # partial R^2 of the station count K in ms = c0 + c1*GMAC + c2*K
python leaf/reproduction/experiments/final_station_and_ci.py all

Inputs: results/main_comparison (plans, per-image correctness, report.json) and
results/equal_gmac (gmacs.json, report.json). Output: results/final_station_and_ci/*.json.

Final-block station. A station in the last block acts after that block's attention, so the class token has already read every
token; the last MLP is per token and a class-token head reads only the class token. For pool='token' backbones the prediction is
therefore identical with and without that station (exact arithmetic; our accuracy evaluation is deterministic), and removing it
only adds the patch-token MLP MACs of the last block. 'lastblock' re-pairs every GMAC-planned schedule at its GMAC *without* the
final station, with the same pre-declared rule as leaf/reproduction/experiments/equal_gmac.py (nearest point within 3%, McNemar, Holm). Latency of
the variant is not claimed: it needs a timing session.
"""
import copy
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
import numpy as np

from leaf.reproduction.experiments.campaign_reports import BBS, load, save, correct, MAIN_COMPARISON

OUT = REPO / 'results/final_station_and_ci'
EQG = REPO / 'results/equal_gmac'


def drop_last_station(cfg, n_tokens):
    """Config with the last-block station removed (197 = no reduction in that block), or None if it has none."""
    c = copy.deepcopy(cfg)
    if c['prune_keep'][-1] == n_tokens and c['merge_keep'][-1] == n_tokens: return None
    c['prune_keep'][-1] = n_tokens; c['merge_keep'][-1] = n_tokens
    if isinstance(c.get('k'), dict): c['k'].pop(str(len(c['prune_keep']) - 1), None)
    return c


def paired_ci(bb, sha_a, sha_b, split, z=1.96):
    """Mean of the per-image correctness difference a - b (points) with a normal 95% CI."""
    d = correct(MAIN_COMPARISON / bb, sha_a, split).astype(float) - correct(MAIN_COMPARISON / bb, sha_b, split).astype(float)
    m, se = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
    return dict(delta=float(m * 100), lo=float((m - z * se) * 100), hi=float((m + z * se) * 100))


def lastblock():
    from leaf.backbones import SPECS
    from leaf.planner.gmac import config_cost
    from leaf.reproduction.experiments.equal_gmac import arms, mcnemar_pair, FAMILIES, REL_GMAC
    from leaf.evaluation.data import holm
    share, pairs, saved = {}, [], {}
    for bb in BBS:
        spec = SPECS[bb]; jobs = load(MAIN_COMPARISON / bb / 'plans.json')['jobs']
        for arm in ('leaf_alpha4/', 'ref/M3/gmac/'):
            rows = [(k, v) for k, v in jobs.items() if k.startswith(arm) and 'config' in v]
            s = share.setdefault(arm, dict(n=0, last=0, by_bb={}))
            for k, v in rows:
                s['n'] += 1
                if drop_last_station(v['config'], spec.n_tokens) is not None:
                    s['last'] += 1; s['by_bb'][bb] = s['by_bb'].get(bb, 0) + 1
        if spec.pool != 'token': continue                      # GAP heads: the final station changes the prediction
        rows = arms(bb)
        for o in [r for r in rows if r['family'] == 'LEAF-GMAC']:
            c = drop_last_station(o['config'], spec.n_tokens)
            g = config_cost(c, spec) if c is not None else o['gmac']
            saved.setdefault(bb, []).append((g - o['gmac']) / g * 100)
            for fam in FAMILIES:
                cand = [r for r in rows if r['family'] == fam and abs(r['gmac'] - g) <= REL_GMAC * g]
                if not cand: continue
                b = min(cand, key=lambda r: (abs(r['gmac'] - g), r['ms']))
                pairs.append(dict(backbone=bb, ours=o['name'], family=fam, ref=b['name'], gmac_with=o['gmac'], gmac_without=g,
                                  ref_gmac=b['gmac'], had_last=c is not None, val=mcnemar_pair(bb, o, b, 'val'), v2=mcnemar_pair(bb, o, b, 'v2')))
    for sp in ('val', 'v2'): holm([p[sp] for p in pairs])
    summ = {}
    for p in pairs:
        s = summ.setdefault(p['family'], dict(n=0, vw=0, vl=0, ww=0, wl=0, dv=[]))
        s['n'] += 1; s['dv'].append(p['val']['delta'])
        for sp, w, l in (('val', 'vw', 'vl'), ('v2', 'ww', 'wl')):
            if p[sp]['holm'] < .05: s[w if p[sp]['delta'] > 0 else l] += 1
    summ = {k: dict(n=v['n'], val_w=v['vw'], val_l=v['vl'], v2_w=v['ww'], v2_l=v['wl'], dval_mean=float(np.mean(v['dv']))) for k, v in summ.items()}
    out = dict(share=share, gmac_saved_pct={bb: [float(min(v)), float(max(v))] for bb, v in saved.items()},
               pairs=pairs, summary=summ, n_pairs=len(pairs),
               rule='GMAC-planned LEAF schedules on pool=token backbones, final-block station removed (same predictions); '
                    'nearest family point within 3% of the GMAC without it; exact McNemar, Holm over these pairs per split')
    save(OUT / 'lastblock.json', out)
    print('share', share); print('saved %', out['gmac_saved_pct']); print('pairs', len(pairs))
    for k, v in summ.items(): print(f'{k:15s}', v)
    return out


def ci():
    from leaf.reproduction.experiments.equal_gmac import arms
    res = {}
    rep = load(MAIN_COMPARISON / 'report.json'); sha = {}
    for bb in BBS: sha[bb] = {k: v['config_sha'] for k, v in load(MAIN_COMPARISON / bb / 'plans.json')['jobs'].items() if 'config_sha' in v}
    lat = []
    for t in rep['tests']:
        if t['family'] == 'LEAF-M0': continue
        bb = t['backbone']
        lat.append(dict(backbone=bb, family=t['family'], ours=t['ours'], ref=t['ref'],
                        **{s: paired_ci(bb, sha[bb][t['ours']], sha[bb][t['ref']], s) for s in ('val', 'v2')}))
    eq = []
    for p in load(EQG / 'report.json')['pairs']:
        bb = p['backbone']
        eq.append(dict(backbone=bb, family=p['family'], ours=p['ours'], ref=p['ref'],
                       **{s: paired_ci(bb, sha[bb][p['ours']], sha[bb][p['ref']], s) for s in ('val', 'v2')}))
    for name, rows in (('latency', lat), ('equal_gmac', eq)):
        r = dict(n=len(rows))
        for s in ('val', 'v2'):
            lo = min(rows, key=lambda x: x[s]['lo'])
            r[s] = dict(min_lo=lo[s]['lo'], min_lo_pair=f"{lo['backbone']} {lo['ours']} vs {lo['ref']}",
                        n_lo_above={str(m): sum(x[s]['lo'] > -m for x in rows) for m in (0.25, 0.5, 1.0)},
                        dr_min_lo=min((x[s]['lo'] for x in rows if x['family'] == 'DiffRate'), default=None))
        res[name] = r
    save(OUT / 'ci.json', dict(summary=res, latency=lat, equal_gmac=eq,
                               rule='per-image paired difference, normal 95% CI (points); margins are descriptive, not pre-registered'))
    print(json.dumps(res, indent=1))
    return res


def stationfit():
    from leaf.reproduction.experiments.equal_gmac import arms
    res = {}
    for bb in BBS:
        pts = [r for r in arms(bb) if r['gmac'] is not None and r['family'] != 'MCTF']
        y = np.array([r['ms'] for r in pts])
        X1 = np.array([[1., r['gmac']] for r in pts]); X2 = np.array([[1., r['gmac'], r['K']] for r in pts])
        sse = lambda X: float(((y - X @ np.linalg.lstsq(X, y, rcond=None)[0]) ** 2).sum())
        s1, s2, sst = sse(X1), sse(X2), float(((y - y.mean()) ** 2).sum())
        res[bb] = dict(n=len(pts), r2_gmac=1 - s1 / sst, r2_gmac_K=1 - s2 / sst, partial_r2_K=(s1 - s2) / s1)
        print(bb, {k: round(v, 3) for k, v in res[bb].items()})
    save(OUT / 'stationfit.json', res)
    return res


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'all': lastblock(); ci(); stationfit()
    else: {'lastblock': lastblock, 'ci': ci, 'stationfit': stationfit}[cmd]()
