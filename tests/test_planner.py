"""Planner on synthetic probes (CPU): exact DP equals brute force, latency model features, every proposal and
search move stays in the latency window, and a fake measured KL drives the path and local search.
    python tests/test_planner.py"""
from common import check, finish
import itertools
import numpy as np

from leaf.backbones import SPECS
from leaf.planner.surrogate import ConditionalCurve, JointTables, ProbeCurve, best_paths, grid, lookup, solve
from leaf.latency.profile_model import ProfileLatencyModel, close_loop, features, fit_model
from leaf.planner.schedule_caps import analytic_gmacs, canonical, full_cost, schedule, station_view, token_counts
from leaf.planner.probes import ALPHAS, LTOL, MARGIN, Planner, probe_counts

spec = SPECS['deit_b']; T = spec.n_tokens


def synthetic_curves():
    """D_l(delta; theta, alpha): convex in delta, cheaper late, best alpha = 2/3 early, theta = 1 in the middle."""
    out = {}
    for th in (0., .5, 1.):
        for i, a in enumerate(ALPHAS):
            rows = {}
            for l in range(1, 12):
                w = (1.5 - l / 12) * (1 - .1 * min(i, 2) + .05 * max(i - 2, 0)) * (1 - .1 * th * (4 <= l <= 8))
                rows[l] = {T - n: w * ((T - n) / T) ** 2 for n in probe_counts(T)}
            if a and 11 in rows: del rows[11]
            out[(th, a)] = ProbeCurve(rows)
    return out


def fake_profile():
    g = sorted({197, 3} | set(range(193, 2, -6)), reverse=True)
    return dict(grid=g, ms={f'{bs}/{kind}': [bs * (0.02 + 0.0004 * n + (1e-6 * n * n if kind == 'attn' else 0)) for n in g]
                            for bs in (1, 32, 128) for kind in ('attn', 'mlp')})


class FakeRunner:
    """Measured KL = surrogate + interaction penalty per station + noise-free per-image spread."""
    def __init__(self, planner_ref): self.p = planner_ref; self.calls = 0

    def kl(self, cfg, split='half'):
        self.calls += 1
        g = canonical(dict(p=cfg['prune_keep'], m=cfg['merge_keep']), T); sites, counts, th = station_view(g, T)
        st = dict(sites=sites, counts=counts, thetas=th, alphas=[cfg['alpha'].get(str(l), 0.) for l in sites])
        base = self.p[0].fhat(st) + 0.003 * len(sites)                 # fewer stations are better than F^ says
        rng = np.random.default_rng(0); return base * (1 + 0.01 * rng.standard_normal(500))


class FakeD:
    G, rank, triton = 16, 32, True
    def at(self, l): return [0] * ((11 - l) * 12)


def main():
    # --- DP vs brute force on a small grid
    cv = ProbeCurve({l: {d: (1.2 - l / 10) * (d / 20) ** 2 for d in range(4, 20, 4)} for l in range(4)}, total=20)
    jt = JointTables({(.5, 0.): cv}, total=20, depth=4)
    for mu in (0., 1e2, 1e4):
        for K in (1, 2):
            s = solve(jt, K, mu, 1); counts = [int(c) for c in jt.counts]; best = None
            for sites in itertools.combinations(range(1, 4), K):
                for cs in itertools.combinations(counts[1:], K):
                    if list(cs) != sorted(cs, reverse=True): continue
                    n, v = 20, 0.
                    for l in range(4):
                        if l in sites: c = cs[sites.index(l)]; i, j = counts.index(n), counts.index(c); v += jt.kl[l, i, j] + mu * jt.mac[l, i, j]; n = c
                        else: i = counts.index(n); v += mu * jt.mac[l, i, i]
                    best = v if best is None else min(best, v)
            check(f'DP == brute force (mu={mu:g}, K={K})', s is not None and abs(s['surrogate'] - best) < 1e-9, f"{s and s['surrogate']:.6g} vs {best:.6g}")
    # Exact ordered paths, including a distinct second resource on every edge.
    rng = np.random.default_rng(71); gm = rng.random(jt.mac.shape)
    for K in (0, 1, 2):
        enumerated = []
        for sites in itertools.combinations(range(1, jt.depth), K):
            for idx in itertools.combinations(range(1, len(jt.counts)), K):
                n, v, path = 0, 0., []
                for l in range(jt.depth):
                    j = idx[sites.index(l)] if l in sites else n
                    v += jt.kl[l, n, j] + .7 * jt.mac[l, n, j] + .13 * gm[l, n, j]
                    if j != n: path.append((l, int(jt.counts[j])))
                    n = j
                enumerated.append((v, path))
        enumerated.sort(key=lambda x: x[0]); rs = best_paths(jt, K, .7, width=8, mu_g=.13, gmac=gm)
        check(f'best paths, two multipliers == brute force K={K}',
              np.allclose([r['surrogate'] for r in rs], [v for v, _ in enumerated[:8]]))
        check(f'best paths are unique K={K}', len({str(r['genome']) for r in rs}) == len(rs))
        r = solve(jt, K, .7, 1, mu_g=.13, gmac=gm)
        check(f'DP two multipliers == brute force K={K}', abs(r['surrogate'] - enumerated[0][0]) < 1e-10)
    cond = ConditionalCurve({1: {12: {4: .12, 8: .04}}}, cv)
    check('2D dense boundary == 1D', all(cond.transition(1, 20, n) == cv.transition(1, 20, n) for n in (3, 7, 12, 19)))
    check('2D measured knots and diagonal', cond.transition(1, 12, 4) == .12 and cond.transition(1, 12, 12) == 0)
    st = dict(sites=[1], counts=[100], thetas=[.5], alphas=[0.])
    r = close_loop(st, 10., lambda s: 5. + .04 * s['counts'][0])
    check('closed loop accepts only measured window', r['state'] is not None and 9.92 <= r['ms'] <= 9.98 and len(r['measurements']) <= 6)
    r = close_loop(st, 10., lambda s: 20.)
    check('closed loop rejects unreachable budget', r['state'] is None and not r['feasible'] and len(r['measurements']) <= 6)
    r = close_loop(st, 10., lambda s: 9.9, fallback_window=(.15, .02))
    check('wider fallback is explicit and still measured', r.get('fallback') and r['ms'] == 9.9 and
          r['state'] in [x['state'] for x in r['measurements']] and len(r['measurements']) <= 6)
    quantized = dict(sites=[1,5,6,8,11], counts=[158,126,102,62,3], thetas=[0.] * 5, alphas=[0.] * 5)
    r = close_loop(quantized, 21.25, lambda s: 21.065 + .02 * sum(w * (a-b) for w,a,b in
                   zip([4,1,2,3,1], s['counts'], quantized['counts'])), fallback_window=(.15,.02))
    check('integer refinement escapes simultaneous count jumps within six measurements',
          r.get('discrete_refine') and 21.17 <= r['ms'] <= 21.23 and len(r['measurements']) <= 6)
    fixed_st = dict(sites=[1, 5], counts=[180, 100], thetas=[0., .5], alphas=[0., 0.])
    r = close_loop(fixed_st, 10., lambda s: 5. + .04 * s['counts'][1], fixed_counts={0:180})
    check('closed loop preserves early-count ceiling for W4', r['state'] is not None and
          all(x['state']['counts'][0] == 180 for x in r['measurements']))
    gg = schedule([1, 5], [100, 50], .5)
    a = full_cost(gg['p'], gg['m'], {}); b = full_cost(gg['p'], gg['m'], {'1': 2 / 3})
    check('full cost counts D only at stations with alpha > 0', a['d_gmac'] == 0 and b['d_gmac'] > 0 and b['full'] > a['full'])
    small = full_cost([197] * 12, [197] * 12, d=384)
    check('full cost corrects the actual RGB patch16 embedding on DeiT-S',
          abs(small['patch_adjustment_gmac'] - 196 * 384 * (768 - 384) / 1e9) < 1e-12)
    # --- schedule helpers
    g = schedule([1, 5, 11], [150, 80, 5], [0., 1., .5], T, 12)
    check('schedule <-> station_view round trip', station_view(g, T)[:2] == ([1, 5, 11], [150, 80, 5]))
    check('token_counts', token_counts(g, T)[1][11] == 5 and token_counts(g, T)[0][1] == T)
    check('dense GMAC of DeiT-B ~ 17.6', abs(analytic_gmacs([T] * 12, [T] * 12) - 17.56) < .05)
    # --- latency model: exact recovery of a linear ground truth
    prof = fake_profile(); rng = np.random.default_rng(1); rows = []
    truth = ProfileLatencyModel(dict(fit={'32': dict(chosen='profile', profile=dict(coef=[1.0, 1.1, .17, .05]))}, profile=prof), spec, 32)
    check('profile latency honors explicit zero-D schedule',
          truth(g['p'], g['m'], alphas={}) == truth(g['p'], g['m'], nd=0) and
          truth(g['p'], g['m'], alphas={'1': 2 / 3}) == truth(g['p'], g['m'], nd=1))
    for _ in range(40):
        K = int(rng.integers(1, 12)); sites = sorted(rng.choice(np.arange(1, 12), K, replace=False).tolist())
        counts = sorted(rng.integers(3, 196, K).tolist(), reverse=True); gg = schedule(sites, counts, .5, T, 12)
        nd = int(rng.integers(0, len(gg['sites']) + 1)); rows.append(dict(p=gg['p'], m=gg['m'], nd=nd, ms=truth(gg['p'], gg['m'], nd)))
    fm = fit_model(rows, prof, spec, 32)
    check('latency fit recovers coefficients', np.allclose(fm['profile']['coef'], [1.0, 1.1, .17, .05], atol=1e-6) and fm['chosen'] == 'profile')
    coef = [1., 1.1, .17, .05, .08, .01, .03]
    rich = ProfileLatencyModel(dict(fit={'32':dict(chosen='profile_station', profile_station=dict(coef=coef))}, profile=prof), spec)
    row = dict(p=gg['p'], m=gg['m'], alpha={'1': 2 / 3}, nd=int(1 in gg['sites']))
    check('station features agree with latency prediction and actual active D',
          abs(np.array(features(row, 'profile_station', prof, 32, spec)) @ coef - rich(gg['p'], gg['m'], alphas=row['alpha'])) < 1e-10)
    # --- planner on synthetic curves
    lat = ProfileLatencyModel(dict(fit={'32': fm}, profile=prof), spec, 32); ref = []
    fr = FakeRunner(ref); p = Planner(fr, spec, lat, synthetic_curves(), FakeD(), log=lambda s: None); ref.append(p)
    restricted = Planner(fr, spec, lat, synthetic_curves(), FakeD(), keys=[(.5, 2 / 3)], log=lambda s: None)
    check('fixed-alpha ablation retains P-only final station', restricted.jt.best(11, T, 100)[2] == 0 and
          restricted.jt.best(5, T, 100)[2] == 2 / 3)
    L = lat([T] * 12, [T] * 12, 0) * .6
    for K in (3, 6, 11):
        r = p.plan_K(K, L - MARGIN)
        # None is legal (no K-station schedule fits); otherwise the schedule must sit in the window.
        check(f'plan_K K={K}: none or within the latency window', (K < 11 and r is None) or
              (r is not None and abs(r['T'] - (L - MARGIN)) <= LTOL and len(r['state']['sites']) == K))
    st = p.plan_K(11, L - MARGIN)['state']
    moves = p.merge_moves(st, L - MARGIN)
    check('merge moves: K-1 stations, latency kept', moves and all(len(s['sites']) == 10 and p.lat_ok(s, L - MARGIN) for _, s in moves))
    for t in range(len(st['sites'])):
        nb = p.neighbours(st, t, L - MARGIN, dict(theta=.25, alpha=.125), ('theta', 'alpha', 'tok', 'site'))
        if not all(p.ok_state(s, L - MARGIN) for _, s in nb): check(f'neighbours of station {t} keep the window', False); break
    else: check('every local-search neighbour keeps the latency window', True)
    p.kl_odd = lambda s: float(fr.kl(p.cfg(s)).mean())
    res = p.run(L)
    check('path_dp_ls: fewer stations than the DP start, inside the window', len(res['state']['sites']) < 11 and p.lat_ok(res['state'], L - MARGIN),
          f"K={len(res['state']['sites'])}, {fr.calls} evaluations")
    check('last station never uses D', all(a == 0 for l, a in zip(res['state']['sites'], res['state']['alphas']) if l == 11))
    # The A6 search-method checks (archived search comparison) stay in the research repository.


if __name__ == '__main__':
    main(); finish()
