"""path_dp_ls: the latency-budget planner (artifacts/method/latency-schedule-planner-2026-10-13.md §2).

    probe one station (once per backbone)  ->  F^ with the inner min over (theta, alpha)
    DP + Lagrange at L' = L - MARGIN       ->  schedule with depth-1 stations
    merge adjacent stations on measured KL ->  path K = depth-1 ... 3 (freed latency goes to tokens)
    choose K on calib odd                  ->  local search on calib half (theta, alpha, tokens, site; z <= -2)

Planner(runner, spec, lat, curves, dcfg, ...) holds everything one backbone needs; options switch the
components off for the ablations (fixed theta, shared alpha, fixed K, no refinement, GMAC cost).
"""
import copy
import json

import numpy as np

from leaf.evaluation.data import paired_z
from leaf.planner.configs import state_caps, config_from_state
from leaf.planner.surrogate import JointTables, ProbeCurve, lookup, solve
from leaf.planner.schedule_caps import analytic_gmacs, canonical, schedule, station_view

LTOL = 0.05     # ms: half-width of the latency window around the planning target
MARGIN = 0.1    # ms: plan at L - MARGIN so a schedule lands in [L - 0.15, L - 0.05], never above the budget
FIRST = 1       # earliest station block


# ---------------------------------------------------------------- probes
def probe_counts(total):
    """Tokens after the single probed station (197 tokens: 189, 173, 149, ..., 29, 3)."""
    return tuple(n for n in (total - 8, total - 24, total - 48, total - 72, total - 96, total - 120, total - 144, total - 168) if n > 3) + (3,)


THETAS = (0., .5, 1.)
ALPHAS = (0., 1 / 2, 2 / 3, 3 / 4)


def probe_jobs(spec, thetas=THETAS, alphas=ALPHAS):
    for l in range(1, spec.depth):
        for n in probe_counts(spec.n_tokens):
            for th in thetas:
                for a in (alphas if l < spec.depth - 1 else (0.,)):
                    yield l, n, th, a


def probe_state(l, n, th, a):
    return dict(sites=[l], counts=[n], thetas=[th], alphas=[a])


def probe_curves(runner, spec, dcfg, thetas=THETAS, alphas=ALPHAS, idx=None):
    """{(theta, alpha): ProbeCurve} from cached probe runs on calib half (idx: per-image subset)."""
    rows = {}
    for l, n, th, a in probe_jobs(spec, thetas, alphas):
        arr = runner.cached(config_from_state(probe_state(l, n, th, a), spec, dcfg), 'half')
        if arr is None: continue
        v = arr['kl'] if idx is None else arr['kl'][idx]
        rows.setdefault((th, a), {}).setdefault(l, {})[spec.n_tokens - n] = float(v.mean())
    return {key: ProbeCurve(v, spec.n_tokens) for key, v in rows.items()}


class Planner:
    def __init__(self, runner, spec, lat, curves, dcfg, keys=None, gmac_cost=False, log=print, ltol=LTOL, margin=MARGIN):
        """gmac_cost: the budget and every schedule cost are GMAC (= the 'GFLOPs' of ViT papers) instead of ms;
        ltol / margin are then in GMAC (the window is [B - margin - ltol, B - margin + ltol])."""
        self.runner, self.spec, self.lat, self.dcfg, self.log = runner, spec, lat, dcfg, log
        self.ltol, self.margin, self.gmac = ltol, margin, gmac_cost
        self.cost = (lambda p, m: analytic_gmacs(p, m, spec.n_tokens, spec.dim)) if gmac_cost else lat
        self.jt = self.tables(curves, keys)

    # ------------------------------------------------------------ tables
    def tables(self, curves, keys=None):
        """Joint tables restricted to (theta, alpha) in keys; DP transition cost = latency (+ station constant)."""
        last = self.spec.depth - 1; total = self.spec.n_tokens; sub = {}
        for key, v in curves.items():
            if keys is not None and key not in keys: continue
            if last not in v.curves and (key[0], 0.) in curves:    # the last block is probed with alpha = 0 only
                v = copy.copy(v); v.curves = dict(v.curves); v.curves[last] = curves[(key[0], 0.)].curves[last]
            sub[key] = v
        # Restricted-alpha ablations still need a P-only final station. Keep that
        # boundary explicitly rather than labelling its cost with a positive alpha.
        for th, _ in list(sub):
            key = (th, 0.)
            if key not in sub and key in curves and last in curves[key].curves:
                final = copy.copy(curves[key]); final.curves = {last: curves[key].curves[last]}
                sub[key] = final
        jt = JointTables(sub, total, self.spec.depth, dim=self.spec.dim)
        if self.cost is not self.lat: return jt
        for l in range(jt.depth):
            for i, n_in in enumerate(jt.counts):
                for j in range(i, len(jt.counts)):
                    n_out = int(jt.counts[j])
                    if j == i: jt.mac[l, i, i] = self.lat.block(int(n_in), total, total)
                    elif np.isfinite(jt.kl[l, i, j]):
                        p = n_out + round((int(n_in) - n_out) * jt.th[l, i, j]); jt.mac[l, i, j] = self.lat.block(int(n_in), p, n_out) + self.lat.cst
        return jt

    # ------------------------------------------------------------ helpers
    def state_caps(self, st): return state_caps(st, self.spec)
    def cfg(self, st): return config_from_state(st, self.spec, self.dcfg)
    def T(self, st): g = self.state_caps(st); return self.cost(g['p'], g['m'])
    def kl_half(self, st): return self.runner.kl(self.cfg(st), 'half')
    def kl_odd(self, st): return float(self.runner.kl(self.cfg(st), 'odd').mean())

    def state(self, sites, counts):
        th, alphas = lookup(self.jt, sites, counts); return dict(sites=list(sites), counts=list(counts), thetas=list(th), alphas=list(alphas))

    def fit_latency(self, sites, counts, thetas, L, tol=None):
        tol = self.ltol if tol is None else tol
        """Scale every station's keep ratio by a common exponent until the cost is within tol of L."""
        total = self.spec.n_tokens; ratios, n = [], total
        for c in counts: ratios.append(max(1e-6, min(1., c / n))); n = c

        def build(s):
            out, n = [], total
            for r in ratios: c = max(3, min(n - 1, round(n * r ** s))); out.append(c); n = c
            return schedule(sites, out, thetas, total, self.spec.depth)
        cost = lambda g: self.cost(g['p'], g['m'])
        lo, hi = 1e-3, 20.
        if cost(build(lo)) < L - tol or cost(build(hi)) > L + tol: return None
        for _ in range(60):
            mid = (lo + hi) / 2; g = build(mid); c = cost(g)
            if abs(c - L) <= tol and len(g['sites']) == len(sites): return g
            if c > L: lo = mid
            else: hi = mid
        return None

    def fhat(self, st):
        n, s = self.spec.n_tokens, 0.
        for l, c in zip(st['sites'], st['counts']):
            b = self.jt.best(l, n, c); s += b[0] if b else np.inf; n = c
        return s

    # ------------------------------------------------------------ DP proposal
    def plan_K(self, K, L):
        """Bisection on mu (exact DP) -> latency fit -> re-lookup (theta*, alpha*); best F^ over a mu sweep."""
        jt, lo, hi = self.jt, 0., 1.
        while True:
            s = solve(jt, K, hi, FIRST)
            if s is None: return None
            if self.cost(s['genome']['p'], s['genome']['m']) <= L or hi > 1e7: break
            hi *= 4
        for _ in range(50):
            mid = (lo + hi) / 2; s = solve(jt, K, mid, FIRST)
            if self.cost(s['genome']['p'], s['genome']['m']) > L: lo = mid
            else: hi = mid
        best = None
        for f in (1., .7, .85, 1.15, 1.3, .5, 1.6):
            s = solve(jt, K, hi * f, FIRST)
            if s is None: continue
            sites, counts, _ = station_view(s['genome'], self.spec.n_tokens)
            if len(sites) != K: continue
            th, alphas = lookup(jt, sites, counts); g = None
            for _ in range(4):
                g = self.fit_latency(sites, counts, th, L)
                if g is None: break
                _, counts, _ = station_view(g, self.spec.n_tokens); th2, alphas = lookup(jt, sites, counts)
                if th2 == th: break
                th = th2
            if g is None: continue
            g = schedule(sites, counts, th, self.spec.n_tokens, self.spec.depth)
            if abs(self.cost(g['p'], g['m']) - L) > self.ltol or len(g['sites']) != K: continue
            st = dict(sites=list(sites), counts=list(counts), thetas=list(th), alphas=list(alphas)); f_hat = self.fhat(st)
            if best is None or f_hat < best['fhat']: best = dict(state=st, fhat=f_hat, T=self.cost(g['p'], g['m']))
        return best

    # ------------------------------------------------------------ station-merge path
    def lat_ok(self, st, L): return abs(self.T(st) - L) <= self.ltol

    def merge_moves(self, st, L):
        out, K = [], len(st['sites'])
        for t in range(K - 1):
            for where in (t, t + 1):
                sites = st['sites'][:t] + [st['sites'][where]] + st['sites'][t + 2:]
                counts = st['counts'][:t] + [st['counts'][t + 1]] + st['counts'][t + 2:]
                th, _ = lookup(self.jt, sites, counts); g = self.fit_latency(sites, counts, th, L)
                if g is None: continue
                s2, c2, _ = station_view(g, self.spec.n_tokens)
                if len(s2) != K - 1: continue
                s3 = self.state(s2, c2)
                if self.lat_ok(s3, L): out.append((f'merge{t}@{st["sites"][where]}', s3))
        return out

    def merge_path(self, start, L, K_min=3):
        st = start; path = [dict(K=len(st['sites']), state=st, kl=float(self.kl_half(st).mean()), move='start')]
        while len(st['sites']) > K_min:
            best = None
            for name, s2 in self.merge_moves(st, L):
                v = float(self.kl_half(s2).mean())
                if best is None or v < best[0]: best = (v, name, s2)
            if best is None: break
            st = best[2]; path.append(dict(K=len(st['sites']), state=st, kl=best[0], move=best[1]))
            self.log(json.dumps(dict(K=len(st['sites']), move=best[1], kl=round(best[0], 5), sites=st['sites'])))
        return path

    # ------------------------------------------------------------ local search on the measured KL
    def ok_state(self, st, L):
        total, last = self.spec.n_tokens, self.spec.depth - 1
        if any(c >= p for p, c in zip([total] + st['counts'][:-1], st['counts'])) or min(st['counts']) < 3: return False
        if any(b <= a for a, b in zip(st['sites'], st['sites'][1:])) or st['sites'][0] < FIRST or st['sites'][-1] > last: return False
        g = self.state_caps(st)
        return len(g['sites']) == len(st['sites']) and abs(self.cost(g['p'], g['m']) - L) <= self.ltol

    def compensate(self, st, idx, L):
        """Adjust counts[idx] by the smallest |delta| that brings the cost back into the window."""
        for d in sorted(range(-48, 49), key=abs):
            s2 = dict(st, counts=list(st['counts'])); s2['counts'][idx] += d
            if self.ok_state(s2, L): return s2
        return None

    def neighbours(self, st, t, L, steps, moves):
        out, last = [], self.spec.depth - 1
        for name, key, step, lo, hi in (('theta', 'thetas', steps['theta'], 0., 1.), ('alpha', 'alphas', steps['alpha'], 0., 1.)):
            if name not in moves or (key == 'alphas' and st['sites'][t] >= last): continue
            for sgn in (-1, 1):
                s2 = {k: list(v) for k, v in st.items()}; s2[key][t] = float(np.clip(s2[key][t] + sgn * step, lo, hi))
                if s2[key][t] == st[key][t]: continue
                if not self.ok_state(s2, L): s2 = self.compensate(s2, t, L)
                if s2 is not None: out.append((f'{name}{sgn:+d}', s2))
        K = len(st['sites']); u = t + 1 if t + 1 < K else t - 1
        for sgn in (-1, 1):
            if 'tok' in moves:
                s2 = {k: list(v) for k, v in st.items()}; s2['counts'][t] += 4 * sgn
                s2 = self.compensate(s2, u, L) if not self.ok_state(s2, L) else s2
                if s2 is not None: out.append((f'tok{sgn:+d}', s2))
            if 'site' in moves:
                s3 = {k: list(v) for k, v in st.items()}; s3['sites'][t] += sgn
                if 1 <= s3['sites'][t] <= last and len(set(s3['sites'])) == K and s3['sites'] == sorted(s3['sites']):
                    s3 = s3 if self.ok_state(s3, L) else self.compensate(s3, t, L)
                    if s3 is not None: out.append((f'site{sgn:+d}', s3))
        return out

    def local_search(self, st, L, moves=('theta', 'alpha', 'tok', 'site'), max_rounds=4):
        """Cyclic pattern search over stations; accept the best neighbour if paired z <= -2 and >= 0.5% better.
        A round without acceptance halves the continuous steps; stop at the floor or after max_rounds."""
        steps = dict(theta=.25, alpha=.125); cur = self.kl_half(st); hist = []
        for rnd in range(max_rounds):
            accepted = False
            for t in range(len(st['sites'])):
                best = None
                for name, s2 in self.neighbours(st, t, L, steps, moves):
                    v = self.kl_half(s2)
                    if v is None: continue
                    z = paired_z(v, cur); gain = 1 - v.mean() / cur.mean()
                    if best is None or z < best[0]: best = (z, gain, name, s2, v)
                if best and best[0] <= -2 and best[1] >= .005:
                    st, cur = best[3], best[4]; accepted = True
                    hist.append(dict(round=rnd, station=t, move=best[2], z=best[0], gain=best[1], kl=float(cur.mean()), state=st))
                    self.log(json.dumps(dict(round=rnd, t=t, move=best[2], z=round(best[0], 2), kl=round(float(cur.mean()), 5))))
            if not accepted:
                if steps['theta'] <= .0625: break
                steps = dict(theta=steps['theta'] / 2, alpha=steps['alpha'] / 2)
        return st, hist

    # ------------------------------------------------------------ the whole method at one budget
    def run(self, L_nominal, K_start=None, refine=True, choose_K='odd', moves=('theta', 'alpha', 'tok', 'site'), fixed_K=None):
        """path_dp_ls at budget L_nominal (ms). Returns dict(state, start, path, history, L).

        K_start: stations of the DP start (default depth-1). refine=False: DP only, K = argmin F^ (ablation A5).
        fixed_K: keep exactly this K (ablations A2/A4). choose_K: 'odd' (KL on calib odd) or 'fhat'."""
        L = L_nominal - self.margin; last = self.spec.depth - 1
        if not refine:
            Ks = [fixed_K] if fixed_K else range(3, last + 1)
            rs = [r for r in (self.plan_K(K, L) for K in Ks) if r]
            best = min(rs, key=lambda r: r['fhat']); return dict(L=L, state=best['state'], start=best['state'], path=[], history=[])
        start = self.plan_K(fixed_K or K_start or last, L)
        if start is None: raise RuntimeError(f'no feasible DP schedule at L={L:.2f}')
        start = start['state']
        if fixed_K: path = [dict(K=fixed_K, state=start, move='start')]; pick = start
        else:
            path = self.merge_path(start, L)
            pick = min(path, key=lambda r: self.kl_odd(r['state']) if choose_K == 'odd' else self.fhat(r['state']))['state']
        final, hist = self.local_search(pick, L, moves=moves)
        return dict(L=L, state=final, start=pick, path=path, history=hist)
