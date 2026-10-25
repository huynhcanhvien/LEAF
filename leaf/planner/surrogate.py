"""Separable surrogate F^ and the exact DP over (block, tokens, stations used).

Probe curve D_l(delta; theta, k): measured KL of removing `delta` tokens at block l alone, with merged
share theta and D weight k. The cost of a transition n_in -> n_out at block l is the telescoped curve
    f_l(n_in -> n_out) = min_{theta, k} [D_l(total - n_out; theta, k) - D_l(total - n_in; theta, k)]
(the inner minimum is exact because each (theta_t, k_t) appears in one term only). The latency enters
through a Lagrange multiplier mu: minimise sum_t f_t + mu * cost_t with exactly K stations.
Theory: artifacts/method/station-planning-problem-2026-10-12.md.
"""
import numpy as np

from leaf.planner.schedule_caps import block_gmacs, schedule


class ProbeCurve:
    """Monotone piecewise-linear D_l(delta) per block from single-cut probes {block: {delta: KL}}."""

    def __init__(self, probes, total=197):
        self.total = total; self.curves = {}
        for l, rows in probes.items():
            deltas = np.array([0] + sorted(rows), dtype=float)
            values = np.array([0.] + [rows[d] for d in sorted(rows)])
            self.curves[int(l)] = (deltas, np.maximum.accumulate(np.maximum(values, 0.)))

    def removal(self, l, delta):
        deltas, values = self.curves[l]
        if delta <= deltas[-1]: return float(np.interp(delta, deltas, values))
        slope = (values[-1] - values[-2]) / max(1., deltas[-1] - deltas[-2])
        return float(values[-1] + slope * (delta - deltas[-1]))

    def transition(self, l, n_in, n_out):
        if n_out >= n_in: return 0.
        return self.removal(l, self.total - n_out) - self.removal(l, self.total - n_in)


def grid(total=197, step=4, floor=3):
    values = list(range(total, floor - 1, -step))
    if values[-1] != floor: values.append(floor)
    return values


class JointTables:
    """Transition tables on the count grid with the inner minimum over the probed (theta, k).

    curves: {(theta, k): ProbeCurve}. kl/th/kk: cost and arg-min per (block, n_in, n_out); mac: the
    transition cost used by the DP (GMAC here; planner.latency.tables overwrites it with latency)."""

    def __init__(self, curves, total=197, depth=12, step=4, dim=768):
        self.curves = curves; self.counts = np.array(grid(total, step)); c = len(self.counts)
        self.kl = np.full((depth, c, c), np.inf); self.mac = np.zeros((depth, c, c))
        self.th = np.full((depth, c, c), .5); self.kk = np.zeros((depth, c, c))
        self.theta, self.total, self.depth = .5, total, depth
        for l in range(depth):
            for i, n_in in enumerate(self.counts):
                self.mac[l, i, i] = block_gmacs(int(n_in), total, total, dim); self.kl[l, i, i] = 0.
                for j in range(i + 1, c):
                    n_out = int(self.counts[j]); best = self.best(l, int(n_in), n_out)
                    if best is None: continue
                    self.kl[l, i, j], self.th[l, i, j], self.kk[l, i, j] = best
                    self.mac[l, i, j] = block_gmacs(int(n_in), n_out + round((int(n_in) - n_out) * best[1]), n_out, dim)

    def best(self, l, n_in, n_out):
        """(cost, theta*, k*) of cutting n_in -> n_out at block l over the probed (theta, k)."""
        out = None
        for (th, k), cv in self.curves.items():
            if l == self.depth - 1 and k > 0: continue
            if l not in cv.curves: continue
            v = cv.transition(l, n_in, n_out)
            if out is None or v < out[0] - 1e-12: out = (v, th, k)
        return out


def lookup(tables, sites, counts):
    """theta*, k* for the actual transitions of a schedule (counts need not lie on the DP grid)."""
    thetas, ks, n = [], [], tables.total
    for l, c in zip(sites, counts):
        _, th, k = tables.best(l, n, c); thetas.append(th); ks.append(k); n = c
    return thetas, ks


def solve(tables, k, mu, first_site, allowed=None, mu_g=0., gmac=None):
    """Exact DP minimiser of sum kl + mu * mac with exactly k stations at blocks >= first_site; None if infeasible."""
    t = tables; c = len(t.counts); inf = np.inf
    cost = np.full((k + 1, c), inf); cost[0, 0] = 0.
    back = np.zeros((t.depth, k + 1, c), dtype=np.int64)
    for l in range(t.depth):
        step = t.kl[l] + mu * t.mac[l]
        if mu_g:
            if gmac is None: raise ValueError('gmac table required for the second multiplier')
            step = step + mu_g * gmac[l]
        stay = cost + np.diag(step)[None]                    # keep the count
        new = stay.copy(); arg = np.tile(np.arange(c), (k + 1, 1))
        if l >= first_site and (allowed is None or l in allowed):
            off = step.copy(); np.fill_diagonal(off, inf)
            moved = cost[:-1, :, None] + off[None]           # [used, n_in, n_out]
            best_in = moved.argmin(1); best = moved.min(1)
            better = best < stay[1:]
            new[1:] = np.where(better, best, stay[1:]); arg[1:] = np.where(better, best_in, arg[1:])
        back[l] = arg; cost = new
    j = int(cost[k].argmin())
    if not np.isfinite(cost[k, j]): return None
    total_cost = float(cost[k, j]); path, used = [], k
    for l in range(t.depth - 1, -1, -1):
        i = int(back[l, used, j])
        if i != j: path.append((l, int(t.counts[j]))); used -= 1
        j = i
    if used != 0: return None
    path.reverse()
    return dict(genome=schedule([l for l, _ in path], [n for _, n in path], t.theta, t.total, t.depth),
                surrogate=total_cost, mu=mu)


def k_best(tables, k, mu, first_site=1, width=8, allowed=None, mu_g=0., gmac=None):
    """Exact k-best paths of the layered DAG, retaining width paths per (used, count).

    Ties use predecessor order, making the result deterministic. mac and gmac contain
    per-block costs; schedule constants do not affect ranking.
    """
    if width < 1 or k < 0: raise ValueError('width must be positive and k nonnegative')
    t = tables; c = len(t.counts)
    cost = np.full((k + 1, c, width), np.inf); cost[0, 0, 0] = 0.
    backs = []
    for l in range(t.depth):
        edge = t.kl[l] + mu * t.mac[l]
        if mu_g:
            if gmac is None: raise ValueError('gmac table required')
            edge = edge + mu_g * gmac[l]
        new = np.full_like(cost, np.inf)
        back = np.full((*cost.shape, 3), -1, dtype=int)
        for used in range(k + 1):
            stay = (cost[used] + edge.diagonal()[:, None]).T
            vals = stay
            if used and l >= first_site and (allowed is None or l in allowed):
                off = np.where(np.triu(np.ones((c, c), bool), 1), edge, np.inf)
                moved = (cost[used - 1, :, :, None] + off[:, None, :]).reshape(c * width, c)
                vals = np.concatenate((stay, moved), axis=0)
            idx = np.argsort(vals, axis=0, kind='stable')[:width]
            new[used] = np.take_along_axis(vals, idx, axis=0).T
            is_move = idx >= width
            back[used, :, :, 0] = (used - is_move.astype(int)).T
            back[used, :, :, 1] = np.where(is_move, (idx - width) // width, np.arange(c)[None, :]).T
            back[used, :, :, 2] = np.where(is_move, (idx - width) % width, idx).T
        cost = new; backs.append(back)
    out = []
    for flat in np.argsort(cost[k].ravel(), kind='stable')[:width]:
        j, r = divmod(int(flat), width); v = float(cost[k, j, r])
        if not np.isfinite(v): continue
        used = k; path = []
        for l in range(t.depth - 1, -1, -1):
            u, i, rank = backs[l][used, j, r]
            if i != j: path.append((l, int(t.counts[j])))
            used, j, r = int(u), int(i), int(rank)
        path.reverse()
        out.append(dict(genome=schedule([l for l, _ in path], [n for _, n in path],
                                       t.theta, t.total, t.depth), surrogate=v, mu=mu, mu_g=mu_g))
    return out


class ConditionalCurve:
    """Measured incremental KL for each block and incoming count.

    rows = {block: {n_in: {n_out: KL(with cut) - KL(upstream only)}}}.
    Interpolate outgoing counts within each row, then incoming counts. The dense
    boundary uses the original 1D curve exactly; sparse boundaries fall back to it.
    """
    def __init__(self, rows, dense):
        self.rows, self.dense, self.total = rows, dense, dense.total
        self.curves = dense.curves  # block availability used by JointTables

    def transition(self, l, n_in, n_out):
        if n_out >= n_in: return 0.
        if n_in == self.total: return self.dense.transition(l, n_in, n_out)
        rows = self.rows.get(l, {})
        anchors = []
        for ni, rr in sorted(rows.items()):
            if ni < n_out or not rr: continue
            xs = sorted(set(rr) | {ni}); ys = [0. if x == ni else rr[x] for x in xs]
            if n_out < xs[0]: continue
            anchors.append((ni, float(np.interp(n_out, xs, ys))))
        anchors.append((self.total, self.dense.transition(l, self.total, n_out)))
        if n_in < anchors[0][0]: return self.dense.transition(l, n_in, n_out)
        return float(np.interp(n_in, [x for x, _ in anchors], [y for _, y in anchors]))


class CorrectedCurve:
    """A station-local learned correction that preserves exact DP separability."""
    def __init__(self, base, coef, depth):
        self.base, self.coef, self.depth = base, np.asarray(coef), depth
        self.total, self.curves = base.total, base.curves

    @staticmethod
    def features(l, n_in, n_out, total, depth):
        r = (n_in - n_out) / n_in
        return np.array([r * r, r * r * n_in / total, r * r * l / depth])

    def transition(self, l, n_in, n_out):
        return self.base.transition(l, n_in, n_out) + float(
            self.features(l, n_in, n_out, self.total, self.depth) @ self.coef)
