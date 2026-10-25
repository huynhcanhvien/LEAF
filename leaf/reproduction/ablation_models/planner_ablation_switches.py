"""Copy of leaf.latency.salt.plan_with_salt with switches for the planner ablation (A4); the original is unchanged.

rescale=False returns DP grid solutions as they are (no keep-ratio rescaling by eta);
factors=(1.,) keeps only the solution at the bisected price mu_T* instead of seven multiples;
fixed_k=K constrains the DP to exactly K stations (leaf.planner.lagrange.lagrange_dp, fixed_k branch);
discount=rho charges the k-th station rho**(k-1) times its probed damage, in the DP (station count in the state, still
exact) and in the choice among candidates (leaf/reproduction/experiments/station_discount.py, mentor comment L775).
With the defaults the result equals plan_with_salt (tests/test_ablation_and_alpha_grid.py).
"""
import numpy as np

FACTORS = (.5, .7, .85, 1., 1.15, 1.3, 1.6)


def lagrange_dp_discount(jt, times, mu_t, rho):
    """Exact DP of sum_k rho**(k-1) f_k + mu_T*T: like leaf.planner.lagrange.lagrange_dp with the station count k in
    the state, so that the damage of the k-th station can be discounted."""
    depth, width = jt.depth, len(jt.counts); kmax = depth - 1
    upper = np.triu(np.ones((width, width), bool), 1)
    cost = np.full((width, kmax + 1), np.inf); cost[0, 0] = 0.; back = []
    for l in range(depth):
        new = cost + mu_t * np.diag(times[l])[:, None]                  # no station: i == j, k unchanged
        arg = np.stack([np.broadcast_to(np.arange(width)[:, None], cost.shape), np.broadcast_to(np.arange(kmax + 1), cost.shape)], -1).copy()
        if l > 0:
            for k in range(1, kmax + 1):
                edge = np.where(upper, rho ** (k - 1) * jt.kl[l] + mu_t * times[l], np.inf)
                values = cost[:, k - 1][:, None] + edge
                i = values.argmin(0); v = values[i, np.arange(width)]
                better = v < new[:, k]
                new[better, k] = v[better]; arg[better, k] = np.stack([i[better], np.full(better.sum(), k - 1)], -1)
        cost = new; back.append(arg)
    j, k = np.unravel_index(np.argmin(cost), cost.shape)
    if not np.isfinite(cost[j, k]): return None
    value = float(cost[j, k]); path = []
    for l in reversed(range(depth)):
        i, prev_k = back[l][j, k]
        if j > i: path.append((l, int(jt.counts[j])))
        j, k = int(i), int(prev_k)
    path.reverse()
    return dict(sites=[l for l, _ in path], counts=[n for _, n in path], value=value, mu_t=float(mu_t), mu_g=0.)


def plan_proxy_variant(curves, spec, dcfg, cost, budget, rescale=True, factors=FACTORS, margin=.1, fixed_k=None, discount=None):
    from leaf.planner.surrogate import JointTables, lookup
    from leaf.planner.lagrange import lagrange_dp
    from leaf.reproduction.ablation_models.merge_operator_planner import norm_config
    jt = JointTables(curves, spec.n_tokens, spec.depth, dim=spec.dim)
    times = np.zeros_like(jt.mac)
    for l in range(spec.depth):
        for i, ni in enumerate(jt.counts):
            for j in range(i, len(jt.counts)):
                ni, nj = int(ni), int(jt.counts[j]); cut = j > i
                pp = nj + round((ni - nj) * jt.th[l, i, j]) if cut else ni
                active = cut and l < spec.depth - 1 and jt.kk[l, i, j] > 0
                groups = len(set(dcfg.at(l))) if active else 0
                times[l, i, j] = cost.edge(l, ni, pp, nj, active, groups)

    def state(sites, counts):
        th, ks = lookup(jt, sites, counts)
        return dict(sites=list(sites), counts=list(counts), thetas=th, ks=ks)

    def cfg(st): return norm_config(st, spec, dcfg, 'M3')

    def fhat(st):
        n = spec.n_tokens; value = 0.
        for i, (l, c, th, k) in enumerate(zip(st['sites'], st['counts'], st['thetas'], st['ks'])):
            value += (1. if discount is None else discount ** i) * jt.curves[(th, k)].transition(l, n, c); n = c
        return float(value)

    def dp(mu):
        if discount is None: return lagrange_dp(jt, times, times, mu, fixed_k=fixed_k)
        return lagrange_dp_discount(jt, times, mu, discount)

    def as_is(raw):
        st = state(raw['sites'], raw['counts']); config = cfg(st); t = cost(config)
        if t > budget - margin: return None
        return dict(state=st, config=config, predicted_ms=t, fhat=fhat(st))

    def project(raw):
        sites = raw['sites']; n = spec.n_tokens; ratios = []
        for c in raw['counts']: ratios.append(c / n); n = c

        def at(eta):
            n = spec.n_tokens; counts = []
            for r in ratios:
                c = max(3, min(n - 1, round(n * r ** eta)))
                if c >= n: return None
                counts.append(c); n = c
            st = state(sites, counts); config = cfg(st); t = cost(config)
            return dict(state=st, config=config, predicted_ms=t, fhat=fhat(st))
        samples = [(float(e), at(float(e))) for e in np.geomspace(.001, 20, 96)]
        hit = next((i for i, (_, v) in enumerate(samples) if v is not None and v['predicted_ms'] <= budget - margin), None)
        if hit is None: return None
        hi, best = samples[hit]; lo = samples[hit - 1][0] if hit else .001
        for _ in range(40):
            mid = (lo + hi) / 2; v = at(mid)
            if v is not None and v['predicted_ms'] <= budget - margin: hi = mid; best = v
            else: lo = mid
        return best

    lo, hi = 0., 1.; candidates = []; seen = set()
    for _ in range(24):
        raw = dp(hi)
        if raw is None: break
        if cost(cfg(state(raw['sites'], raw['counts']))) <= budget: break
        hi *= 4
    for _ in range(32):
        mid = (lo + hi) / 2; raw = dp(mid)
        if raw is None or cost(cfg(state(raw['sites'], raw['counts']))) > budget: lo = mid
        else: hi = mid
    for factor in factors:
        raw = dp(hi * factor)
        if raw is None: continue
        result = project(raw) if rescale else as_is(raw)
        if result is None: continue
        key = str(result['state'])
        if key not in seen: seen.add(key); candidates.append(result)
    if not candidates: return dict(infeasible=True, budget_ms=budget)
    best = min(candidates, key=lambda v: (v['fhat'], v['predicted_ms'], str(v['state'])))
    return dict(best, budget_ms=budget, n_candidates=len(candidates), rescale=rescale, factors=list(factors), fixed_k=fixed_k,
                **({} if discount is None else dict(discount=discount)))


def plan_latency_variant(curves, spec, dcfg, model, budget, mode='M3', use_d=True, **kw):
    """leaf.planner.plan.plan_latency with plan_proxy_variant."""
    from leaf.planner.gmac import config_cost
    from leaf.planner.plan import OperatorCost, operator_config
    cv = curves if use_d else {key: v for key, v in curves.items() if key[1] == 0}
    cost = OperatorCost(model, spec, dcfg, mode)
    result = plan_proxy_variant(cv, spec, dcfg, cost, budget, **kw)
    if result.get('infeasible'): return dict(result, status='no_feasible_candidate')
    result['config'] = operator_config(result['config'], mode, use_d)
    result.update(status='planned', gmac=config_cost(result['config'], spec))
    return result
