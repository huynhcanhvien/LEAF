"""experiments/ablation_v3.py building blocks on the CPU: the planner copy with default switches reproduces the sealed
leaf_alpha4 schedules, restrictions keep the final-block station available, fixed-path edits touch only what they should,
random head groups keep the group sizes, the SALT refit on all workloads reproduces the frozen model, and probe
configs carry alpha; the discounted-station DP of station_discount.py equals the planner's DP at rho = 1."""
from common import check, finish
from collections import Counter

import numpy as np

import leaf.reproduction.experiments.ablation_and_alpha_grid as a
from leaf.model.alpha_score import config_to_alpha


def main(bb='deit_b'):
    c = a.Ctx(bb); spec = c.spec
    for b in c.budgets[:2]:
        res = c.plan(c.base['M3'], b['latency_ms'])
        sealed = c.leaf_alpha4[b['id']]
        check(f"default planner copy == sealed leaf_alpha4/{b['id']}", {k: res['config'][k] for k in ('prune_keep', 'merge_keep', 'alpha', 'groups')}
              == {k: sealed[k] for k in ('prune_keep', 'merge_keep', 'alpha', 'groups')})
    r = a.restrict(c.base['M3'], spec.depth, alphas=(.75,))
    last_ok = all(list(cv.curves) == [spec.depth - 1] for (th, al), cv in r.items() if al == 0)
    check('restrict: alpha=0 curves kept at the last block only', last_ok and any(al == .75 for _, al in r))
    sealed = c.leaf_alpha4[c.b3[0]['id']]
    nf = a.without_final(sealed, spec)
    check('without_final changes only the last-block caps', nf['prune_keep'][:-1] == sealed['prune_keep'][:-1]
          and nf['merge_keep'][:-1] == sealed['merge_keep'][:-1] and nf['alpha'] == sealed['alpha'])
    d1 = a.with_forecast(sealed, c.d, alpha=1.)
    check('with_forecast alpha=1 keeps stations and groups', set(d1['alpha'].values()) == {1.} and d1['groups'] == sealed['groups']
          and d1['prune_keep'] == sealed['prune_keep'])
    g4 = a.with_forecast(sealed, a.dconfig(bb, 'G4r32'))
    check('with_forecast G4: at most 4 groups per station', all(len(set(v)) <= 4 for v in g4['groups'].values()) and g4['rank'] == 32)
    rnd, base = a.dconfig(bb, a.PLACEBO), a.dconfig(bb)
    check('random groups keep the clustered group sizes', all(Counter(rnd.at(l)) == Counter(base.at(l)) and rnd.at(l) != base.at(l) for l in (1, 5)))
    full = a.salt_subset(bb, 10 ** 6, 0)
    check('SALT refit on all workloads == frozen model', np.allclose(full['coef'], c.model['coef'], atol=1e-8)
          and abs(full['intercept'] - c.model['intercept']) < 1e-8, f"n={full['n']}")
    job = next(j for j in a.probe_jobs(bb, 'alpha_ext') if j['alpha'] == 1.)
    check('probe config: alpha=1 pda station', job['config']['kind'] == 'pda' and set(job['config']['alpha'].values()) == {1.})
    dr = c.diffrate('DR8.7', c.base['M0'], 'M0', False)['config']
    from leaf.planner.configs import diffrate_config
    check('A1 diffrate P/M0 == released DiffRate schedule', dr == config_to_alpha(diffrate_config(spec, '8.7')))
    b = c.b3[1]
    ks = {k: len(c.plan(c.base['M3'], b['latency_ms'], fixed_k=k)['state']['sites']) for k in (2, 6)}
    check('K: fixed_k plans have exactly K stations within the SALT budget', ks == {2: 2, 6: 6}, str(ks))
    want = {'alpha_ext', 'alpha_fine', 'theta_ext'} | ({'theta_alpha_new'} if a.PHASE == 'grid' else set())
    check('probe sets: alpha_ext/alpha_fine/theta_ext (+theta_alpha_new in the grid phase), no G/r re-probes', set(a.probe_sets(bb)) == want)
    combos = {(j['theta'], j['alpha']) for n in a.probe_sets(bb) for j in a.probe_jobs(bb, n)} | {(t, al) for t in a.THETA_BASE for al in a.ALPHA_BASE}
    check('grid phase: probes + existing curves cover the full 5 theta x 7 alpha grid', a.PHASE != 'grid' or
          combos == {(t, al) for t in a.THETA_BASE + a.THETA_EXT for al in a.ALPHA_BASE + a.ALPHA_EXT + a.ALPHA_FINE})
    h = a.h_schedules(bb)['schedules']
    check('H: 30 random schedules per main budget, all under the SALT budget', len(h) == 3 * a.H_SCHEDULES
          and all(r['predicted_ms'] <= next(x['latency_ms'] for x in c.b3 if x['id'] == r['budget']) - a.MARGIN for r in h))
    # station_discount.py (L775): the DP with the station count in its state equals the planner's DP at rho = 1
    from leaf.planner.lagrange import lagrange_dp
    from leaf.planner.surrogate import JointTables
    from leaf.reproduction.ablation_models.planner_ablation_switches import lagrange_dp_discount
    jt = JointTables(c.base['M3'], spec.n_tokens, spec.depth, dim=spec.dim); times = np.random.default_rng(0).uniform(.5, 3, jt.mac.shape)
    same = all((lambda x, y: x['sites'] == y['sites'] and x['counts'] == y['counts'] and abs(x['value'] - y['value']) < 1e-9)
               (lagrange_dp(jt, times, times, mu), lagrange_dp_discount(jt, times, mu, 1.)) for mu in (0., .01, .1, 1., 10.))
    check('discounted DP at rho = 1 == lagrange_dp', same)
    res = c.plan(c.base['M3'], c.b3[1]['latency_ms'], discount=1.)
    check('discount = 1 reproduces the default planner copy', res['config'] == c.plan(c.base['M3'], c.b3[1]['latency_ms'])['config'])


if __name__ == '__main__':
    main(); finish()
