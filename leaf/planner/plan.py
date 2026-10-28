"""Operator-aware adapters for the unchanged paper DP and frozen latency features."""
from copy import deepcopy
import numpy as np
from leaf.planner.configs import config_from_state, ordered
from leaf.planner.gmac import config_cost
from leaf.backbones import head_gmac
from leaf.latency.salt import features, edge_features, constants, plan_with_salt


def operator_config(cfg, mode='M3', use_d=True):
    out = deepcopy(cfg)
    if mode == 'M0':
        out['base'] = 'pd'
        out.pop('merge', None)
    elif mode == 'M3':
        out.update(base='pdm', merge=dict(mode='M3', kernel='triton_norm'))
    else:
        raise ValueError(mode)
    if not use_d:
        out.update(alpha={}, groups={}, rank=None, triton=True)
    return ordered(out)


class OperatorCost:
    def __init__(self, model, spec, dcfg, mode='M3'):
        self.model, self.spec, self.dcfg, self.mode = model, spec, dcfg, mode
        self.names = model['feature_names']
        self.weights = np.asarray(model['coef'])
        c = constants(spec)
        self.c0 = float(model['intercept'] + sum(c.get(k, 0) * v for k, v in zip(self.names, self.weights)))

    def edge(self, l, n, p, m, active=False, groups=16):
        f = edge_features(self.spec, l, n, p, m, active, groups, self.dcfg.rank, self.mode == 'M3')
        return float(sum(f.get(k, 0) * v for k, v in zip(self.names, self.weights)))

    def __call__(self, cfg):
        f = features(operator_config(cfg, self.mode), self.spec)
        return float(self.model['intercept'] + sum(f.get(k, 0) * v for k, v in zip(self.names, self.weights)))


def plan_latency(curves, spec, dcfg, model, budget, mode='M3', use_d=True):
    cv = curves if use_d else {key: v for key, v in curves.items() if key[1] == 0}
    cost = OperatorCost(model, spec, dcfg, mode)
    result = plan_with_salt(cv, spec, dcfg, cost, budget)
    if result.get('infeasible'):
        # This is a search result, not a proof that no real schedule meets the budget.
        return dict(result, status='no_feasible_candidate', global_infeasibility_proven=False)
    result['config'] = operator_config(result['config'], mode, use_d)
    assert abs(cost(result['config']) - result['predicted_ms']) < 1e-7
    if not use_d:
        assert not result['config']['alpha']
    result.update(status='planned', gmac=config_cost(result['config'], spec))
    return result


def config_details(cfg, spec):
    if cfg['kind'] != 'pda':
        return {}
    from leaf.planner.schedule_caps import full_cost
    n, path = spec.n_tokens, []
    for l, (p, m) in enumerate(zip(cfg['prune_keep'], cfg['merge_keep'])):
        p, m = min(n, p), min(n, p, m)
        path.append(dict(block=l, n_in=n, prune_keep=p, merge_keep=m,
                         removed=n-m, merged=p-m, pruned=n-p,
                         theta=(p-m)/(n-m) if n>m else None,
                         alpha=float(cfg['alpha'].get(str(l), 0))))
        n = m
    mac = full_cost(cfg['prune_keep'], cfg['merge_keep'], cfg['alpha'], spec.n_tokens,
                    spec.dim, {k: len(set(v)) for k, v in cfg['groups'].items()}, cfg['rank'], getattr(spec, 'patch', 16),
                    head_gmac(spec))
    return dict(token_path=path, K=sum(r['removed'] > 0 for r in path),
                N_D=len(cfg['alpha']), final_tokens=n, total_gmac=config_cost(cfg, spec),
                backbone_full_gmac=mac['backbone_full'], d_gmac=mac['d_gmac'],
                station_scalar_gops_estimate=mac['station_gops'])
