"""Planner adapters for the alpha score (leaf/model/alpha_score.py); the k planner code is unchanged.

The DP only needs a forecast setting that is > 0 exactly when D is active, and the inner minimum over
settings iterates the curve dict in order. Relabelling the probe curves (theta, k) -> (theta, k/(1+k))
keeps both, so plan_latency on relabelled curves returns the legacy schedule with alpha in the 'k' field;
to_alpha_config turns that into a 'pda' config. Curves probed directly in alpha (alpha = 1 included)
plug in the same way.
"""
from copy import deepcopy

from leaf.model.alpha_score import BASES, alpha_to_kappa_config, kappa_to_alpha


def curves_to_alpha(curves):
    """{(theta, k): ProbeCurve} -> {(theta, alpha): ProbeCurve}, same order."""
    out = {}
    for (th, k), cv in curves.items():
        key = (th, kappa_to_alpha(k))
        if key in out: raise ValueError(f'two curves map to {key}')
        out[key] = cv
    return out


def to_alpha_config(cfg):
    """A planner config whose 'k' holds alpha values -> 'pda' config."""
    if cfg['kind'] not in BASES: return cfg
    out = deepcopy(cfg)
    out['base'] = out.pop('kind')
    out['alpha'] = {str(l): float(a) for l, a in out.pop('k', {}).items() if float(a) > 0}
    return dict(kind='pda', **out)


def kappa_view(cfg):
    """Legacy config with the same modules and SALT/GMAC features (placeholder k) for cost functions."""
    return alpha_to_kappa_config(cfg) if cfg['kind'] == 'pda' else cfg


def plan_under_latency(curves, spec, dcfg, model, budget, mode='M3', use_d=True):
    """planner.low_budget_addon.plan_latency on alpha-keyed curves; returns a 'pda' config."""
    from leaf.planner.plan import OperatorCost, plan_latency
    result = plan_latency(curves, spec, dcfg, model, budget, mode, use_d)
    if 'config' not in result: return result
    result = dict(result, config=to_alpha_config(result['config']))
    if 'state' in result:
        st = dict(result['state']); st['alphas'] = st.pop('ks'); result['state'] = st
    cost = OperatorCost(model, spec, dcfg, mode)
    assert abs(cost(kappa_view(result['config'])) - result['predicted_ms']) < 1e-7
    return result


def config_details_alpha(cfg, spec):
    """planner.low_budget_addon.config_details with 'alpha' per station."""
    from leaf.planner.plan import config_details
    if cfg['kind'] != 'pda': return config_details(cfg, spec)
    tmp = dict(alpha_to_kappa_config(cfg), k=dict(cfg['alpha']))
    if tmp['kind'] == 'extra_pdm': tmp = dict({k: v for k, v in tmp.items() if k != 'ablation'}, kind='pdm')
    out = config_details(tmp, spec)
    for r in out.get('token_path', []): r['alpha'] = r.pop('kappa')
    return out


def config_gmac(cfg, spec):
    from leaf.planner.gmac import config_cost
    return config_cost(kappa_view(cfg), spec)
