"""Timing workloads for fitting SALT on a new backbone or GPU.

The paper fitted SALT on about 48 schedules per backbone (schedules/leaf/planning/<backbone>/salt_workloads.json):
- DiffRate's released schedules;
- GMAC-planned schedules with the M0 and M3 merges;
- the earlier LEAF-M0 schedules.

These have stations at different blocks, with and without the forecast, pruning and merging.

For a backbone without that history, `make_workloads` builds a set of the same kind from the user's probes:
- GMAC-planned schedules at a grid of fractions of the dense GMAC, in five variants: the full grid with M3, the same
  grid with M0, without the forecast (alpha = 0), pruning only (theta = 0) and merging only (theta = 1);
- DiffRate's released schedules when the backbone has them.

Duplicates are removed. `leaf fit-latency` groups the rows for cross-validation by token path (M0 and M3 of one path
share a group).
"""
from leaf.evaluation.runner import config_hash
from leaf.planner.configs import diffrate_config, diffrate_schedules
from leaf.planner.gmac import GmacPlanner, dense_cost

GMAC_FRACTIONS = (.35, .42, .49, .56, .63, .7, .77, .84)
VARIANTS = (   # (name, curve filter on (theta, alpha), merge operator)
    ('m3', lambda th, a: True, 'M3'),
    ('m0', lambda th, a: True, 'M0'),
    ('no_forecast', lambda th, a: a == 0, 'M3'),
    ('prune_only', lambda th, a: th == 0, 'M3'),
    ('merge_only', lambda th, a: th == 1, 'M3'),
)


def make_workloads(spec, dcfg, curves, fractions=GMAC_FRACTIONS, variants=VARIANTS, diffrate=True, log=print):
    """curves: {(theta, alpha): ProbeCurve} -> [{name, config, gmac_target}] ('pda' configs)."""
    rows, seen = [], set()

    def add(name, cfg, target):
        key = config_hash(cfg)
        if key in seen: return
        seen.add(key); rows.append(dict(name=name, config=cfg, gmac_target=target))

    if diffrate:
        for b in sorted(diffrate_schedules(spec), key=float):
            add(f'diffrate/DR{b}', diffrate_config(spec, b), None)
    dense = dense_cost(spec)
    for name, keep, mode in variants:
        cv = {k: v for k, v in curves.items() if keep(*k)}
        if not cv:
            log(f'workloads: variant {name} skipped (no probe curve on its grid)'); continue
        planner = GmacPlanner(spec, dcfg, cv, mode)
        for f in fractions:
            res = planner.plan(dense * f)
            if 'config' not in res:
                log(f'workloads: {name} at {f:g} x dense GMAC infeasible'); continue
            add(f'{name}/gmac{f:g}', res['config'], dense * f)
    return rows
