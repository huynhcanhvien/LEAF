"""Planner state <-> evaluable config.

A state is dict(sites, counts, thetas, alphas): station blocks, tokens after each station, merged share of
the removed tokens, and forecast weight per station (alpha = 0: P only). `config_from_state` turns it into the JSON
config of leaf.model.reduced_vit.ReducedViT (kind 'pda', base 'pd'); D group labels come from
leaf.forecast.head_groups (G groups per station).
"""
import json
from pathlib import Path

from leaf.planner.schedule_caps import canonical, schedule

DIFFRATE_RATES = Path(__file__).with_name('diffrate_rates.json')


class ForecastConfig:
    """Structure of D shared by every config of a campaign: G groups, rank r (ablation A7 varies them)."""

    def __init__(self, groups_path, G=16, rank=32, triton=True):
        self.labels = json.loads(Path(groups_path).read_text())['groups']; self.G, self.rank, self.triton = G, rank, triton

    def at(self, l):
        return list(self.labels[str(l)][str(self.G)])


def state_caps(st, spec):
    return schedule(st['sites'], st['counts'], st['thetas'], spec.n_tokens, spec.depth)


KEYS = ('kind', 'prune_keep', 'merge_keep', 'groups', 'rank', 'triton', 'fuse_merge', 'merge', 'base', 'alpha')


def ordered(cfg):
    """The config with its keys in the order of the released files (the config hash does not depend on it)."""
    return {**{k: cfg[k] for k in KEYS if k in cfg}, **{k: v for k, v in cfg.items() if k not in KEYS}}


def config_from_state(st, spec, dcfg):
    g = state_caps(st, spec)
    alpha = {int(l): float(a) for l, a in zip(st['sites'], st['alphas'])
             if int(l) in g['sites'] and int(l) < spec.depth - 1 and float(a) > 0}
    return ordered(dict(kind='pda', base='pd', prune_keep=g['p'], merge_keep=g['m'], alpha={str(l): a for l, a in alpha.items()},
                        groups={str(l): dcfg.at(l) for l in alpha}, rank=dcfg.rank if alpha else None,
                        triton=dcfg.triton if alpha else True, fuse_merge=True))


def diffrate_schedules(spec):
    """{budget: (prune_kept, merge_kept)} published by DiffRate for this backbone (compression_rate.json).

    The published counts are for 197 tokens (patch 16 at 224). On a backbone with another token count (rates
    transferred, e.g. MAE-L rates on CLIP ViT-L/14 with 257 tokens) the patch-token counts are scaled by
    (n_tokens - 1) / 196, keeping the CLS token: the same fraction of patch tokens is kept in every block."""
    if spec.diffrate_key is None: return {}
    rates = json.loads(DIFFRATE_RATES.read_text())['rates'][spec.diffrate_key]
    out = {b: (json.loads(v['prune_kept_num']), json.loads(v['merge_kept_num'])) for b, v in rates.items()}
    if spec.n_tokens == 197: return out
    scale = lambda counts: [1 + round((c - 1) * (spec.n_tokens - 1) / 196) for c in counts]
    return {b: (scale(p), scale(m)) for b, (p, m) in out.items()}


def diffrate_config(spec, budget):
    """DiffRate's published schedule with its own P ranking (alpha = 0 everywhere): the DR baseline."""
    return diffrate_config_from(spec, *diffrate_schedules(spec)[budget])


def diffrate_config_from(spec, p, m):
    """DR baseline config of given DiffRate kept counts per block (prune, merge), e.g. rates searched on a new backbone."""
    p = list(p) + [p[-1]] * (spec.depth - len(p)); m = list(m) + [m[-1]] * (spec.depth - len(m))
    g = canonical(dict(p=p, m=m), spec.n_tokens, spec.depth)
    return ordered(dict(kind='pda', base='pd', prune_keep=g['p'], merge_keep=g['m'], alpha={}, groups={}, rank=None,
                        triton=True, fuse_merge=True))


def norm_config(state, spec, dcfg, mode):
    """Config of a planner state with one fixed merge operator: M0 (mean merge) or the MLERP merge M3 (N4: its variant)."""
    cfg = config_from_state(state, spec, dcfg)
    if mode == 'M0': return cfg
    if mode not in ('M3', 'N4'): raise ValueError('one fixed operator per method')
    return ordered(dict(cfg, base='pdm', merge=dict(mode=mode, kernel='triton_norm')))
