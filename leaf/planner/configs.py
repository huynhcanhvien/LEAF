"""Planner state <-> evaluable config.

A state is dict(sites, counts, thetas, ks): station blocks, tokens after each station, merged share of
the removed tokens, and D weight per station (k = 0: P only). `config_from_state` turns it into the JSON config
of models.pd_vit.ReducedViT (kind 'pd'); D group labels come from planner.grouping (G groups per station).
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


def config_from_state(st, spec, dcfg):
    g = state_caps(st, spec)
    ks = {int(l): float(k) for l, k in zip(st['sites'], st['ks']) if int(l) in g['sites'] and int(l) < spec.depth - 1 and float(k) > 0}
    return dict(kind='pd', prune_keep=g['p'], merge_keep=g['m'], k={str(l): k for l, k in ks.items()},
                groups={str(l): dcfg.at(l) for l in ks}, rank=dcfg.rank if ks else None, triton=dcfg.triton if ks else True,
                fuse_merge=True)


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
    """DiffRate's published schedule with its own P ranking (k = 0 everywhere): the DR baseline."""
    return diffrate_config_from(spec, *diffrate_schedules(spec)[budget])


def diffrate_config_from(spec, p, m):
    """DR baseline config of given DiffRate kept counts per block (prune, merge), e.g. rates searched on a new backbone."""
    p = list(p) + [p[-1]] * (spec.depth - len(p)); m = list(m) + [m[-1]] * (spec.depth - len(m))
    g = canonical(dict(p=p, m=m), spec.n_tokens, spec.depth)
    return dict(kind='pd', prune_keep=g['p'], merge_keep=g['m'], k={}, groups={}, rank=None, triton=True, fuse_merge=True)
