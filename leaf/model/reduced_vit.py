"""Frozen ViT with K token-reduction stations scored by P + k_t * D.

At each station block l (between attention and MLP), tokens are ranked by
    score = z(log P) + k_l * z(D)
P is the head-mean CLS attention row of block l. D is the closed-form log-domain grouped forecast of
CLS attention over every later block (models.grouped_future, rank-r, Triton). The DiffRate operator
(models.ops.compress) then keeps prune_keep[l] tokens and merges down to merge_keep[l].
Stations with k_l = 0 (and the last block, which has no future) rank by P alone.

Spec: artifacts/method/latency-schedule-planner-2026-10-13.md and pid-methods-summary-2026-10-12.md §3.
"""
from dataclasses import asdict, dataclass, field
import json

import torch
import torch.nn as nn

from leaf.forecast.attention_forecast import AttentionForecast
from leaf.model.dense import SourceMap
from leaf.model.token_ops import cls_attention, compress, zscore


@dataclass
class ReductionConfig:
    prune_keep: tuple                     # canonical per-block caps (planner.schedule.canonical)
    merge_keep: tuple
    k: dict = field(default_factory=dict) # {station block: k_D}; absent or 0 -> P only at that station
    groups: dict = field(default_factory=dict)   # {station block: one group label per future (block, head)}
    rank: int | None = 32                 # low-rank D (SVD of each group's bilinear matrix)
    triton: bool = True                   # fused Triton D + score kernels on CUDA
    fuse_merge: bool = True               # Triton FP32 merge kernel at inference

    def __post_init__(self):
        self.prune_keep, self.merge_keep = tuple(int(v) for v in self.prune_keep), tuple(int(v) for v in self.merge_keep)
        if len(self.prune_keep) != len(self.merge_keep): raise ValueError('one prune/merge cap per block')
        self.k = {str(int(l)): float(v) for l, v in self.k.items() if float(v) > 0}
        if any(v < 0 for v in self.k.values()): raise ValueError('k must be >= 0')
        self.groups = {str(int(l)): [int(g) for g in lab] for l, lab in self.groups.items() if str(int(l)) in self.k}
        if set(self.groups) != set(self.k): raise ValueError('every station with k > 0 needs its group labels')

    def to_dict(self):
        return json.loads(json.dumps(asdict(self)))


class ReducedViT(nn.Module):
    def __init__(self, vit, cfg: ReductionConfig, pool='token'):
        super().__init__()
        self.vit, self.cfg, self.pool = vit, cfg, pool
        self.sources = None                # SourceMap of the current forward (pool 'dense' only)
        self.n_blocks = len(vit.blocks)
        self.n_tokens = vit.patch_embed.num_patches + vit.num_prefix_tokens
        if len(cfg.prune_keep) != self.n_blocks: raise ValueError('one prune/merge cap per block')
        self.stations, n = [], self.n_tokens
        for l, (p, m) in enumerate(zip(cfg.prune_keep, cfg.merge_keep)):
            if p < 2 or m < 2 or m > p: raise ValueError('invalid token schedule')
            if min(n, p, m) < n: self.stations.append(l)
            n = min(n, p, m)
        if any(int(l) not in self.stations or int(l) >= self.n_blocks - 1 for l in cfg.k):
            raise ValueError('D is defined only at stations before the last block')
        self.use_d = bool(cfg.k)
        self.future = nn.ModuleDict()
        for l, labels in cfg.groups.items():
            if any(p.requires_grad for p in vit.parameters()): raise ValueError('closed-form D requires a frozen backbone')
            self.future[l] = AttentionForecast([vit.blocks[m] for m in range(int(l) + 1, self.n_blocks)], labels, cfg.rank)
            if cfg.triton: self.future[l].use_triton()

    @property
    def blocks(self):
        return self.vit.blocks

    def embed(self, img):
        """Input tokens; with a dense head (pool 'dense') also a fresh source map (leaf.model.dense)."""
        vit = self.vit
        x = vit.norm_pre(vit.patch_drop(vit._pos_embed(vit.patch_embed(img))))
        self.sources = SourceMap(x) if self.pool == 'dense' else None
        return x

    def track(self, x, order, destination, n_merge):
        """Record a station (x: its input tokens) in the source map of a dense head; no-op otherwise."""
        if self.sources is not None: self.sources.update(x, order, destination, n_merge)

    def head(self, x, size):
        vit = self.vit
        if self.pool == 'dense':            # every input position, then the decoder (vit.head) on the full grid
            return vit.head(self.sources.reconstruct(x, vit.norm))
        x = vit.norm(x)
        if self.pool == 'avg':
            w = size[:, 1:]
            x = (x[:, 1:] * w).sum(1) / w.sum(1)
        else:
            x = x[:, 0]
        return vit.head(vit.head_drop(vit.fc_norm(x)))

    def score(self, l, x, size, raw_p):
        key = str(l)
        if key in self.future:
            return self.future[key].zscored(x, size, raw_p, self.cfg.k[key])
        # P only. With D elsewhere in the model, z(log P) (same ranking up to ties of the clamp).
        return zscore(raw_p.clamp_min(1e-12).log(), 1) if self.use_d else raw_p

    def forward(self, img, start=0, stop=None):
        """start > 0: img holds the tokens entering block start, as this loop computes them without a station before
        it (leaf.evaluation.prefix); stop: return the tokens entering block stop instead of the logits."""
        cfg = self.cfg
        x = self.embed(img) if start == 0 else img
        size = x.new_ones(x.shape[0], x.shape[1], 1)
        for l, block in enumerate(self.blocks):
            if l < start: continue
            if l == stop: return x
            out, raw_p = cls_attention(block, block.norm1(x), None if l == 0 else size)
            x = x + block.drop_path1(block.ls1(out))
            if l in self.stations:
                y, size, order, destination = compress(x, size, self.score(l, x, size, raw_p), cfg.prune_keep[l],
                                                       cfg.merge_keep[l], fuse_merge=cfg.fuse_merge)
                self.track(x, order, destination, y.shape[1]); x = y
            x = x + block.drop_path2(block.ls2(block.mlp(block.norm2(x))))
        return self.head(x, size)
