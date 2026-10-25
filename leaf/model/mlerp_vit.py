"""Optimized M3/N4 wrapper; additive alternative to the sealed stage-1 implementation."""
from dataclasses import dataclass, field
import math

import torch
import torch.nn.functional as F

from leaf.model.token_ops import cls_attention, compress
from leaf.model.reduced_vit import ReductionConfig, ReducedViT
from leaf.model.merge_variants import compress_variant, gather


@dataclass
class MlerpConfig(ReductionConfig):
    merge: dict = field(default_factory=lambda: dict(mode='M3', kernel='triton_norm'))

    def __post_init__(self):
        super().__post_init__()
        if set(self.merge) != {'mode', 'kernel'} or self.merge['mode'] not in ('M0', 'M3', 'N4') or self.merge['kernel'] != 'triton_norm':
            raise ValueError('expected M0/M3/N4 with kernel=triton_norm')


def compress_norm(x, size, scores, n_prune, n_merge, mode, fuse_merge=True):
    if mode == 'M0':
        return compress(x, size, scores, n_prune, n_merge, fuse_merge=fuse_merge)
    if not x.is_cuda or not fuse_merge or torch.is_grad_enabled():
        return compress_variant(x, size, scores, n_prune, n_merge, mode, fuse_merge=fuse_merge)
    b, n, _ = x.shape
    p, m = min(n, n_prune), min(n, n_prune, n_merge)
    order = torch.cat((torch.zeros(b, 1, device=x.device, dtype=torch.long),
                       scores[:, 1:].argsort(-1, descending=True) + 1), 1)[:, :p]
    y, s = gather(x, order), gather(size, order)
    if p == m:
        return y, s, order, None
    metric = F.normalize(y.float(), dim=-1)
    sim = metric[:, m:] @ metric[:, :m].transpose(1, 2)
    sim[..., 0] = -math.inf
    rho, destination = sim.max(-1)
    from leaf.model.kernels.mlerp_merge import merge_norm_inference
    out, sizes = merge_norm_inference(y, s, destination, rho.clamp_min(0), m, mode)
    return out, sizes, order, destination


class MlerpViT(ReducedViT):
    def forward(self, img):
        if self.cfg.merge['mode'] == 'M0':
            return super().forward(img)
        x = self.embed(img)
        size = x.new_ones(x.shape[0], x.shape[1], 1)
        cfg = self.cfg
        for l, block in enumerate(self.blocks):
            out, raw_p = cls_attention(block, block.norm1(x), None if l == 0 else size)
            x = x + block.drop_path1(block.ls1(out))
            if l in self.stations:
                y, size, order, destination = compress_norm(x, size, self.score(l, x, size, raw_p), cfg.prune_keep[l],
                                                            cfg.merge_keep[l], cfg.merge['mode'], cfg.fuse_merge)
                self.track(x, order, destination, y.shape[1]); x = y
            x = x + block.drop_path2(block.ls2(block.mlp(block.norm2(x))))
        return self.head(x, size)
