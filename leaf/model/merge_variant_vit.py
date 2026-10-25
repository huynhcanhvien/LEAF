"""Additive experimental PD wrapper: legacy ReductionConfig and ReducedViT stay unchanged."""
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from leaf.model.merge_variants import MODES, cls_attention_with_keys, compress_variant
from leaf.model.token_ops import cls_attention, zscore
from leaf.model.reduced_vit import ReductionConfig, ReducedViT


@dataclass
class MergeVariantConfig(ReductionConfig):
    merge: dict = field(default_factory=lambda: {'mode': 'M0'})

    def __post_init__(self):
        super().__post_init__()
        self.merge = dict(self.merge)
        if set(self.merge) - {'mode'}:
            raise ValueError('stage 1 freezes all merge parameters; only mode is configurable')
        self.merge.setdefault('mode', 'M0')
        if self.merge['mode'] not in MODES:
            raise ValueError(f'unknown merge mode: {self.merge["mode"]}')


def score_and_forecast(future, x, size, raw_p, k):
    """Reuse D's group token-side vectors for both ranking and N2.

    Scalar λ is the mean group logit weighted by group head counts. Padded
    groups have zero weight. Vectors respect the same rank/dtype as D inference.
    This convention is explicitly frozen in the stage-1 rule.
    """
    b, _, c = x.shape
    cls = F.layer_norm(x[:, 0].float(), (c,), eps=future.eps).to(future.U.dtype)
    if getattr(future, 'triton', False) and x.is_cuda:
        if future.rank is None:
            u = torch.addmm(future.U[c], cls, future.U[:c]).view(b, -1, c)
        else:
            t = torch.addmm(future.A[c], cls, future.A[:c]).view(b, -1, future.rank)
            u = torch.bmm(t.transpose(0, 1), future.Bf).transpose(0, 1)
        from leaf.forecast.kernels.kappa_score import zscored_future
        score = zscored_future(x, size, u, future.log_count, future.H, future.eps, raw_p, k)
    else:
        aug = torch.cat((cls, cls.new_ones(b, 1)), -1)
        u = (aug @ future.U).view(b, -1, c)
        hat = F.layer_norm(x.float(), (c,), eps=future.eps)
        logits = torch.bmm(hat.to(u.dtype), u.transpose(1, 2)).float() + size.float().log()
        d = (logits.log_softmax(1) + future.log_count).logsumexp(-1)
        # Constant -log H cancels under zscore, but keep the legacy operation.
        import math
        score = k * zscore(d - math.log(future.H), 1) + zscore(raw_p.clamp_min(1e-12).log(), 1)
    counts = future.log_count.exp()
    a = (u.float() * (counts / counts.sum())[None, :, None]).sum(1)
    return score, (a, future.eps)


class MergeVariantViT(ReducedViT):
    def forward(self, img):
        mode = self.cfg.merge['mode']
        if mode == 'M0':
            return super().forward(img)
        cfg = self.cfg
        x = self.embed(img)
        size = x.new_ones(x.shape[0], x.shape[1], 1)
        for l, block in enumerate(self.blocks):
            keys = None
            if mode.startswith('M2') and l in self.stations:
                out, raw_p, keys = cls_attention_with_keys(block, block.norm1(x), None if l == 0 else size)
            else:
                out, raw_p = cls_attention(block, block.norm1(x), None if l == 0 else size)
            x = x + block.drop_path1(block.ls1(out))
            if l in self.stations:
                forecast = None
                if 'N2' in mode and str(l) in self.future:
                    score, forecast = score_and_forecast(self.future[str(l)], x, size, raw_p, cfg.k[str(l)])
                else:
                    score = self.score(l, x, size, raw_p)
                y, size, order, destination = compress_variant(x, size, score, cfg.prune_keep[l], cfg.merge_keep[l],
                                                               mode, keys=keys, forecast=forecast, fuse_merge=cfg.fuse_merge)
                self.track(x, order, destination, y.shape[1]); x = y
            x = x + block.drop_path2(block.ls2(block.mlp(block.norm2(x))))
        return self.head(x, size)
