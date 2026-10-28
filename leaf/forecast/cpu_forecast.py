"""Low-rank D on CPU: the same computation as the Triton path, for CPU timing.

AttentionForecast.zscored (leaf/forecast/attention_forecast.py, unchanged) falls back to the full-rank reference
forward when the input is not on CUDA; that reads G*C*(C+1) weights per station instead of G*r*(2C+1).
enable_cpu_lowrank(model) binds this function to each AttentionForecast instance of a built model, so the
CPU measures the deployed low-rank forecast. The class and the GPU path are not modified.
"""
import math
import types

import torch
import torch.nn.functional as F

from leaf.forecast.attention_forecast import AttentionForecast
from leaf.model.token_ops import zscore


def zscored_lowrank_cpu(self, x, size, raw_p=None, alpha=1.):
    """[B,N] alpha*z(D) [+ (1-alpha)*z(log raw_p)], CLS -> 0, as leaf.forecast.kernels.station_score.zscored_future."""
    if x.is_cuda:
        return AttentionForecast.zscored(self, x, size, raw_p, alpha)
    B, N, C = x.shape
    cls = F.layer_norm(x[:, 0].float(), (C,), eps=self.eps).to(self.U.dtype)
    if self.rank is None:
        u = torch.addmm(self.U[C], cls, self.U[:C]).view(B, -1, C)
    else:
        t = torch.addmm(self.A[C], cls, self.A[:C]).view(B, -1, self.rank)            # [B,G,r]
        u = torch.bmm(t.transpose(0, 1), self.Bf).transpose(0, 1)                      # [B,G,C]
    hat = F.layer_norm(x.float(), (C,), eps=self.eps).to(u.dtype)
    logits = torch.bmm(hat, u.transpose(1, 2)).float() + size.float().log()            # [B,N,G]
    d = (logits.log_softmax(1) + self.log_count.float()).logsumexp(-1) - math.log(self.H)
    out = alpha * zscore(d, 1)
    if raw_p is not None and alpha < 1:
        out = out + (1. - alpha) * zscore(raw_p.clamp_min(1e-12).log(), 1)
    return out


def enable_cpu_lowrank(model):
    """Bind the low-rank CPU forecast to every AttentionForecast in `model`; returns the number bound."""
    count = 0
    for mod in model.modules():
        if isinstance(mod, AttentionForecast):
            mod.zscored = types.MethodType(zscored_lowrank_cpu, mod)
            count += 1
    return count
