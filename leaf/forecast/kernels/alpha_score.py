"""Convex P/D score for one station: (1 - alpha)*z(log P) + alpha*z(D), alpha in [0, 1].

Additive counterpart of leaf/forecast/kernels/kappa_score.py (unchanged). The legacy score z(log P) + k*z(D)
gives the same token order for alpha = k/(1 + k), since the station only sorts the score
(models.ops.compress). token_logits_kernel and _zbody are reused; finish_alpha_kernel is finish_kernel
with explicit weights on both terms. alpha = 1 ranks by D alone.
"""
import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from leaf.forecast.kernels.kappa_score import _zbody, token_logits_kernel
from leaf.latency.timing import INFERENCE_DTYPE
from leaf.model.token_ops import zscore


@triton.jit
def finish_alpha_kernel(L, S, LC, P, O, N, G: tl.constexpr, NP: tl.constexpr, log_h, w_d, w_p,
                        HAS_P: tl.constexpr):
    b = tl.program_id(0)
    rows = tl.arange(0, NP)
    g = tl.arange(0, G)
    rmask = rows < N
    z = tl.load(L + b * N * G + rows[:, None] * G + g[None, :], rmask[:, None], other=-float('inf'))
    log_size = tl.log(tl.load(S + b * N + rows, rmask, other=1.))
    z = z + log_size[:, None]
    mx = tl.max(z, 0)
    lse = tl.log(tl.sum(tl.exp(z - mx[None, :]), 0)) + mx
    t = z - lse[None, :] + tl.load(LC + g)[None, :]           # padded groups carry log|g| = -inf
    mg = tl.max(t, 1)
    s = tl.log(tl.sum(tl.exp(t - mg[:, None]), 1)) + mg - log_h
    body = (rows >= 1) & rmask
    cnt = N - 1
    out = w_d * _zbody(s, body, cnt)
    if HAS_P:
        p = tl.log(tl.maximum(tl.load(P + b * N + rows, rmask, other=1.), 1e-12))
        out += w_p * _zbody(p, body, cnt)
    tl.store(O + b * N + rows, out, rmask)


def zscored_future_alpha(x, size, u, log_count, n_heads, eps, raw_p=None, alpha=.5):
    """As leaf.forecast.kernels.kappa_score.zscored_future, returning alpha*z(D) [+ (1-alpha)*z(log raw_p)]."""
    B, N, C = x.shape; G = u.shape[1]
    x = x.float().contiguous(); u = u.to(INFERENCE_DTYPE).contiguous()
    logits = torch.empty(B, N, G, device=x.device, dtype=torch.float32)
    BN = 16
    token_logits_kernel[(B, triton.cdiv(N, BN))](x, u, logits, N, C, G, eps, BN=BN,
                                                 BC=triton.next_power_of_2(C), num_warps=4)
    out = torch.empty(B, N, device=x.device, dtype=torch.float32)
    size = size.float().contiguous()
    has_p = raw_p is not None and alpha < 1
    finish_alpha_kernel[(B,)](logits, size, log_count.float().contiguous(),
                              raw_p.float().contiguous() if has_p else size, out, N, G,
                              NP=max(16, triton.next_power_of_2(N)), log_h=math.log(n_heads),
                              w_d=float(alpha), w_p=float(1. - alpha), HAS_P=has_p, num_warps=4)
    return out


def zscored_alpha(future, x, size, raw_p=None, alpha=.5):
    """[B,N] (1-alpha)*z(log P) + alpha*z(D) of a AttentionForecast, CLS -> 0 (AttentionForecast.zscored with alpha)."""
    if not (getattr(future, 'triton', False) and x.is_cuda):
        out = alpha * zscore(future(x, size), 1)
        if raw_p is not None and alpha < 1: out = out + (1. - alpha) * zscore(raw_p.clamp_min(1e-12).log(), 1)
        return out
    B, N, C = x.shape
    cls = F.layer_norm(x[:, 0].float(), (C,), eps=future.eps).to(future.U.dtype)
    # Bias row split off as in AttentionForecast.zscored (aligned K = C GEMM).
    if future.rank is None:
        u = torch.addmm(future.U[C], cls, future.U[:C]).view(B, -1, C)
    else:
        t = torch.addmm(future.A[C], cls, future.A[:C]).view(B, -1, future.rank)
        u = torch.bmm(t.transpose(0, 1), future.Bf).transpose(0, 1)
    return zscored_future_alpha(x, size, u, future.log_count, future.H, future.eps, raw_p, alpha)
