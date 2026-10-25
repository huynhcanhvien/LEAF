"""Inference kernels for AttentionForecast: z-scored log-domain grouped D in two launches.

token_logits: one read of the FP32 residual per token tile -> unaffine LayerNorm -> BF16 dot
              with every group's token-side vector u_g (G padded to >= 16 for tl.dot).
finish:       + log size, log_softmax over tokens, + log|g|, logsumexp over groups, - log H,
              then the population z-score over patch tokens (CLS -> 0), as models.ops.zscore;
              optionally adds z(log P): the full score z(log P) + k_d*z(D) of one station.
Matches AttentionForecast.forward followed by zscore(., 1) up to BF16 rounding of the bmm output
(tests/test_pid_fast_d.py).
"""
import math

import torch
import triton
import triton.language as tl

from leaf.latency.timing import INFERENCE_DTYPE


@triton.jit
def token_logits_kernel(X, U, L, N, C: tl.constexpr, G: tl.constexpr, eps,
                        BN: tl.constexpr, BC: tl.constexpr):
    b = tl.program_id(0)
    rows = tl.program_id(1) * BN + tl.arange(0, BN)
    cols = tl.arange(0, BC)
    rmask = rows < N
    cmask = cols < C
    m = rmask[:, None] & cmask[None, :]
    x = tl.load(X + b * N * C + rows[:, None] * C + cols[None, :], m, other=0.)
    mean = tl.sum(x, 1) / C
    xc = tl.where(m, x - mean[:, None], 0.)
    rstd = 1. / tl.sqrt(tl.sum(xc * xc, 1) / C + eps)
    hat = (xc * rstd[:, None]).to(U.dtype.element_ty)
    g = tl.arange(0, G)
    u = tl.load(U + b * G * C + g[None, :] * C + cols[:, None], cmask[:, None], other=0.)
    acc = tl.dot(hat, u)
    tl.store(L + b * N * G + rows[:, None] * G + g[None, :], acc, rmask[:, None])


@triton.jit
def _zbody(v, body, cnt):
    mean = tl.sum(tl.where(body, v, 0.), 0) / cnt
    d = tl.where(body, v - mean, 0.)
    return d / tl.maximum(tl.sqrt(tl.sum(d * d, 0) / cnt), 1e-6)


@triton.jit
def finish_kernel(L, S, LC, P, O, N, G: tl.constexpr, NP: tl.constexpr, log_h, k_d, HAS_P: tl.constexpr):
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
    out = k_d * _zbody(s, body, cnt)
    if HAS_P:
        p = tl.log(tl.maximum(tl.load(P + b * N + rows, rmask, other=1.), 1e-12))
        out += _zbody(p, body, cnt)
    tl.store(O + b * N + rows, out, rmask)


def zscored_future(x, size, u, log_count, n_heads, eps, raw_p=None, k_d=1.):
    """x [B,N,C] float, size [B,N,1], u [B,Gp,C] bf16 (Gp >= 16, power of 2), log_count [Gp].

    Returns k_d*z(D) [+ z(log raw_p)] over patch tokens, CLS -> 0."""
    B, N, C = x.shape; G = u.shape[1]
    x = x.float().contiguous(); u = u.to(INFERENCE_DTYPE).contiguous()
    logits = torch.empty(B, N, G, device=x.device, dtype=torch.float32)
    BN = 16
    token_logits_kernel[(B, triton.cdiv(N, BN))](x, u, logits, N, C, G, eps, BN=BN,
                                                 BC=triton.next_power_of_2(C), num_warps=4)
    out = torch.empty(B, N, device=x.device, dtype=torch.float32)
    size = size.float().contiguous()
    finish_kernel[(B,)](logits, size, log_count.float().contiguous(),
                        size if raw_p is None else raw_p.float().contiguous(), out, N, G,
                        NP=max(16, triton.next_power_of_2(N)), log_h=math.log(n_heads), k_d=float(k_d),
                        HAS_P=raw_p is not None, num_warps=4)
    return out
