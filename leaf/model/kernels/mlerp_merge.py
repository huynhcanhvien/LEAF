"""Deterministic FP32 M3/N4 inference kernels; no atomics, three short launches.

Token norms are computed once. Each destination owns its mean and norm correction.
M0's existing kernel is reused verbatim for M3 mean/size. No legacy kernel changes.
"""
import torch
import triton
import triton.language as tl

from leaf.model.kernels.mean_merge import merge_inference


@triton.jit
def token_norms(X, NORM, N: tl.constexpr, C: tl.constexpr, CP: tl.constexpr, CENTER: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, CP)
    x = tl.load(X + row * C + col, col < C, 0)
    if CENTER:
        mean = tl.sum(x, 0) / C
        x = tl.where(col < C, x - mean, 0.)
    norm = tl.sqrt(tl.sum(x * x, 0))
    tl.store(NORM + row, norm)


@triton.jit
def weighted_mean(X, S, DEST, OX, OS, NP: tl.constexpr, NM: tl.constexpr, C: tl.constexpr,
                  SRC: tl.constexpr, SP: tl.constexpr, DT: tl.constexpr, CT: tl.constexpr):
    tiles = tl.cdiv(NM, DT)
    b = tl.program_id(0) // tiles
    d = (tl.program_id(0) % tiles) * DT + tl.arange(0, DT)
    c = tl.program_id(1) * CT + tl.arange(0, CT)
    src = tl.arange(0, SP)
    target = tl.load(DEST + b * SRC + src, src < SRC, -1)
    assigned = (d[:, None] == target[None, :]) & (src[None, :] < SRC) & (d[:, None] < NM)
    base = tl.load(X + b * NP * C + d[:, None] * C + c[None, :], (d[:, None] < NM) & (c[None, :] < C), 0.)
    bs = tl.load(S + b * NP + d, d < NM, 0.)
    ss = tl.load(S + b * NP + NM + src, src < SRC, 0.)
    total = bs + tl.sum(tl.where(assigned, ss[None, :], 0.), 1)
    sources = tl.load(X + b * NP * C + (NM + src[None, :, None]) * C + c[None, None, :],
                      assigned[:, :, None] & (c[None, None, :] < C), 0.)
    value = (base * bs[:, None] + tl.sum(sources * ss[None, :, None], 1)) / total[:, None]
    # A singleton is copied bitwise, avoiding multiply/divide roundoff.
    has_source = tl.sum(assigned.to(tl.int32), 1) > 0
    value = tl.where(has_source[:, None], value, base)
    tl.store(OX + b * NM * C + d[:, None] * C + c[None, :], value, (d[:, None] < NM) & (c[None, :] < C))
    if tl.program_id(1) == 0:
        tl.store(OS + b * NM + d, total, d < NM)


@triton.jit
def stable_sizes(S, DEST, MAXIMUM, OS, NP: tl.constexpr, NM: tl.constexpr, SRC: tl.constexpr,
                 CENTER: tl.constexpr, DT: tl.constexpr):
    b = tl.program_id(0)
    d = tl.program_id(1) * DT + tl.arange(0, DT)
    mass = tl.load(S + b * NP + d, d < NM, 0.)
    for i in range(SRC):
        target = tl.load(DEST + b * SRC + i)
        si = tl.load(S + b * NP + NM + i)
        if not CENTER:
            si = si * tl.load(MAXIMUM + b * SRC + i).to(tl.float32)
        mass = mass + tl.where(target == d, si, 0.)
    tl.store(OS + b * NM + d, mass, d < NM)


@triton.jit
def correct_norm(X, S, DEST, NORM, MEAN, OS, OUT, NP: tl.constexpr, NM: tl.constexpr,
                 C: tl.constexpr, CP: tl.constexpr, SRC: tl.constexpr, SP: tl.constexpr, CENTER: tl.constexpr):
    row = tl.program_id(0)
    b, d = row // NM, row % NM
    src = tl.arange(0, SP)
    target = tl.load(DEST + b * SRC + src, src < SRC, -1)
    assigned = (target == d) & (src < SRC)
    count = tl.sum(assigned.to(tl.int32), 0)
    base_norm = tl.load(NORM + b * NP + d)
    source_norm = tl.load(NORM + b * NP + NM + src, src < SRC, 0.)
    peak = tl.maximum(base_norm, tl.max(tl.where(assigned, source_norm, 0.), 0))
    mass = tl.load(OS + row)
    if CENTER:
        bs = tl.load(S + b * NP + d)
        ss = tl.load(S + b * NP + NM + src, src < SRC, 0.)
        desired = (base_norm * bs + tl.sum(tl.where(assigned, ss * source_norm, 0.), 0)) / mass
    else:
        desired = peak
    col = tl.arange(0, CP)
    y = tl.load(MEAN + row * C + col, col < C, 0.)
    center = tl.full((), 0., tl.float32)
    if CENTER:
        center = tl.sum(y, 0) / C
        direction = tl.where(col < C, y - center, 0.)
    else:
        direction = y
    ynorm = tl.sqrt(tl.sum(direction * direction, 0))
    if ynorm == 0.:
        # First largest member, matching the stable PyTorch reference at cancellation.
        first = tl.min(tl.where(assigned & (source_norm == peak), NM + src, NP), 0)
        first = tl.where(base_norm == peak, d, first)
        fallback = tl.load(X + b * NP * C + first * C + col, col < C, 0.)
        if CENTER:
            fm = tl.sum(fallback, 0) / C
            fallback = tl.where(col < C, fallback - fm, 0.)
        direction = fallback
        ynorm = tl.sqrt(tl.sum(direction * direction, 0))
    value = center + direction * (desired / tl.maximum(ynorm, 1.e-30))
    value = tl.where(count > 0, value, y)
    tl.store(OUT + row * C + col, value, col < C)


def merge_norm_inference(x, size, destination, maximum, n_merge, mode):
    if mode not in ('M3', 'N4'):
        raise ValueError('norm kernel supports M3 and N4')
    if x.dtype != torch.float32 or size.dtype != torch.float32 or not x.is_cuda:
        raise ValueError('norm kernels require CUDA FP32 residual and size')
    b, p, c = x.shape
    src = p - n_merge
    if src == 0:
        return x, size
    norm = torch.empty((b, p), device=x.device)
    center = mode == 'N4'
    token_norms[(b * p,)](x, norm, p, c, triton.next_power_of_2(c), center,
                          num_warps=4, enable_fp_fusion=False)
    if mode == 'M3':
        mean, sizes = merge_inference(x, size, destination, maximum, n_merge)
    else:
        mean = torch.empty((b, n_merge, c), device=x.device)
        sizes = torch.empty((b, n_merge, 1), device=x.device)
        weighted_mean[(b * triton.cdiv(n_merge, 8), triton.cdiv(c, 128))](
            x, size, destination, mean, sizes, p, n_merge, c, src, triton.next_power_of_2(src), 8, 128,
            num_warps=4, enable_fp_fusion=False)
    out = torch.empty_like(mean)
    stable_sizes[(b, triton.cdiv(n_merge, 128))](size, destination, maximum, sizes, p, n_merge, src, center, 128,
                                              num_warps=4, enable_fp_fusion=False)
    correct_norm[(b * n_merge,)](x, size, destination, norm, mean, sizes, out,
                                p, n_merge, c, triton.next_power_of_2(c), src,
                                triton.next_power_of_2(src), center, num_warps=4, enable_fp_fusion=False)
    return out, sizes
