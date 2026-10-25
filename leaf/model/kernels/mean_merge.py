"""Inference FP32 mean merge of DiffRate's operator: x and similarity-weighted size in one kernel.

No atomic updates: each destination tile owns its output. Training uses the differentiable
scatter path in models.ops.compress."""
import torch
import triton
import triton.language as tl


@triton.jit
def merge_kernel(X, S, DEST, MAX, OX, OS, NP: tl.constexpr, NM: tl.constexpr, C: tl.constexpr,
                 SRC: tl.constexpr, SPAD: tl.constexpr, DT: tl.constexpr, CT: tl.constexpr):
    tiles = tl.cdiv(NM, DT)
    batch = tl.program_id(0) // tiles
    dst = (tl.program_id(0) % tiles) * DT + tl.arange(0, DT)
    col = tl.program_id(1) * CT + tl.arange(0, CT)
    src = tl.arange(0, SPAD)
    target = tl.load(DEST + batch * SRC + src, src < SRC, other=-1)
    assigned = (dst[:, None] == target[None, :]) & (src[None, :] < SRC) & (dst[:, None] < NM)
    count = 1 + tl.sum(assigned.to(tl.int32), 1)
    base = tl.load(X + batch * NP * C + dst[:, None] * C + col[None, :], (dst[:, None] < NM) & (col[None, :] < C), other=0)
    sources = tl.load(X + batch * NP * C + (NM + src[None, :, None]) * C + col[None, None, :],
                      assigned[:, :, None] & (col[None, None, :] < C), other=0)
    value = (base + tl.sum(sources, 1)) / count[:, None]
    tl.store(OX + batch * NM * C + dst[:, None] * C + col[None, :], value, (dst[:, None] < NM) & (col[None, :] < C))
    if tl.program_id(1) == 0:
        size = tl.load(S + batch * NP + dst, dst < NM, other=0)
        ss = tl.load(S + batch * NP + NM + src, src < SRC, other=0)
        maximum = tl.load(MAX + batch * SRC + src, src < SRC, other=0).to(tl.float32)
        total = size + tl.sum(tl.where(assigned, ss[None, :] * maximum[None, :], 0), 1)
        tl.store(OS + batch * NM + dst, total, dst < NM)


# The tile kernel scans every source for every destination (cost ~ destinations x sources): right for the paper's 197 and
# 257 tokens, quadratic beyond. Larger inputs (segmentation windows, 1,025 tokens) sort the sources by destination and
# give each destination its own segment (cost ~ tokens); its sums run in source order, so its bits differ from the tile
# kernel's tree sums, which is why the paper's token counts keep the tile kernel.
TILE_MAX_TOKENS = 257


@triton.jit
def segment_merge_kernel(X, S, MAX, ORDER, BOUND, OX, OS, NP: tl.constexpr, NM: tl.constexpr, C: tl.constexpr,
                         SRC: tl.constexpr, CT: tl.constexpr):
    row = tl.program_id(0)
    batch, dst = row // NM, row % NM
    col = tl.program_id(1) * CT + tl.arange(0, CT)
    start = tl.load(BOUND + batch * (NM + 1) + dst)
    end = tl.load(BOUND + batch * (NM + 1) + dst + 1)
    value = tl.load(X + batch * NP * C + dst * C + col, col < C, other=0)
    total = tl.load(S + batch * NP + dst)
    for i in range(start, end):
        s = tl.load(ORDER + batch * SRC + i)
        value += tl.load(X + batch * NP * C + (NM + s) * C + col, col < C, other=0)
        total += tl.load(S + batch * NP + NM + s) * tl.load(MAX + batch * SRC + s).to(tl.float32)
    tl.store(OX + batch * NM * C + dst * C + col, value / (1 + end - start).to(tl.float32), col < C)
    if tl.program_id(1) == 0:
        tl.store(OS + batch * NM + dst, total)


def segment_merge(x, size, destination, maximum, n_merge):
    """merge_inference for any token count: sources sorted by destination (stable), one segment per destination."""
    b, np_, c = x.shape; src = np_ - n_merge
    ordered, order = destination.sort(dim=1, stable=True)
    bound = torch.searchsorted(ordered, torch.arange(n_merge + 1, device=x.device).expand(b, -1).contiguous())
    out = torch.empty((b, n_merge, c), device=x.device, dtype=x.dtype)
    sizes = torch.empty((b, n_merge, 1), device=x.device, dtype=size.dtype)
    segment_merge_kernel[(b * n_merge, triton.cdiv(c, 256))](
        x, size, maximum, order, bound, out, sizes, np_, n_merge, c, src, 256, num_warps=4, enable_fp_fusion=False)
    return out, sizes


def merge_inference(x, size, destination, maximum, n_merge):
    b, np_, c = x.shape; src = np_ - n_merge
    if x.dtype != torch.float32 or size.dtype != torch.float32:
        raise ValueError('merge kernel requires FP32 residuals and size')
    if np_ > TILE_MAX_TOKENS: return segment_merge(x, size, destination, maximum, n_merge)
    out = torch.empty((b, n_merge, c), device=x.device, dtype=x.dtype)
    sizes = torch.empty((b, n_merge, 1), device=x.device, dtype=size.dtype)
    merge_kernel[(b * triton.cdiv(n_merge, 8), triton.cdiv(c, 128))](
        x, size, destination, maximum, out, sizes, np_, n_merge, c, src, triton.next_power_of_2(src), 8, 128,
        num_warps=4, enable_fp_fusion=False)
    return out, sizes
