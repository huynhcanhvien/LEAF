"""Log-domain grouped D with one precomputed bilinear matrix per group.

Under the frozen-residual proxy, the CLS logit of future pair (m, h) on token j is
    s_m (W_q^{mh}(γ_m⊙ĉ+β_m)+b_q^{mh})·(W_k^{mh}(γ_m⊙ĵ+β_m)+b_k^{mh}) + log size_j
with ĉ, ĵ the unaffine LayerNorm of the CLS and token. Terms constant in j cancel in a
softmax over tokens, so the group's mean logit is [ĉ;1]ᵀ U_g ĵ / |g| + log size_j with
    U_g = Σ_{(m,h)∈g} s_m [W_q^{mh} diag γ_m , W_q^{mh}β_m + b_q^{mh}]ᵀ W_k^{mh} diag γ_m .
The score is log Σ_g |g|·softmax_j(mean logit of g) − log H, equal to
`grouped_log_rows(folded_rows(...), labels)` up to rounding (tests/test_attention_forecast.py).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from leaf.forecast.future_attention import pack_blocks
from leaf.model.token_ops import zscore


SVD_THREADS = 4
_CACHE = {}   # (block ids, labels) -> (U, count, SVD factors) on the CPU: building a config reuses them (deterministic)


def _group_matrices(blocks, labels):
    key = (tuple(id(b) for b in blocks), tuple(int(g) for g in labels))
    if key in _CACHE: return _CACHE[key]
    p = pack_blocks(blocks)
    M, H, d, C = p['wk'].shape
    lab = torch.unique(torch.as_tensor(labels), return_inverse=True)[1]
    if lab.numel() != M * H: raise ValueError('one group label per future (block, head) pair')
    G = int(lab.max()) + 1
    with torch.no_grad(), torch.autocast(p['wk'].device.type, enabled=False):
        wq = p['wq'].float().reshape(M, H, d, C)
        aq = torch.cat((wq * p['gamma'].float()[:, None, None, :],
                        (torch.einsum('mhdc,mc->mhd', wq, p['beta'].float())
                         + p['bq'].float().reshape(M, H, d))[..., None]), -1)          # [M,H,d,C+1]
        ak = p['wk'].float() * p['gamma'].float()[:, None, None, :]                  # [M,H,d,C]
        pair = torch.einsum('mhda,mhdc->mhac', aq, ak) * p['scale'].float()[:, None, None, None]
        pair = pair.reshape(M * H, C + 1, C); lab_d = lab.to(pair.device)
        # Per-group sums instead of index_add_ (CUDA atomics are not run-to-run deterministic).
        U = torch.stack([pair[lab_d == g].sum(0) for g in range(G)])
        count = torch.bincount(lab, minlength=G).float().to(pair.device)
        U = U / count[:, None, None]                                                   # group-mean logit
        # SVD in float64 on the CPU: the CUDA solver is not run-to-run deterministic, which made
        # repeated evaluations of the same config differ (0.1-0.2% of predictions).
        # LAPACK's result also depends on the CPU thread count: pin it (4, the evaluation runner's setting).
        threads = torch.get_num_threads(); torch.set_num_threads(SVD_THREADS)
        try: svd = tuple(t.float() for t in torch.linalg.svd(U.double().cpu(), full_matrices=False))
        finally: torch.set_num_threads(threads)
    _CACHE[key] = out = (U.cpu(), count.cpu(), svd, H, blocks[0].norm1.eps)
    return out


class AttentionForecast(nn.Module):
    def __init__(self, blocks, labels, rank=None):
        super().__init__()
        U, count, (Us, S, Vh), H, eps = _group_matrices(blocks, labels)
        dev = blocks[0].attn.qkv.weight.device
        U, count = U.to(dev), count.to(dev); G, C = U.shape[0], U.shape[2]
        if rank is not None:
            # Rank-r SVD per group: the CLS side then reads G*r*(2C+1) instead of G*C*(C+1) weights.
            A = (Us[..., :rank] * S[:, None, :rank]).to(dev)                                   # [G,C+1,r]
            Bf = Vh[:, :rank].to(dev)                                                         # [G,r,C]
            U = A @ Bf                                                                        # reference path
            self.register_buffer('A', A.permute(1, 0, 2).reshape(C + 1, G * rank).contiguous(), persistent=False)
            self.register_buffer('Bf', Bf.contiguous(), persistent=False)
        self.rank = rank
        self.C, self.H, self.eps = C, H, eps
        # [C+1, G*C]: one GEMM gives every group's token-side vector for a CLS.
        self.register_buffer('U', U.permute(1, 0, 2).reshape(C + 1, G * C).contiguous(), persistent=False)
        self.register_buffer('log_count', count.log(), persistent=False)

    def prepare_inference(self, dtype):
        self.U = self.U.to(dtype)
        if self.rank is not None: self.A, self.Bf = self.A.to(dtype), self.Bf.to(dtype)

    def use_triton(self):
        """Pad the groups to a power of two >= 16 (tl.dot); padded groups get log|g| = -inf."""
        G = self.log_count.numel(); Gp = max(16, 1 << (G - 1).bit_length())
        if Gp > G:
            C = self.C
            self.U = torch.cat((self.U, self.U.new_zeros(C + 1, (Gp - G) * C)), 1)
            self.log_count = torch.cat((self.log_count, self.log_count.new_full((Gp - G,), -math.inf)))
            if self.rank is not None:
                self.A = torch.cat((self.A, self.A.new_zeros(C + 1, (Gp - G) * self.rank)), 1)
                self.Bf = torch.cat((self.Bf, self.Bf.new_zeros(Gp - G, self.rank, C)), 0)
        self.triton = True

    def zscored(self, x, size, raw_p=None, alpha=1.):
        """[B,N] alpha*z(D) [+ (1-alpha)*z(log raw_p)], CLS -> 0 (leaf.model.token_ops.zscore); alpha = 1: z(D) alone."""
        if not (getattr(self, 'triton', False) and x.is_cuda):
            out = alpha * zscore(self(x, size), 1)
            if raw_p is not None and alpha < 1: out = out + (1. - alpha) * zscore(raw_p.clamp_min(1e-12).log(), 1)
            return out
        from leaf.forecast.kernels.station_score import zscored_future
        B, N, C = x.shape
        cls = F.layer_norm(x[:, 0].float(), (C,), eps=self.eps).to(self.U.dtype)
        # Bias row split off: a K = C+1 (odd) GEMM falls back to an unaligned cutlass kernel (~5x slower).
        if self.rank is None:
            u = torch.addmm(self.U[C], cls, self.U[:C]).view(B, -1, C)
        else:
            t = torch.addmm(self.A[C], cls, self.A[:C]).view(B, -1, self.rank)            # [B,G,r]
            u = torch.bmm(t.transpose(0, 1), self.Bf).transpose(0, 1)                      # [B,G,C]
        return zscored_future(x, size, u, self.log_count, self.H, self.eps, raw_p, alpha)

    def forward(self, x, size):
        """[B,N] log of the group mixture (before z-scoring)."""
        B, N, C = x.shape
        hat = F.layer_norm(x.float(), (C,), eps=self.eps)
        cls = torch.cat((hat[:, 0], hat.new_ones(B, 1)), -1).to(self.U.dtype)
        u = (cls @ self.U).view(B, -1, C)                                                  # [B,G,C]
        logits = torch.bmm(hat.to(u.dtype), u.transpose(1, 2)).float()                     # [B,N,G]
        logits = logits + size.float().log()
        lc = self.log_count.to(logits.dtype)
        return (logits.log_softmax(1) + lc).logsumexp(-1) - math.log(self.H)
