"""Frozen-residual future CLS attention: reference math for D and for grouping (block, head) pairs.

Under the proxy "token j's residual stays fixed after the station", the CLS attention row of future
block m, head h is softmax_j(s_m q_mh(LN_m(c))·k_mh(LN_m(x_j)) + log size_j). `folded_rows` returns
these rows for every future (m, h); `grouped_log_rows` turns them into the log-domain grouped D that
leaf.forecast.attention_forecast.AttentionForecast computes in closed form.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def pack_blocks(blocks):
    if not blocks:
        raise ValueError("at least one future block is required")
    for b in blocks:
        if not isinstance(b.attn.q_norm, nn.Identity) or not isinstance(b.attn.k_norm, nn.Identity):
            raise ValueError("folding requires identity Q/K normalization")
        if b.norm1.eps != blocks[0].norm1.eps:
            raise ValueError("shared normalization requires equal LayerNorm eps")
    C = blocks[0].attn.qkv.in_features
    H, d = blocks[0].attn.num_heads, blocks[0].attn.head_dim
    zeros = lambda *s: blocks[0].attn.qkv.weight.new_zeros(*s)
    return dict(
        wq=torch.stack([b.attn.qkv.weight[:C] for b in blocks]),
        wk=torch.stack([b.attn.qkv.weight[C:2 * C].reshape(H, d, C) for b in blocks]),
        bq=torch.stack([b.attn.qkv.bias[:C] if b.attn.qkv.bias is not None else zeros(C) for b in blocks]),
        bk=torch.stack([b.attn.qkv.bias[C:2 * C].reshape(H, d) if b.attn.qkv.bias is not None else zeros(H, d) for b in blocks]),
        gamma=torch.stack([b.norm1.weight for b in blocks]),
        beta=torch.stack([b.norm1.bias for b in blocks]),
        scale=torch.tensor([b.attn.scale for b in blocks], device=blocks[0].attn.qkv.weight.device),
    )


def folded_rows(blocks, x, log_s=None):
    """[B, M, H, N] future CLS attention probabilities of every (future block, head), fp32 math."""
    p = pack_blocks(blocks)
    B, N, C = x.shape
    M, H, d, _ = p['wk'].shape
    hat = F.layer_norm(x, (C,), eps=blocks[0].norm1.eps)
    affine_tokens = blocks[0].norm1(x) if M == 1 else None
    cls = affine_tokens[:, :1] if M == 1 else hat[:, :1] * p['gamma'][None] + p['beta'][None]
    q = torch.stack([F.linear(cls[:, m], p['wq'][m], p['bq'][m]) for m in range(M)], 1).reshape(B, M, H, d)
    q = q * p['scale'][None, :, None, None].to(q.dtype)
    with torch.autocast(x.device.type, enabled=False):
        wk = p['wk'].to(q.dtype).float() if q.dtype in (torch.bfloat16, torch.float16) else p['wk'].float()
        u = torch.einsum('bmhd,mhdc->bcmh', q.float(), wk)
        if M == 1:
            token_input = affine_tokens.to(q.dtype).float(); offset = p['bk'].to(q.dtype).float()
        else:
            u = u * p['gamma'].T[None, :, :, None].float()
            token_input = hat.float()
            offset = torch.einsum('mc,mhdc->mhd', p['beta'].float(), wk) + p['bk'].to(q.dtype).float()
        logits = (token_input @ u.reshape(B, C, M * H)).reshape(B, N, M, H).permute(0, 2, 3, 1)
        logits = logits + torch.einsum('bmhd,mhd->bmh', q.float(), offset)[..., None]
    logits = logits.to(q.dtype)
    if log_s is not None:
        logits = logits + log_s.reshape(B, 1, 1, N).to(logits.dtype)
    return logits.float().softmax(-1)


def grouped_log_rows(rows, labels):
    """log of the group-size-weighted mixture of renormalized geometric means over tokens.

    rows: [B,M,H,N] probabilities; labels: one group per (block, head), block-major."""
    B, M, H, N = rows.shape
    lab = torch.unique(torch.as_tensor(labels, device=rows.device), return_inverse=True)[1]
    if lab.numel() != M * H: raise ValueError('one group label per future (block, head) pair')
    onehot = F.one_hot(lab, int(lab.max()) + 1).float()
    count = onehot.sum(0)
    lp = rows.float().clamp_min(1e-30).log().reshape(B, M * H, N)
    geo = torch.einsum('bpn,pg->bgn', lp, onehot) / count[None, :, None]
    geo = geo - geo.logsumexp(-1, keepdim=True)
    return (geo + count.log()[None, :, None]).logsumexp(1) - math.log(H)
