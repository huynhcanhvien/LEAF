"""Token operations shared by the method and the DiffRate baseline.

cls_attention: one encoder attention sublayer (proportional attention: + log size on the keys)
               that also returns the head-mean CLS attention row P used for scoring.
compress:      DiffRate's inference operator: sort by score -> prune to n_prune -> merge the
               n_prune - n_merge lowest tokens into their most similar kept token (mean of x,
               similarity-weighted size).
zscore:        population z-score over patch tokens (prefix tokens -> 0).
"""
import math

import torch
import torch.nn.functional as F


def zscore(v, start):
    """z-score over token positions >= start (patch tokens); fixed tokens get 0."""
    body = v[:, start:]
    mean = body.mean(dim=1, keepdim=True)
    std = body.std(dim=1, keepdim=True, correction=0).clamp_min(1e-6)
    out = torch.zeros_like(v)
    out[:, start:] = (body - mean) / std
    return out


def cls_attention(blk, x, size):
    """Attention sublayer output and the head-mean CLS attention row [B, N] (fp32)."""
    attn = blk.attn
    b, n, c = x.shape
    q, k, v = attn.qkv(x).reshape(b, n, 3, attn.num_heads, attn.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
    q, k = attn.q_norm(q), attn.k_norm(k)
    bias = None if size is None else size.log().view(b, 1, 1, n).to(q.dtype)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, scale=attn.scale)
    out = out.transpose(1, 2).reshape(b, n, c)
    out = attn.proj_drop(attn.proj(attn.norm(out) if hasattr(attn, "norm") else out))
    logits = (q[:, :, :1] * attn.scale) @ k.transpose(-2, -1)
    if bias is not None:
        logits = logits + bias
    return out, logits.float().softmax(-1).squeeze(2).mean(1)


def compress(x, size, scores, n_prune, n_merge, fuse_merge=True):
    """DiffRate prune -> mean merge with similarity-scaled size. Returns (x, size, order, destination).

    Token 0 (CLS) is always kept and never a merge destination."""
    b, n, c = x.shape
    n_prune, n_merge = min(n, n_prune), min(n, n_prune, n_merge)
    order = scores[:, 1:].argsort(-1, descending=True) + 1
    order = torch.cat([torch.zeros(b, 1, dtype=torch.long, device=x.device), order], 1)[:, :n_prune]
    x = x.gather(1, order[..., None].expand(-1, -1, c))
    size = size.gather(1, order[..., None])
    destination = None
    if n_merge < n_prune:
        metric = F.normalize(x.float(), dim=-1)
        similarity = metric[:, n_merge:] @ metric[:, :n_merge].transpose(1, 2)
        similarity[..., 0] = -math.inf
        maximum, destination = similarity.max(-1)
        # Similarity-scaled size turns negative when many tokens merge into few destinations with
        # best cosine < 0 (log size -> NaN). A source with negative best similarity adds no size.
        maximum = maximum.clamp_min(0)
        needs_grad = torch.is_grad_enabled() and (x.requires_grad or size.requires_grad)
        if fuse_merge and x.device.type == 'cuda' and x.dtype == torch.float32 and not needs_grad:
            from leaf.model.kernels.mean_merge import merge_inference
            x, size = merge_inference(x, size, destination, maximum, n_merge)
            return x, size, order, destination
        dst = destination[..., None]
        x = x[:, :n_merge].scatter_reduce(1, dst.expand(-1, -1, c), x[:, n_merge:], reduce="mean")
        size = size[:, :n_merge].scatter_reduce(1, dst, size[:, n_merge:] * maximum[..., None].to(size.dtype), reduce="sum")
    return x, size, order, destination
