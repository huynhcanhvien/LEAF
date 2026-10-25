"""Experimental station merges. M0 delegates to the unchanged DiffRate operator.

All new reductions sort by (batch, destination) and use segment_reduce: no floating
point atomics. FP32 accumulation is used even for BF16 residuals. Ties follow token
order. CLS is never a source or destination. Pruning is identical to compress.
"""
import math

import torch
import torch.nn.functional as F

from leaf.model.token_ops import cls_attention, compress

MODES = ('M0', 'M1', 'M2', 'M3', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6',
         'N1+N2', 'N2+N5', 'M2+N1')


def cls_attention_with_keys(blk, x, size):
    """Existing attention result plus normalized-attention keys, averaged over heads.

    Compute QKV once; the attention math and operation order match cls_attention.
    Keys precede the residual update, just as in the block's actual attention.
    """
    attn = blk.attn
    b, n, c = x.shape
    q, k, v = attn.qkv(x).reshape(b, n, 3, attn.num_heads, attn.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
    q, k = attn.q_norm(q), attn.k_norm(k)
    bias = None if size is None else size.log().view(b, 1, 1, n).to(q.dtype)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, scale=attn.scale)
    out = out.transpose(1, 2).reshape(b, n, c)
    out = attn.proj_drop(attn.proj(attn.norm(out) if hasattr(attn, 'norm') else out))
    logits = (q[:, :, :1] * attn.scale) @ k.transpose(-2, -1)
    if bias is not None:
        logits = logits + bias
    return out, logits.float().softmax(-1).squeeze(2).mean(1), k.float().mean(1)


def gather(x, order):
    return x.gather(1, order if x.ndim == 2 else order[..., None].expand(-1, -1, x.shape[-1]))


def segment(values, targets, count, reduce='sum'):
    """[B,E,C] -> [B,count,C], in stable input order within each segment."""
    b, _, c = values.shape
    ids = targets + torch.arange(b, device=values.device)[:, None] * count
    ids = ids.flatten()
    order = ids.argsort(stable=True)
    lengths = torch.bincount(ids, minlength=b * count)
    result = torch.segment_reduce(values.reshape(-1, c)[order], reduce, lengths=lengths)
    return result.reshape(b, count, c)


def _cosine(x, n_merge):
    metric = F.normalize(x.float(), dim=-1)
    sim = metric[:, n_merge:] @ metric[:, :n_merge].transpose(1, 2)
    sim[..., 0] = -math.inf
    return sim


def _set_norm(mean, members, targets, count, norm):
    """Use the largest member's direction if the mean cancels to exactly zero."""
    lengths = members.norm(dim=-1, keepdim=True)
    peak = segment(lengths, targets, count, 'max')
    index = torch.arange(members.shape[1], device=members.device).float()[None, :, None].expand(members.shape[0], -1, -1)
    first = segment(torch.where(lengths == gather(peak, targets), index, torch.inf), targets, count, 'min')
    fallback = gather(members, first.squeeze(-1).long())
    direction = torch.where(mean.norm(dim=-1, keepdim=True) > 0, mean, fallback)
    return direction * (norm / direction.norm(dim=-1, keepdim=True).clamp_min(1e-30))


def _n6_order(x, scores, order, n_merge, candidate_factor):
    """Choose low-cost sources within the low-score pool, retaining one patch anchor.

    Costs use the best cosine to the guaranteed destinations outside that pool.
    Recompute final assignments after source selection; output order stays by score.
    candidate_factor=1 recovers the original source set and order exactly.
    """
    d = order.shape[1] - n_merge
    pool = min(candidate_factor * d, order.shape[1] - 2)
    split = order.shape[1] - pool
    ordered = gather(x, order)
    sim = _cosine(ordered, split)
    rho = sim.max(-1).values
    # Subtract a common score offset to avoid exp overflow without changing rank.
    s = gather(scores, order)[:, split:]
    cost = (s - s.max(1, keepdim=True).values).exp() * (1 - rho)
    chosen = cost.argsort(dim=1, stable=True)[:, :d] + split
    mask = torch.zeros_like(order, dtype=torch.bool).scatter(1, chosen, True)
    # Stable partition: destinations followed by sources, both in original rank order.
    return order.gather(1, mask.to(torch.int64).argsort(dim=1, stable=True))


def compress_variant(x, size, scores, n_prune, n_merge, mode='M0', *, keys=None,
                     forecast=None, tau=1., candidate_factor=2, fuse_merge=True):
    """Return (residual, attention_size, retained_order, source_destination).

    forecast=(a, eps) supplies a single forecast logit vector [B,C] for N2.
    It is the head-count-weighted mean of D's group vectors; absent means M1 size.
    N5 returns [B,sources,2] destinations (one when only one patch destination).
    M3: unweighted mean, rescaled to max member norm; M0 similarity size.
    N4: preserve size-weighted channel mean and weighted centered-vector norm.
    A cancelled mean uses the largest member's direction (stable first-member tie).
    """
    if mode not in MODES:
        raise ValueError(f'unknown merge mode: {mode}')
    if mode == 'M0':
        return compress(x, size, scores, n_prune, n_merge, fuse_merge=fuse_merge)
    if mode.startswith('M2') and keys is None:
        raise ValueError('M2 requires current attention keys')
    if candidate_factor not in (1, 2):
        raise ValueError('candidate_factor must be 1 or 2')
    b, n, _ = x.shape
    p, m = min(n, n_prune), min(n, n_prune, n_merge)
    if m < 2 or p < m:
        raise ValueError('merge needs CLS and at least one patch destination')
    order = torch.cat((torch.zeros(b, 1, device=x.device, dtype=torch.long),
                       scores[:, 1:].argsort(-1, descending=True) + 1), 1)[:, :p]
    if mode == 'N6' and m < p:
        order = _n6_order(x, scores, order, m, candidate_factor)
    y, s = gather(x, order), gather(size, order)
    if p == m:
        return y, s, order, None
    sim = _cosine(gather(keys, order) if mode.startswith('M2') else y, m)
    rho, dest = sim.max(-1)
    targets = torch.arange(m, device=x.device)[None].expand(b, -1)
    fraction = torch.ones(b, p, 1, device=x.device)
    values, sizes, score = y.float(), s.float(), gather(scores, order).float()
    if 'N5' in mode:
        # Stable ties choose the lowest destination index, like max for hard merges.
        dest = sim.argsort(dim=-1, descending=True, stable=True)[..., :min(2, m - 1)]
        r = sim.gather(-1, dest).clamp_min(0)
        total = r.sum(-1, keepdim=True)
        fallback = torch.zeros_like(r); fallback[..., 0] = 1
        f = torch.where(total > 0, r / total.clamp_min(1e-30), fallback)
        k = dest.shape[-1]
        values = torch.cat((values[:, :m], values[:, m:, None].expand(-1, -1, k, -1).flatten(1, 2)), 1)
        sizes = torch.cat((sizes[:, :m], sizes[:, m:, None].expand(-1, -1, k, -1).flatten(1, 2)), 1)
        score = torch.cat((score[:, :m], score[:, m:, None].expand(-1, -1, k).flatten(1, 2)), 1)
        fraction = torch.cat((fraction[:, :m], f.flatten(1)[..., None]), 1)
        targets = torch.cat((targets, dest.flatten(1)), 1)
    else:
        targets = torch.cat((targets, dest), 1)
    mass = sizes * fraction
    total_mass = segment(mass, targets, m)
    if mode in ('M2', 'M3'):
        weight = fraction
    elif 'N1' in mode:
        # Global offset cancels in each mean; group normalization avoids underflow.
        peak = segment(score[..., None], targets, m, 'max')
        weight = mass * (tau * (score[..., None] - gather(peak, targets))).exp()
    else:
        weight = mass
    merged = segment(values * weight, targets, m) / segment(weight, targets, m)
    if mode == 'M3':
        norm = segment(values.norm(dim=-1, keepdim=True), targets, m, 'max')
        merged = _set_norm(merged, values, targets, m, norm)
    if mode == 'N4':
        center = merged.mean(-1, keepdim=True)
        centered = merged - center
        norms = (values - values.mean(-1, keepdim=True)).norm(dim=-1, keepdim=True)
        norm = segment(norms * mass, targets, m) / total_mass
        members = values - values.mean(-1, keepdim=True)
        merged = center + _set_norm(centered, members, targets, m, norm)
    if mode == 'N3':
        merged = y[:, :m].float()
    new_size = total_mass
    if mode in ('M2', 'M3', 'N3'):
        size_weight = torch.cat((torch.ones(b, m, 1, device=x.device), rho.clamp_min(0)[..., None]), 1)
        new_size = segment(s.float() * size_weight, targets, m)
    if 'N2' in mode and forecast is not None:
        a, eps = forecast
        lam = (F.layer_norm(values, (values.shape[-1],), eps=eps) * a.float()[:, None]).sum(-1, keepdim=True)
        after = (F.layer_norm(merged, (merged.shape[-1],), eps=eps) * a.float()[:, None]).sum(-1, keepdim=True)
        logs = mass.log() + lam
        peak = segment(logs, targets, m, 'max')
        logmass = segment((logs - gather(peak, targets)).exp(), targets, m).log() + peak
        new_size = (logmass - after).exp()
    # Preserve untouched destinations exactly, including CLS and low precision inputs.
    sources = segment(fraction[:, m:], targets[:, m:], m) > 0
    merged = torch.where(sources, merged.to(x.dtype), y[:, :m])
    new_size = torch.where(sources, new_size.to(size.dtype), s[:, :m])
    return merged, new_size, order, dest
