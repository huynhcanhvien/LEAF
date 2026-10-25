"""Group the future (block, head) pairs of every station block by CLS-attention pattern (once per backbone).

For each block l: on the dense trajectory of `n` calib images, the token-wise Pearson correlation between
the log frozen-residual CLS-attention rows of every future pair (m > l, h), averaged over images; then
average-linkage clustering of 1 - corr into G groups (G capped by the number of pairs). Output:
<root>/groups.json {groups: {l: {G: labels (block-major, one per pair)}}}.
"""
import json

import numpy as np
import torch

from leaf.evaluation.data import iterate, load_split
from leaf.evaluation.runner import save
from leaf.forecast.future_attention import folded_rows

GS = (2, 4, 8, 16, 32)


@torch.no_grad()
def group_heads(vit, spec, out_path, n=1024, split='calib'):
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    images, labels = load_split(split); blocks = list(vit.blocks); L = len(blocks)
    sums = {l: None for l in range(L - 1)}; seen = 0
    for x, _ in iterate(images, labels, 64, 'cuda', spec.mean, spec.std, limit=n):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            h = vit.norm_pre(vit.patch_drop(vit._pos_embed(vit.patch_embed(x))))
            for l, blk in enumerate(blocks):
                mid = h + blk.ls1(blk.attn(blk.norm1(h)))
                if l < L - 1:
                    rows = folded_rows(blocks[l + 1:], mid.float(), None)               # [B,M,H,N] fp32
                    lp = rows.clamp_min(1e-30).log()[..., 1:].flatten(1, 2)           # [B,P,N-1] patch tokens
                    lp = (lp - lp.mean(-1, keepdim=True)) / lp.std(-1, keepdim=True, correction=0).clamp_min(1e-6)
                    corr = torch.einsum('bpn,bqn->pq', lp, lp) / lp.shape[-1]
                    sums[l] = corr if sums[l] is None else sums[l] + corr
                h = mid + blk.ls2(blk.mlp(blk.norm2(mid)))
        seen += x.shape[0]
    out = {}
    for l, s in sums.items():
        corr = (s / seen).double().cpu().numpy(); dist = np.clip(1 - corr, 0, None); np.fill_diagonal(dist, 0)
        tree = linkage(squareform((dist + dist.T) / 2, checks=False), 'average')
        out[str(l)] = {str(g): fcluster(tree, min(g, len(corr)), 'maxclust').tolist() for g in GS}
        out[str(l)]['mean_offdiag_corr'] = float(corr[~np.eye(len(corr), dtype=bool)].mean())
    save(out_path, dict(n_images=seen, split=split, backbone=spec.name, groups=out))
    return out
