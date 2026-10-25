"""Shared frame for re-run baselines: one frozen timm ViT, a per-method token-reduction step between each
block's attention and MLP (where every compared method reduces), the same head and pooling as the method.

Each baseline supplies:
    needs:   set of what its reduction reads from attention: 'kmean' (head-mean keys, ToMe/PiToMe),
             'cls' (head-mean CLS attention row, EViT), 'clsh' (per-head CLS row [B,H,N], PPT),
             'attn' (full attention probabilities); q, k, v are always returned
    prop:    proportional attention (+ log size on the keys)
    reduce(l, x, size, info) -> (x, size): the paper's algorithm, from the vendored official code
Attention uses F.scaled_dot_product_attention for every method (same math as the official explicit
softmax), unless the method needs the full probability matrix, so latency compares algorithms, not
attention kernels.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def attention(blk, x, size, needs, prop):
    """Attention sublayer output and what the reduction needs."""
    attn = blk.attn
    b, n, c = x.shape
    q, k, v = attn.qkv(x).reshape(b, n, 3, attn.num_heads, attn.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
    q, k = attn.q_norm(q), attn.k_norm(k)
    bias = size.log().view(b, 1, 1, n).to(q.dtype) if prop and size is not None else None
    info = {}
    if 'attn' in needs:
        logits = (q * attn.scale) @ k.transpose(-2, -1)
        if bias is not None: logits = logits + bias
        probs = logits.softmax(-1)
        out = probs @ v; info['attn'] = probs
    else:
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, scale=attn.scale)
        if 'cls' in needs or 'clsh' in needs:
            logits = (q[:, :, :1] * attn.scale) @ k.transpose(-2, -1)
            if bias is not None: logits = logits + bias
            rows = logits.softmax(-1).squeeze(2)
            info['clsh'] = rows; info['cls'] = rows.float().mean(1)
    if 'kmean' in needs: info['kmean'] = k.mean(1)
    out = out.transpose(1, 2).reshape(b, n, c)
    out = attn.proj_drop(attn.proj(attn.norm(out) if hasattr(attn, 'norm') else out))
    info['k'] = k; info['q'] = q; info['v'] = v
    return out, info


class TokenViT(nn.Module):
    needs = ('kmean',)
    prop = True

    def __init__(self, vit, spec):
        super().__init__()
        self.vit, self.spec = vit, spec
        self.depth = len(vit.blocks)

    def reduce(self, l, x, size, info):
        return x, size

    def start(self, x):
        """Per-forward state reset (python-side only: CUDA graphs replay kernels)."""

    tracks = False      # True: reduce() records every step in self.sources (dense heads, leaf.model.dense)

    def forward(self, img):
        vit = self.vit
        x = vit.norm_pre(vit.patch_drop(vit._pos_embed(vit.patch_embed(img))))
        size = x.new_ones(x.shape[0], x.shape[1], 1); self.start(x)
        self.sources = None
        if self.spec.pool == 'dense':
            if not self.tracks: raise NotImplementedError(f'{type(self).__name__} has no dense (segmentation) output')
            from leaf.model.dense import SourceMap
            self.sources = SourceMap(x)
        for l, blk in enumerate(vit.blocks):
            out, info = attention(blk, blk.norm1(x), size, self.needs, self.prop)
            x = x + blk.drop_path1(blk.ls1(out))
            x, size = self.reduce(l, x, size, info)
            x = x + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(x))))
        x = vit.norm(x)
        if self.spec.pool == 'dense':       # every input position (carrier's token), then the decoder
            return vit.head(self.sources.reconstruct(x))
        if self.spec.pool == 'avg':
            w = size[:, 1:]; x = (x[:, 1:] * w).sum(1) / w.sum(1)
        else:
            x = x[:, 0]
        return vit.head(vit.head_drop(vit.fc_norm(x)))
