"""Segmenter's mask-transformer decoder as the head of a timm ViT, and the MMSegmentation checkpoint conversion.

The backbone is created with global_pool='' so that timm's forward_head hands the whole normed token sequence to
`vit.head`: SegmenterMaskHead drops the prefix tokens, decodes the patch grid and returns per-class mask logits
[B, classes, h, w] on the patch grid (1/16 of the window). Upsampling and sliding-window stitching are left to the
evaluation (leaf/evaluation/segmentation.py), as in MMSegmentation's slide inference.

Architecture and weight layout follow MMSegmentation's SegmenterMaskTransformerHead (Apache-2.0), a port of
Segmenter (Strudel et al., ICCV 2021; MIT). See THIRD_PARTY_NOTICES.md.
"""
from functools import partial
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmenterMaskHead(nn.Module):
    def __init__(self, dim, n_classes, depth=2, heads=12, prefix=1):
        super().__init__()
        from timm.models.vision_transformer import Block
        norm = partial(nn.LayerNorm, eps=1e-5)             # MMSegmentation's LN default (the backbone uses 1e-6)
        self.prefix, self.n_classes = prefix, n_classes
        self.dec_proj = nn.Linear(dim, dim)
        self.cls_emb = nn.Parameter(torch.zeros(1, n_classes, dim))
        self.blocks = nn.ModuleList(Block(dim, heads, mlp_ratio=4., qkv_bias=True, norm_layer=norm) for _ in range(depth))
        self.decoder_norm = norm(dim)
        self.patch_proj = nn.Linear(dim, dim, bias=False)
        self.classes_proj = nn.Linear(dim, dim, bias=False)
        self.mask_norm = norm(n_classes)

    def forward(self, x):
        """[B, prefix + h*w, C] normed tokens -> [B, classes, h, w] mask logits."""
        x = x[:, self.prefix:]
        b, n, _ = x.shape
        side = math.isqrt(n)
        if side * side != n: raise ValueError(f'{n} patch tokens do not form a square grid')
        x = torch.cat((self.dec_proj(x), self.cls_emb.expand(b, -1, -1).to(x.dtype)), 1)
        for blk in self.blocks: x = blk(x)
        x = self.decoder_norm(x)
        patches = F.normalize(self.patch_proj(x[:, :-self.n_classes]), dim=-1)
        classes = F.normalize(self.classes_proj(x[:, -self.n_classes:]), dim=-1)
        masks = self.mask_norm(patches @ classes.transpose(1, 2))
        return masks.transpose(1, 2).reshape(b, self.n_classes, side, side)


def _layer(prefix, out_prefix):
    """MMCV transformer layer -> timm Block key map (pre-norm, packed q/k/v, two-layer FFN)."""
    return {f'{prefix}ln1.': f'{out_prefix}norm1.', f'{prefix}ln2.': f'{out_prefix}norm2.',
            f'{prefix}attn.attn.in_proj_weight': f'{out_prefix}attn.qkv.weight',
            f'{prefix}attn.attn.in_proj_bias': f'{out_prefix}attn.qkv.bias',
            f'{prefix}attn.attn.out_proj.': f'{out_prefix}attn.proj.',
            f'{prefix}ffn.layers.0.0.': f'{out_prefix}mlp.fc1.', f'{prefix}ffn.layers.1.': f'{out_prefix}mlp.fc2.'}


def convert_mmseg(state, depth, head_depth):
    """MMSegmentation EncoderDecoder(VisionTransformer, SegmenterMaskTransformerHead) state dict -> timm ViT keys,
    with the decoder under 'head.'."""
    rules = {'backbone.cls_token': 'cls_token', 'backbone.pos_embed': 'pos_embed',
             'backbone.patch_embed.projection.': 'patch_embed.proj.', 'backbone.ln1.': 'norm.',
             'decode_head.cls_emb': 'head.cls_emb', 'decode_head.dec_proj.': 'head.dec_proj.',
             'decode_head.patch_proj.': 'head.patch_proj.', 'decode_head.classes_proj.': 'head.classes_proj.',
             'decode_head.decoder_norm.': 'head.decoder_norm.', 'decode_head.mask_norm.': 'head.mask_norm.'}
    for i in range(depth): rules.update(_layer(f'backbone.layers.{i}.', f'blocks.{i}.'))
    for i in range(head_depth): rules.update(_layer(f'decode_head.layers.{i}.', f'head.blocks.{i}.'))
    out = {}
    for key, value in state.items():
        match = [src for src in rules if key.startswith(src)]
        if len(match) != 1: raise KeyError(f'no unique conversion rule for {key}: {match}')
        out[rules[match[0]] + key[len(match[0]):]] = value
    return out


def load_segmenter(spec, path):
    """timm ViT at spec.img_size with SegmenterMaskHead as head, filled from an MMSegmentation checkpoint."""
    import timm
    vit = timm.create_model(spec.timm_id, pretrained=False, img_size=spec.img_size, num_classes=0, global_pool='')
    vit.head = SegmenterMaskHead(spec.dim, spec.num_classes, depth=2, heads=spec.heads)
    state = torch.load(path, map_location='cpu', weights_only=False)
    state = convert_mmseg(state.get('state_dict', state), spec.depth, 2)
    missing, unexpected = vit.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f'checkpoint mismatch: missing {missing}, unexpected {unexpected}')
    return vit
