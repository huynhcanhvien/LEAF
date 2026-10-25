"""Frozen ViT backbones and the facts the planner needs about them.

`load(name)` returns (vit, spec). Every backbone is a timm VisionTransformer at 224 with one CLS prefix token,
patch 16 (197 tokens) unless its entry sets `patch` and `n_tokens` (CLIP ViT-L/14: 14 and 257). MAE checkpoints
are the official fine-tuned weights (global average pooling + fc_norm), the ones ToMe and DiffRate report.

Eval images are the shared uint8 cache (Resize 256 bicubic -> CenterCrop 224, crop_pct 0.875);
only the normalization differs per backbone. `published_top1` is the source's dense number at its
own crop; the measured dense top-1 on our cache is reported next to it, not forced to match.

Segmentation backbones (task 'segmentation', pool 'dense') carry their decoder as `vit.head` and return per-patch
mask logits [B, classes, h, w] of one fixed square window of img_size (Segmenter, leaf/model/segmenter_head.py).

CLIP image towers (head 'clip_zeroshot') keep timm's projection and classify by cosine to the ImageNet
class embeddings of their text tower (leaf/model/clip_head.py), stored by `leaf data clip-text` as
<cache>/clip_text_<backbone>_imagenet.npy.
"""
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re

import torch

NORMALIZATION = {'imagenet': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), 'inception': ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                 'clip': ((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))}
IMAGENET, INCEPTION = NORMALIZATION['imagenet'], NORMALIZATION['inception']
REGISTRY = Path(__file__).with_name('backbones.json')


@dataclass(frozen=True)
class Spec:
    name: str              # short key, used in cache and result paths
    timm_id: str
    depth: int
    dim: int
    heads: int
    mean: tuple
    std: tuple
    pool: str = 'token'    # 'token' (CLS) | 'avg' (size-weighted mean of patch tokens, MAE)
    n_tokens: int = 197
    patch: int = 16        # patch side: the RGB embedding costs 3*patch*patch*dim MACs per patch token
    diffrate_key: str | None = None    # key in leaf/planner/diffrate_rates.json
    published_top1: float | None = None
    source: str = ''
    weights: str = 'timm'  # 'timm' or the URL of a state dict under 'model' (MAE fine-tuned checkpoints)
    aliases: tuple = ()
    head: str = 'linear'   # 'linear' (the checkpoint's classifier) | 'clip_zeroshot' (cosine to text embeddings)
                           # | 'segmenter_mask' (Segmenter's mask transformer on the patch grid)
    clip_text: tuple = ()  # open_clip (model, pretrained) of the text tower, for head 'clip_zeroshot'
    img_size: int = 224    # input side; segmentation backbones run on fixed square windows of this side
    task: str = 'classification'   # 'classification' | 'segmentation' (pool 'dense': per-patch logits [B, K, h, w])
    num_classes: int = 1000
    dataset: str = 'imagenet'      # segmentation: prefix of the cached splits (<dataset>_calib, _group, _val)
    count_step: int = 4            # token-count grid of the planner's DP (N, N - step, ..., 3)


def head_gmac(spec):
    """GMAC per image of what follows the last block: the classifier (classes x dim), or Segmenter's mask decoder on
    the full patch grid (projection, two blocks over patches + class embeddings, the two projections and the masks)."""
    d, k = spec.dim, getattr(spec, 'num_classes', 1000)
    if getattr(spec, 'head', 'linear') != 'segmenter_mask': return k * d / 1e9
    p = spec.n_tokens - 1; t = p + k
    return (2 * p * d * d + 2 * (12 * t * d * d + 2 * t * t * d) + k * d * d + p * k * d) / 1e9


def _registry_files():
    extra = [p for p in os.environ.get('LEAF_BACKBONES', '').split(':') if p]
    return [REGISTRY] + [Path(p) for p in extra]


def _specs():
    out = {}
    for path in _registry_files():
        for name, e in json.loads(Path(path).read_text())['backbones'].items():
            mean, std = NORMALIZATION[e.get('normalization', 'imagenet')]
            out[name] = Spec(name, e['timm_id'], e['depth'], e['dim'], e['heads'], mean, std, pool=e.get('pool', 'token'),
                             n_tokens=e.get('n_tokens', 197), patch=e.get('patch', 16), diffrate_key=e.get('diffrate_key'),
                             published_top1=e.get('published_top1'), source=e.get('source', ''),
                             weights=e.get('weights', 'timm'), aliases=tuple(e.get('aliases', ())),
                             head=e.get('head', 'linear'), clip_text=tuple(e.get('clip_text', ())),
                             img_size=e.get('img_size', 224), task=e.get('task', 'classification'),
                             num_classes=e.get('num_classes', 1000), dataset=e.get('dataset', 'imagenet'),
                             count_step=e.get('count_step', 4))
    return out


SPECS = _specs()


def _key(text): return re.sub(r'[^a-z0-9]+', '_', text.lower()).strip('_')


def resolve(name):
    """Backbone name from a name, an alias (case and '-'/'_' ignored) or a timm id (leaf/backbones.json, LEAF_BACKBONES)."""
    if name in SPECS: return name
    for spec in SPECS.values():
        if _key(name) in {_key(spec.name), *map(_key, spec.aliases)} or name == spec.timm_id: return spec.name
    raise KeyError(f'unknown backbone {name!r}; known: ' + ', '.join(
        f"{s.name} ({', '.join(s.aliases)})" if s.aliases else s.name for s in SPECS.values()))


def _download(url):
    path = os.path.join(torch.hub.get_dir(), 'checkpoints', os.path.basename(url))
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True); torch.hub.download_url_to_file(url, path)
    return path


def _from_url(spec):
    """timm model filled with a state dict downloaded from spec.weights (the official MAE fine-tuned checkpoints;
    Segmenter: MMSegmentation's checkpoint, converted by leaf.model.segmenter_head)."""
    import timm
    path = _download(spec.weights)
    if spec.head == 'segmenter_mask':
        from leaf.model.segmenter_head import load_segmenter
        return load_segmenter(spec, path)
    kw = dict(global_pool='avg', fc_norm=True) if spec.pool == 'avg' else {}
    vit = timm.create_model(spec.timm_id, pretrained=False, num_classes=1000, **kw)
    state = torch.load(path, map_location='cpu', weights_only=False)['model']
    missing, unexpected = vit.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f'checkpoint mismatch: missing {missing}, unexpected {unexpected}')
    return vit


def text_embeddings(spec, task='imagenet', cache_dir='data/cache'):
    """Text embeddings of a CLIP backbone written by `leaf data clip-text` ([classes or captions, dim], unit norm)."""
    import numpy as np
    path = Path(cache_dir) / f'clip_text_{spec.name}_{task}.npy'
    if not path.exists():
        raise FileNotFoundError(f'{path}: run `leaf data clip-text --backbone {spec.name} --task {task}` first')
    return torch.from_numpy(np.load(path).astype(np.float32))


def load(name, device='cuda', cache_dir='data/cache'):
    """Frozen eval-mode backbone and its Spec."""
    import timm
    spec = SPECS[resolve(name)]
    vit = timm.create_model(spec.timm_id, pretrained=True) if spec.weights == 'timm' else _from_url(spec)
    if spec.head == 'clip_zeroshot':
        from leaf.model.clip_head import ZeroShotHead
        vit.head = ZeroShotHead(vit.head, text_embeddings(spec, 'imagenet', cache_dir))
    vit = vit.to(device).eval().requires_grad_(False)
    if (len(vit.blocks), vit.embed_dim, vit.blocks[0].attn.num_heads, vit.patch_embed.num_patches + vit.num_prefix_tokens,
            tuple(vit.patch_embed.patch_size)) != (spec.depth, spec.dim, spec.heads, spec.n_tokens, (spec.patch, spec.patch)) or vit.num_prefix_tokens != 1:
        raise RuntimeError(f'{spec.name}: backbone does not match its entry in backbones.json')
    return vit, spec
