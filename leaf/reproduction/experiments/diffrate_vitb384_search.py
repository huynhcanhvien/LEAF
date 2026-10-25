"""DiffRate's own rate search on ViT-B/16 at 384 px (augreg_b384), like diffrate_clip_search.py on CLIP: the official
DiffRate code (same commit, same 3-epoch recipe, batch 1024 by gradient accumulation, frozen weights) applied to the
timm classifier with its ImageNet head, 384-px training augmentation (timm create_transform at 384) and ImageNet
normalisation. Targets are GFLOPs in DiffRate's convention; after the search the schedules are costed with LEAF's GMAC
convention and LEAF is planned at that GMAC (vitb384_comparison.py plan-dr-search).

Stages (python leaf/reproduction/experiments/diffrate_vitb384_search.py <stage>; results/diffrate_vitb384_search/):
  search  (GPU) --budgets 27.1 35.2 [--micro 32] [--max-steps N for a smoke test]
  rates   (CPU) collect -> rates.json (key 'ViT-B384-searched')
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
REGISTRY = Path(__file__).resolve().parent / 'backbones_vitb384.json'
os.environ['LEAF_BACKBONES'] = ':'.join([p for p in os.environ.get('LEAF_BACKBONES', '').split(':') if p and p != str(REGISTRY)] + [str(REGISTRY)])
sys.path.insert(0, str(REPO))

from leaf.reproduction.experiments import diffrate_clip_search as D

BB = 'augreg_b384'
ROOT = REPO / 'results/diffrate_vitb384_search'
BUDGETS = ('23.8', '27.1', '30.8', '35.2')   # GMAC of the LEAF latency schedules at 0.48x, 0.56x, 0.64x and 0.72x the dense latency
KEY = 'ViT-B384-searched'
D.BB, D.ROOT, D.BUDGETS = BB, ROOT, BUDGETS


def build(drop_path=0., device='cuda'):
    """The timm ViT-B/16-384 classifier patched by the official DiffRate code; only the rates are trainable."""
    import torch.nn as nn
    from timm.layers import DropPath
    from leaf.backbones import load
    deit = D.diffrate_modules()
    vit, spec = load(BB, device=device)
    for i, blk in enumerate(vit.blocks):
        if not all(isinstance(m, nn.Identity) for m in (blk.ls1, blk.ls2, blk.attn.q_norm, blk.attn.k_norm, getattr(blk.attn, 'norm', nn.Identity()))):
            raise RuntimeError('the DiffRate adapter assumes no LayerScale and no q/k norm')
        rate = drop_path * i / (len(vit.blocks) - 1)
        blk.drop_path1 = blk.drop_path2 = DropPath(rate) if rate > 0 else nn.Identity()
    deit.apply_patch(vit, prune_granularity=D.HP['granularity'], merge_granularity=D.HP['granularity'])
    for blk in vit.blocks: blk.drop_path = blk.drop_path1
    vit.num_classes = vit.head.out_features                   # classifier FLOPs in DiffRate's count (1000 classes)
    vit.to(device)
    for n, p in vit.named_parameters(): p.requires_grad_('ddp' in n and n.endswith('selected_probability'))
    return vit, spec


def train_transform():
    from timm.data import create_transform
    from leaf.backbones import SPECS
    spec = SPECS[BB]
    return create_transform(spec.img_size, is_training=True, color_jitter=D.HP['color_jitter'], auto_augment=D.HP['aa'],
                            interpolation='bicubic', re_prob=D.HP['reprob'], re_mode='pixel', re_count=1, mean=spec.mean, std=spec.std)


D.build, D.train_transform = build, train_transform


def rates():
    out = {}
    for b in BUDGETS:
        p = ROOT / b / 'rates.json'
        if p.exists():
            r = json.loads(p.read_text()); out[b] = dict(prune_kept_num=json.dumps(r['prune_kept_num']), merge_kept_num=json.dumps(r['merge_kept_num']), gflops=r['gflops'])
    D.save(ROOT / 'rates.json', dict(source=f'DiffRate search on {BB} ({D.DIFFRATE_URL} @ {D.DIFFRATE_COMMIT})', hp=D.HP, rates={KEY: out}, time=time.time()))
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=('search', 'rates'))
    ap.add_argument('--budgets', nargs='*', default=list(BUDGETS))
    ap.add_argument('--micro', type=int, default=32, help='micro-batch (384 px: 32 fits a 16 GB card; global batch stays 1024)')
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--max-steps', type=int, default=None)
    a = ap.parse_args()
    if a.stage == 'search':
        for b in a.budgets: D.search(b, a.micro, a.workers, max_steps=a.max_steps)
    else: rates()


if __name__ == '__main__':
    main()
