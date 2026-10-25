"""Baseline adapters (baselines/methods.py) == the official implementations (ToMe, PiToMe timm patches; MCTF, EViT,
PPT model code), CPU fp32, DeiT-S, 2 random images.
Needs the official repos (OFFICIAL_REPOS=<dir with ToMe/ and PiToMe/>); skipped otherwise.
    OFFICIAL_REPOS=/path/to/repos python tests/test_baselines.py"""
from common import check, finish
import copy, os, sys
import torch

from baselines import build
from leaf.backbones import SPECS


def main():
    root = os.environ.get('OFFICIAL_REPOS')
    if not root: print('SKIP: OFFICIAL_REPOS not set'); return
    import timm
    torch.manual_seed(0)
    vit = timm.create_model('deit_small_patch16_224.fb_in1k', pretrained=True).eval(); spec = SPECS['deit_s']
    x = torch.randn(2, 3, 224, 224)
    sys.path.insert(0, root + '/ToMe')
    import tome
    for r in (8, 13):
        m = copy.deepcopy(vit); tome.patch.timm(m); m.r = r
        with torch.no_grad(): a, b = m(x), build(vit, spec, dict(kind='tome', r=r))(x)
        check(f'ToMe r={r} == official timm patch', torch.allclose(a, b, atol=1e-3), f'{(a - b).abs().max():.1e}')
    # Load algo.pitome.patch.deit without the package __init__ files (they import transformers-based patches).
    import types
    for name, sub in (('algo', 'algo'), ('algo.pitome', 'algo/pitome'), ('algo.pitome.patch', 'algo/pitome/patch')):
        mod = types.ModuleType(name); mod.__path__ = [f'{root}/PiToMe/{sub}']; sys.modules[name] = mod
    from algo.pitome.patch.deit import apply_patch
    for q in (.9, .85):
        m = copy.deepcopy(vit); apply_patch(m); m.ratio = q
        # Their forward targets an older timm (no attn_mask, dist_token, pre_logits): run its body directly.
        import math, torch.nn as nn
        m.dist_token = None; m.pre_logits = nn.Identity()
        m._info['ratio'] = [q] * len(m.blocks); nb = math.ceil(len(m.blocks) * .5)
        m._info['use_bsm_pitome'] = [True] * nb + [False] * (len(m.blocks) - nb); m._info['size'] = None; m._info['source'] = None; m.total_flop = 0
        with torch.no_grad(): a, b = m.head(m.forward_features(x)), build(vit, spec, dict(kind='pitome', ratio=q))(x)
        check(f'PiToMe ratio={q} == official DeiT patch', torch.allclose(a, b, atol=1e-3), f'{(a - b).abs().max():.1e}')


    # MCTF / EViT / PPT ship their own model classes and top-level packages: compare in a separate process.
    import subprocess
    for which in ('mctf', 'evit', 'ppt'):
        out = subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), 'official_compare.py'), root, which],
                             capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES='')).stdout
        for line in out.splitlines():
            if line.startswith(which):
                name, err = line.rsplit(' ', 1)
                check(f'{name} == official implementation', float(err) < 1e-3, f'{float(err):.1e}')


if __name__ == '__main__':
    main(); finish()
