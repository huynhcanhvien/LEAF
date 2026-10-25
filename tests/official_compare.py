"""Helper of tests/test_baselines.py, run in its own process (the official repos have top-level packages named
models/utils that clash with ours): compare one official implementation with our adapter on DeiT-S, CPU fp32.
    python tests/official_compare.py <repos dir> mctf|evit|ppt"""
import os, sys, types, copy
import torch
import timm

root, which = sys.argv[1], sys.argv[2]
here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
torch.manual_seed(0)
vit = timm.create_model('deit_small_patch16_224.fb_in1k', pretrained=True).eval()
x = torch.randn(2, 3, 224, 224)
spec = types.SimpleNamespace(pool='token', depth=12)


def ours(cfg):
    sys.path.insert(0, here)
    from baselines.methods import EViT, MCTF, PPT
    cls = dict(mctf=MCTF, evit=EViT, ppt=PPT)[cfg.pop('kind')]
    return cls(vit, spec, *cfg.values())


if which == 'mctf':
    sys.path.insert(0, f'{root}/MCTF')
    from models.deit import deit_small
    for r in (8, 16):
        m = deit_small(use_mctf=True, mctf_type=[r, 0, 0, 1, 1, 1, 20, 40, 1, 1, 0], task_type=[1, 0, 0, 0],
                       activate_layer=list(range(1, 12)), r_evals=[r]).eval()
        print(m.load_state_dict(vit.state_dict(), strict=False))
        with torch.no_grad():
            out = m(x, r_eval=r); a = out['x_student'] if isinstance(out, dict) else out
            b = ours(dict(kind='mctf', r=r))(x)
        print(f'mctf r={r}', (a - b).abs().max().item())
elif which == 'evit':
    sys.path.insert(0, f'{root}/evit')
    sys.modules['torchprofile'] = types.SimpleNamespace(profile_macs=None)   # only used to count MACs
    import evit as E
    for q in (.7, .5):
        m = E.EViT(patch_size=16, embed_dim=384, depth=12, num_heads=6, qkv_bias=True,
                   keep_rate=tuple(q if i in (3, 6, 9) else 1 for i in range(12)), fuse_token=True).eval()
        print(m.load_state_dict(vit.state_dict(), strict=False))
        with torch.no_grad():
            a = m(x); a = a[0] if isinstance(a, tuple) else a; b = ours(dict(kind='evit', keep_rate=q))(x)
        print(f'evit keep={q}', (a - b).abs().max().item())
elif which == 'ppt':
    sys.path.insert(0, f'{root}/PPT')
    from ppt_deit import apply_patch
    for r in (30, 50):
        m = copy.deepcopy(vit); apply_patch(m, prop_attn=True, pp_loc_list=[3, 6, 9], threshold=7e-5); m.r = r
        with torch.no_grad(): a, b = m(x), ours(dict(kind='ppt', r=r, threshold=7e-5))(x)
        print(f'ppt r={r}', (a - b).abs().max().item())
