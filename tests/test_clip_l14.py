"""CLIP ViT-L/14 (patch 14, 257 tokens, 24 blocks): the shape-dependent parts of the planner follow the backbone's
patch and token count, while every 197-token backbone keeps the paper's numbers; the transferred MAE-L DiffRate
rates keep the same fraction of patch tokens; the campaign helpers use the backbone's depth; the timm tower matches
open_clip, keeping all tokens reproduces the dense model and every baseline runs. CPU only (FP32).

Needs the timm/open_clip ViT-L/14 weights for `backbone()`."""
from common import check, finish
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from leaf.backbones import SPECS

OLD = ('deit_t', 'deit_s', 'deit_b', 'augreg_s', 'augreg_b', 'mae_b', 'mae_l', 'clip_b16')


def registry():
    spec = SPECS['clip_l14']
    check('clip_l14 registry', (spec.patch, spec.n_tokens, spec.depth, spec.dim, spec.head, spec.diffrate_key) ==
          (14, 257, 24, 1024, 'clip_zeroshot', 'ViT-L-MAE'))
    check('existing backbones: patch 16, 197 tokens', all((SPECS[b].patch, SPECS[b].n_tokens) == (16, 197) for b in OLD))


def probe_grid():
    from leaf.cli import DEFAULT_DELTAS, default_deltas
    check('probe deltas unchanged on 197 tokens', all(default_deltas(SPECS[b]) == list(DEFAULT_DELTAS) for b in OLD))
    d = default_deltas(SPECS['clip_l14'])
    check('probe deltas on 257 tokens: same fractions, last leaves 3 tokens',
          d == [10, 31, 63, 94, 125, 157, 188, 219, 254], str(d))


def gmac():
    from leaf.planner.gmac import dense_cost, config_cost
    from leaf.planner.schedule_caps import full_cost
    from leaf.latency.salt import constants
    for bb in ('clip_b16', 'clip_l14'):
        spec = SPECS[bb]; n, d, p = spec.n_tokens, spec.dim, spec.patch
        ref = ((n - 1) * 3 * p * p * d + 1000 * d + spec.depth * (12 * n * d * d + 2 * n * n * d)) / 1e9
        check(f'{bb}: dense GMAC uses patch {p}', abs(dense_cost(spec) - ref) < 1e-9, f'{dense_cost(spec):.3f}')
        keep = [n] * spec.depth
        cfg = dict(kind='pd', prune_keep=keep, merge_keep=keep, k={}, groups={}, rank=None)
        row = full_cost(keep, keep, None, n, d, 16, 32, p)
        # keep-all = dense + the CLS-row term d*n of every block, which the dense count leaves out
        check(f'{bb}: keep-all config cost = dense + CLS rows', abs(config_cost(cfg, spec) - dense_cost(spec) - spec.depth * n * d / 1e9) < 1e-9)
        check(f'{bb}: patch adjustment', abs(row['patch_adjustment_gmac'] - (n - 1) * (3 * p * p * d - d * d) / 1e9) < 1e-12)
        check(f'{bb}: SALT total_gmac constant', abs(constants(spec)['total_gmac'] - ((n - 1) * 3 * p * p * d + 1000 * d) / 1e9) < 1e-12)
    check('clip_l14 dense ~81 GMAC (published 81.1 GFLOPs at 224)', 80 < dense_cost(SPECS['clip_l14']) < 82,
          f"{dense_cost(SPECS['clip_l14']):.2f}")


def diffrate():
    from leaf.planner.configs import diffrate_schedules, diffrate_config, DIFFRATE_RATES
    from leaf.planner.schedule_caps import canonical
    rates = json.loads(DIFFRATE_RATES.read_text())['rates']['ViT-L-MAE']
    src = {b: (json.loads(v['prune_kept_num']), json.loads(v['merge_kept_num'])) for b, v in rates.items()}
    check('MAE-L rates unchanged on mae_l', diffrate_schedules(SPECS['mae_l']) == src)
    out = diffrate_schedules(SPECS['clip_l14'])
    ok = set(out) == set(src)
    for b, (p, m) in out.items():
        for new, old in zip(p + m, src[b][0] + src[b][1]):
            ok &= abs((new - 1) / 256 - (old - 1) / 196) <= .5 / 256 + 1e-12      # same patch fraction, rounded
    check('clip_l14 DiffRate: MAE-L patch fractions on 256 patches', ok)
    check('clip_l14 DiffRate: full count 197 -> 257', all((new == 257) == (old == 197) for b in out
                                                          for new, old in zip(out[b][0] + out[b][1], src[b][0] + src[b][1])))
    spec = SPECS['clip_l14']; cfg = diffrate_config(spec, '31.0')
    check('clip_l14 DiffRate config canonical', canonical(dict(p=cfg['prune_keep'], m=cfg['merge_keep']), 257, 24)['p'] == cfg['prune_keep']
          and len(cfg['prune_keep']) == 24 and min(cfg['merge_keep']) >= 2)


def campaign_helpers():
    import leaf.reproduction.experiments.clip_comparison as cc
    from baselines.methods import parse_r
    l14, b16 = SPECS['clip_l14'], SPECS['clip_b16']
    c = cc.candidates('tome', l14)
    check('r-family candidates: 24-block schedules', all(len(parse_r(24, (v['r'][0], v['r'][1]) if isinstance(v['r'], list) else v['r'])) == 24
                                                        and min(parse_r(24, (v['r'][0], v['r'][1]) if isinstance(v['r'], list) else v['r'])) >= 0
                                                        for v in c.values()))
    check('onemore candidates up to 257', max(v['n'] for v in cc.candidates('onemore', l14).values()) == 257)
    check('ppt candidates below (n-1)/2', max(v['r'] for v in cc.candidates('ppt', l14).values()) == 128
          and max(v['r'] for v in cc.candidates('ppt', b16).values()) == 98)
    s = cc.scalar(dict(kind='tome', r=8), l14)
    check('tome scalar on 24 blocks', s == sum(257 - 8 * (i + 1) for i in range(24)), str(s))
    check('DiffRate family names', cc.DR_FAMILIES == {'clip_b16': 'DiffRate (DeiT-B rates)', 'clip_l14': 'DiffRate (MAE-L rates)'}
          and cc.DR_FAMILY == 'DiffRate (DeiT-B rates)')


def campaign_plan():
    """clip_comparison.plan and analyse on a synthetic clip_l14 workspace (no inference)."""
    import tempfile
    from pathlib import Path
    import leaf.reproduction.experiments.clip_comparison as cc
    from leaf.planner.configs import diffrate_config
    spec = SPECS['clip_l14']; root = cc.ROOT
    try:
        with tempfile.TemporaryDirectory() as tmp:
            cc.ROOT = Path(tmp); ws = cc.workspace('clip_l14'); ws.mkdir(parents=True)
            ours = dict(diffrate_config(spec, '38.5'), triton=False)
            sched = {'latency/b1': dict(config=ours, budget=dict(kind='latency', value=60., id='b1'), planner={})}
            (ws / 'schedules.json').write_text(json.dumps(dict(schedules=sched)))
            (ws / 'schedules_gmac.json').write_text(json.dumps(dict(schedules={'gmac/g1': dict(sched['latency/b1'], budget=dict(kind='gmac', value=40., id='g1'))})))
            plan = cc.plan('clip_l14')
            check('plan: MAE-L DiffRate transferred at five points', sum(n.startswith('ref/diffrate_transfer/') for n in plan['jobs']) == 5
                  and plan['rules']['diffrate'] == 'DiffRate (MAE-L rates)')
            rng = np.random.default_rng(0)
            for t in cc.tasks(('clip_l14',)):
                good = .8 if t['names'][0].startswith('leaf/') else .7
                for split, arr in (('val', dict(correct=rng.random(300) < good)), ('v2', dict(correct=rng.random(300) < good))):
                    d = cc.ROOT / 'clip_l14' / split / 'runs'; d.mkdir(parents=True, exist_ok=True)
                    np.savez(d / f"{t['id'].split('-', 1)[1]}.npz", **arr)
                for rnd in range(3):
                    cc.save(cc.ROOT / 'clip_l14' / 'latency' / f"{t['id']}#{rnd}.json", dict(id=t['id'], measurements={'32': dict(ms_mean=60.)}))
            rep = cc.analyse('clip_l14')
            check('report: equal-GMAC pair with the MAE-L DiffRate family',
                  len(rep['gmac_pairs']) == 1 and rep['gmac_pairs'][0]['family'] == 'DiffRate (MAE-L rates)')
    finally:
        cc.ROOT = root


def backbone():
    import timm
    import open_clip
    from leaf.evaluation.runner import build
    from leaf.api import build_model
    from leaf.model.clip_head import ZeroShotHead
    from leaf.planner.configs import diffrate_config
    spec = SPECS['clip_l14']
    vit = timm.create_model(spec.timm_id, pretrained=True).eval().requires_grad_(False)
    check('timm tower: patch 14, 257 tokens', (tuple(vit.patch_embed.patch_size), vit.patch_embed.num_patches + vit.num_prefix_tokens,
                                              len(vit.blocks), vit.embed_dim) == ((14, 14), 257, 24, 1024))
    oc, _, _ = open_clip.create_model_and_transforms(*spec.clip_text); oc = oc.eval()
    torch.manual_seed(0); img = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        cos = F.cosine_similarity(vit(img), oc.encode_image(img)).min().item()
    check('timm image tower = open_clip encode_image', cos > .9999, f'{cos:.6f}')
    vit.head = ZeroShotHead(vit.head, F.normalize(torch.randn(1000, 768), dim=-1))
    T, D = spec.n_tokens, spec.depth
    keep_all = dict(kind='pd', prune_keep=[T] * D, merge_keep=[T] * D, k={}, groups={}, rank=None, triton=False, fuse_merge=False)
    with torch.no_grad():
        dense = vit(img); kept = build(vit, spec, keep_all)(img)
    check('keep-all ReducedViT = dense CLIP logits', torch.allclose(kept, dense, atol=1e-3), f'{(kept - dense).abs().max():.1e}')
    cfgs = dict(diffrate_transfer=dict(diffrate_config(spec, '38.5'), triton=False, fuse_merge=False),
                tome=dict(kind='tome', r=8), tome_m3=dict(kind='tome_m3', r=8), pitome=dict(kind='pitome', ratio=.9),
                evit=dict(kind='evit', keep_rate=.7), ppt=dict(kind='ppt', r=40), tofu=dict(kind='tofu', r=8),
                rapid=dict(kind='rapid', r=8), onemore=dict(kind='onemore', n=130), mctf=dict(kind='mctf', r=8))
    for name, cfg in cfgs.items():
        try:
            with torch.no_grad(): out = build_model(vit, spec, cfg)(img)
            check(f'{name} runs on CLIP ViT-L/14', out.shape == (2, 1000) and bool(torch.isfinite(out).all()))
        except Exception as e:
            check(f'{name} runs on CLIP ViT-L/14', False, repr(e)[:200])


if __name__ == '__main__':
    torch.set_num_threads(min(16, os.cpu_count() or 1))
    registry()
    probe_grid()
    gmac()
    diffrate()
    campaign_helpers()
    campaign_plan()
    backbone()
    finish()
