"""CLIP backbone: the zero-shot head is a scaled cosine to the text embeddings and matches open_clip; keeping all
tokens reproduces the dense backbone; DiffRate (transferred) and the baselines run on it; retrieval ranks and
recall@k are right on a toy matrix and on cached Flickr images. CPU only (FP32).

Needs the timm/open_clip weights; the cached parts (text embeddings, flickr_test) are skipped when missing."""
from common import check, finish
import os

import numpy as np
import torch
import torch.nn.functional as F

from leaf.model.clip_head import ZeroShotHead, with_classifier
from leaf.evaluation.retrieval import ranks, recall, similarities

CACHE = 'data/cache'


def toy_retrieval():
    # 3 images, 2 captions each (image of caption c = c // 2)
    owner = np.array([0, 0, 1, 1, 2, 2])
    sim = torch.tensor([[.9, .1, .8, .2, .0, .0],     # image 0: best own caption 0.9 -> rank 0
                        [.1, .2, .3, .7, .95, .1],    # image 1: best own 0.7, beaten by caption 4 -> rank 1
                        [.5, .4, .3, .2, .1, .05]])   # image 2: best own 0.1, beaten by 4 captions -> rank 4
    i2t, t2i = ranks(sim, owner)
    check('i2t ranks', list(i2t) == [0, 1, 4], str(list(i2t)))
    # caption c: images scoring above its own image in column c (caption 1: own 0.1 < 0.2, 0.4 -> 2)
    check('t2i ranks', list(t2i) == [0, 2, 1, 0, 1, 1], str(list(t2i)))
    r = recall(i2t, t2i)
    check('recall@k', abs(r['i2t_r1'] - 100 / 3) < 1e-9 and r['i2t_r5'] == 100 and abs(r['t2i_r1'] - 200 / 6) < 1e-9
          and abs(r['rsum'] - sum(v for k, v in r.items() if k != 'rsum')) < 1e-9, str(r))


def head_math():
    torch.manual_seed(0)
    proj = torch.nn.Linear(16, 8, bias=False); text = F.normalize(torch.randn(5, 8), dim=-1)
    head = ZeroShotHead(proj, text); x = torch.randn(3, 16)
    ref = 100 * F.normalize(proj(x), dim=-1) @ text.t()
    check('ZeroShotHead = 100 x cosine', torch.allclose(head(x), ref, atol=1e-5))
    other = F.normalize(torch.randn(7, 8), dim=-1)
    with with_classifier(head, other):
        swapped = head(x)
    check('with_classifier swaps and restores', swapped.shape == (3, 7) and torch.equal(head.classifier, text))


def backbone():
    import timm
    import open_clip
    from leaf.backbones import SPECS
    from leaf.evaluation.runner import build
    from leaf.api import build_model
    from leaf.planner.configs import diffrate_config
    spec = SPECS['clip_b16']
    check('clip_b16 registry', (spec.head, spec.n_tokens, spec.depth, spec.mean[0]) == ('clip_zeroshot', 197, 12, 0.48145466))
    vit = timm.create_model(spec.timm_id, pretrained=True).eval().requires_grad_(False)
    oc, _, _ = open_clip.create_model_and_transforms(*spec.clip_text); oc = oc.eval()
    torch.manual_seed(0); img = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        cos = F.cosine_similarity(vit(img), oc.encode_image(img)).min().item()
    check('timm image tower = open_clip encode_image', cos > .9999, f'{cos:.6f}')
    path = os.path.join(CACHE, 'clip_text_clip_b16_imagenet.npy')
    text = torch.from_numpy(np.load(path)) if os.path.exists(path) else F.normalize(torch.randn(1000, 512), dim=-1)
    vit.head = ZeroShotHead(vit.head, text)
    T, D = spec.n_tokens, spec.depth
    keep_all = dict(kind='pd', prune_keep=[T] * D, merge_keep=[T] * D, k={}, groups={}, rank=None, triton=False, fuse_merge=False)
    with torch.no_grad():
        dense = vit(img); kept = build(vit, spec, keep_all)(img)
    check('keep-all ReducedViT = dense CLIP logits', torch.allclose(kept, dense, atol=1e-3), f'{(kept - dense).abs().max():.1e}')
    dr = dict(diffrate_config(spec, '10.0'), triton=False, fuse_merge=False)
    cfgs = dict(diffrate_transfer=dr, tome=dict(kind='tome', r=13), tome_m3=dict(kind='tome_m3', r=13),
                pitome=dict(kind='pitome', ratio=.9), evit=dict(kind='evit', keep_rate=.7), ppt=dict(kind='ppt', r=30),
                tofu=dict(kind='tofu', r=13), rapid=dict(kind='rapid', r=13), onemore=dict(kind='onemore', n=100))
    for name, cfg in cfgs.items():
        try:
            with torch.no_grad(): out = build_model(vit, spec, cfg)(img)
            check(f'{name} runs on CLIP', out.shape == (2, text.shape[0]) and bool(torch.isfinite(out).all()))
        except Exception as e:
            check(f'{name} runs on CLIP', False, repr(e)[:200])
    if not os.path.exists(path):
        print('skip: zero-shot top-1 and retrieval (run `leaf data clip-text --backbone clip_b16`)'); return
    from leaf.evaluation.data import load_split
    images, labels = load_split('val', CACHE)
    from leaf.evaluation.data import iterate
    correct = []
    with torch.no_grad():
        for x, y in iterate(images[:256], labels[:256], 64, 'cpu', spec.mean, spec.std):
            correct.append(vit(x).argmax(-1).eq(y))
    acc = 100 * torch.cat(correct).float().mean().item()
    check('dense zero-shot top-1 on 256 val images in [55, 80]', 55 <= acc <= 80, f'{acc:.1f}%')
    cap = os.path.join(CACHE, 'clip_text_clip_b16_flickr_test.npy')
    if not os.path.exists(cap) or not os.path.exists(os.path.join(CACHE, 'flickr_test_224_u8.npy')):
        print('skip: Flickr retrieval (run `leaf data prepare --split flickr_test` and `leaf data clip-text --task flickr_test`)'); return
    from leaf.evaluation.retrieval import captions
    imgs, _ = load_split('flickr_test', CACHE); _, owner = captions('flickr_test', CACHE); emb = torch.from_numpy(np.load(cap))
    n = 100; keep = owner < n
    sim = similarities(vit, vit.head, imgs[:n], emb[torch.from_numpy(keep)], 50, 'cpu', spec.mean, spec.std, autocast=False)
    i2t, t2i = ranks(sim, owner[keep]); r = recall(i2t, t2i)
    check('dense Flickr i2t R@1 on 100 images (500 captions) >= 85', r['i2t_r1'] >= 85, f"{r['i2t_r1']:.1f} / t2i {r['t2i_r1']:.1f}")


def campaign():
    """clip_comparison on a synthetic workspace: sealed plan, latency pairing, Holm, table and figure (no inference)."""
    import json, tempfile
    from pathlib import Path
    import leaf.reproduction.experiments.clip_comparison as cc
    from leaf.backbones import SPECS
    from leaf.planner.configs import diffrate_config
    spec = SPECS['clip_b16']
    with tempfile.TemporaryDirectory() as tmp:
        cc.ROOT = Path(tmp); ws = cc.workspace('clip_b16'); ws.mkdir(parents=True)
        ours = dict(diffrate_config(spec, '11.5'), triton=False)          # any pd config stands in for a planned schedule
        sched = {'latency/b1': dict(config=ours, budget=dict(kind='latency', value=20., id='b1'), planner={})}
        (ws / 'schedules.json').write_text(json.dumps(dict(schedules=sched)))
        (ws / 'schedules_gmac.json').write_text(json.dumps(dict(schedules={'gmac/g1': dict(sched['latency/b1'], budget=dict(kind='gmac', value=11., id='g1'))})))
        plan = cc.plan('clip_b16')
        names = set(plan['jobs'])
        check('plan: method, transferred DiffRate, every family', {'leaf/latency/b1', 'leaf/gmac/g1', 'ref/dense', 'ref/diffrate_transfer/10.0'} <= names
              and {n.split('/')[2] for n in names if n.startswith('ref/baseline/')} == set(cc.GRIDS), f'{len(names)} arms')
        check('plan sealed (immutable)', cc.plan('clip_b16')['sha'] == plan['sha'])
        rng = np.random.default_rng(0); n_val, n_img, n_cap = 400, 50, 250
        truth = {}
        for t in cc.tasks():
            sha = t['config']['kind'] == 'pd' and t['names'][0].startswith('leaf/')
            good = .8 if sha else .7
            truth[t['id']] = good
            for split, arr in (('val', dict(correct=rng.random(n_val) < good)), ('v2', dict(correct=rng.random(n_val) < good - .1)),
                               ('coco_test', dict(i2t=(rng.random(n_img) > good) * 3, t2i=(rng.random(n_cap) > good) * 3))):
                d = cc.ROOT / 'clip_b16' / split / 'runs'; d.mkdir(parents=True, exist_ok=True)
                np.savez(d / f"{t['id'].split('-', 1)[1]}.npz", **arr)
            for rnd in range(3):     # every arm 20 ms +- noise: all within tau of each other
                cc.save(cc.ROOT / 'clip_b16' / 'latency' / f"{t['id']}#{rnd}.json",
                        dict(id=t['id'], measurements={'32': dict(ms_mean=20 + .01 * rnd)}))
        rep = cc.analyse('clip_b16')
        fams = {r['family'] for r in rep['latency_pairs']}
        check('report: one latency pair per family', len(rep['latency_pairs']) == len(fams) == len(cc.GRIDS) + 1, str(sorted(fams)))
        check('report: retrieval and classification splits', set(rep['splits']) == {'val', 'v2', 'coco_test_i2t', 'coco_test_t2i'}, str(rep['splits']))
        check('report: Holm on every test', all('holm' in r[s] for r in rep['latency_pairs'] for s in rep['splits']))
        check('report: equal-GMAC pair with transferred DiffRate', len(rep['gmac_pairs']) == 1 and rep['gmac_pairs'][0]['family'] == cc.DR_FAMILY)
        check('report: LEAF ahead on synthetic data', all(r['val']['delta'] > 0 for r in rep['latency_pairs']))
        tex = cc.table(rep); cc.figure(rep, Path(tmp) / 'f.pdf')
        check('report: table and figure', tex.count('\\\\') >= len(fams) and (Path(tmp) / 'f.pdf').stat().st_size > 0)


if __name__ == '__main__':
    torch.set_num_threads(min(16, os.cpu_count() or 1))
    toy_retrieval()
    head_math()
    campaign()
    backbone()
    finish()
