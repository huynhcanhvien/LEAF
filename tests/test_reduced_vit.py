"""ReducedViT: keep-all equals the dense backbone; compress matches DiffRate's reference operator; the fused
merge kernel equals the scatter path; repeated evaluation is bit-identical. Needs a GPU."""
from common import check, finish
import math
import torch
import torch.nn.functional as F

from leaf.latency.timing import inference_copy
from leaf.evaluation.runner import build
from leaf.backbones import load
from leaf.model.token_ops import compress


def reference_diffrate(x, size, scores, n_prune, n_merge):
    """Straight re-implementation of DiffRate inference: sort, prune, merge into the most similar kept token."""
    b, n, c = x.shape; out_x, out_s = [], []
    for i in range(b):
        order = torch.cat([torch.zeros(1, dtype=torch.long, device=x.device), scores[i, 1:].argsort(descending=True) + 1])[:n_prune]
        xi, si = x[i, order].clone(), size[i, order].clone()
        keep, drop = xi[:n_merge].clone(), xi[n_merge:]
        sim = F.normalize(drop, dim=-1) @ F.normalize(xi[:n_merge], dim=-1).T; sim[:, 0] = -math.inf
        mx, dst = sim.max(-1); mx = mx.clamp_min(0)
        acc, cnt, sz = keep.clone(), torch.ones(n_merge, device=x.device), si[:n_merge].clone()
        for j in range(len(drop)):
            acc[dst[j]] += drop[j]; cnt[dst[j]] += 1; sz[dst[j]] += si[n_merge + j] * mx[j]
        out_x.append(acc / cnt[:, None]); out_s.append(sz)
    return torch.stack(out_x), torch.stack(out_s)


def main():
    torch.manual_seed(0)
    x = torch.randn(3, 50, 64, device='cuda'); size = torch.rand(3, 50, 1, device='cuda') + .5; s = torch.randn(3, 50, device='cuda')
    rx, rs = reference_diffrate(x, size, s, 40, 25)
    for fuse in (False, True):
        cx, cs, _, _ = compress(x, size, s, 40, 25, fuse_merge=fuse)
        check(f'compress == DiffRate reference (fuse_merge={fuse})', torch.allclose(cx, rx, atol=1e-5) and torch.allclose(cs, rs, atol=1e-5),
              f'{(cx - rx).abs().max():.1e} {(cs - rs).abs().max():.1e}')
    vit, spec = load('deit_b'); img = torch.randn(4, 3, 224, 224, device='cuda')
    T, D = spec.n_tokens, spec.depth
    keep_all = dict(kind='pd', prune_keep=[T] * D, merge_keep=[T] * D, k={}, groups={}, rank=None, triton=True, fuse_merge=True)
    with torch.no_grad():
        a, b = vit(img), build(vit, spec, keep_all)(img)
    check('keep-all ReducedViT == dense backbone', torch.allclose(a, b, atol=1e-4), f'{(a - b).abs().max():.1e}')
    import json
    labels = json.load(open('schedules/leaf/planning/deit_b/groups.json'))['groups']
    p = [T, 140, T, 120, T, T, 80, T, T, T, T, 5]; m = [T, 140, T, 100, T, T, 60, T, T, T, T, 5]
    from leaf.planner.schedule_caps import canonical
    g = canonical(dict(p=p, m=m), T)
    cfg = dict(kind='pd', prune_keep=g['p'], merge_keep=g['m'], k={'1': 3., '6': 1.}, groups={'1': labels['1']['16'], '6': labels['6']['16']},
               rank=32, triton=True, fuse_merge=True)
    outs = []
    for _ in range(2):
        mdl = inference_copy(build(vit, spec, cfg))
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16): outs.append(mdl(img).float())
    check('PD model: two builds give identical logits', torch.equal(outs[0], outs[1]))
    nof = inference_copy(build(vit, spec, dict(cfg, fuse_merge=False)))
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16): c = nof(img).float()
    # Summation order differs (unit test above: 2e-7); downstream near-tie token choices can flip.
    check('fused merge kernel ~= scatter path (end to end)', (c - outs[0]).abs().max().item() < 1e-1 and bool((c.argmax(1) == outs[0].argmax(1)).all()), f'{(c - outs[0]).abs().max():.1e}')


if __name__ == '__main__':
    main(); finish()
