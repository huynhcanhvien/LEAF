"""CPU algebra/integration checks. Never allocates a CUDA tensor or downloads weights."""
from common import check, finish
from types import SimpleNamespace

import torch
from timm.models.vision_transformer import VisionTransformer

from leaf.evaluation.runner import build, config_hash, per_image_kl
from leaf.model.merge_variants import MODES, compress_variant, cls_attention_with_keys
from leaf.model.token_ops import compress, cls_attention
from leaf.model.merge_variant_vit import score_and_forecast


def main():
    torch.set_num_threads(2); torch.manual_seed(41)
    x = torch.randn(2, 12, 8)
    size = torch.rand(2, 12, 1) + .5
    score = torch.arange(12, 0, -1).float().expand(2, -1)

    def run(mode, **kw):
        return compress_variant(x, size, score, 12, 6, mode, **kw)

    old, new = compress(x, size, score, 10, 6), compress_variant(x, size, score, 10, 6)
    check('M0 bit-identical to compress', all(torch.equal(a, b) for a, b in zip(old, new)))
    for mode in MODES:
        kwargs = dict(keys=x, forecast=(torch.randn(2, 8), 1e-6))
        a, b = run(mode, **kwargs), run(mode, **kwargs)
        check(mode + ': repeat bit-identical and finite', all(torch.equal(v, w) for v, w in zip(a, b)) and
              all(torch.isfinite(v).all() for v in a[:2]))
        check(mode + ': CLS unchanged', torch.equal(a[0][:, 0], x[:, 0]) and torch.equal(a[1][:, 0], size[:, 0]))
        singleton = compress_variant(x, size, score, 6, 6, mode, **kwargs)
        check(mode + ': no merge unchanged', torch.equal(singleton[0], x[:, :6]) and torch.equal(singleton[1], size[:, :6]))
    m1 = run('M1')
    check('M1 conserves mass', torch.allclose(m1[1].sum(1), size.sum(1)))
    check('M1 conserves weighted residual', torch.allclose((m1[0] * m1[1]).sum(1), (x * size).sum(1), atol=2e-6))
    n1 = run('N1', tau=0.)
    check('N1 tau=0 equals M1', torch.equal(n1[0], m1[0]) and torch.equal(n1[1], m1[1]))
    m2 = run('M2', keys=x)
    check('M2 identical keys/residual -> M0 assignment', torch.equal(m2[3], run('M0')[3]))
    m3 = run('M3')
    for batch in range(2):
        for dst in range(6):
            members = torch.cat((x[batch, dst:dst + 1], x[batch, 6:][m3[3][batch] == dst]))
            check(f'M3 max norm {batch}/{dst}', torch.allclose(m3[0][batch, dst].norm(), members.norm(dim=-1).max(), atol=2e-6))
    check('N3 destinations untouched', torch.equal(run('N3')[0], x[:, :6]))
    check('N2 without D falls back to M1', torch.equal(run('N2')[0], m1[0]) and torch.equal(run('N2')[1], m1[1]))
    same = torch.randn(2, 1, 8).expand(-1, 12, -1).clone()
    n2 = compress_variant(same, size, score, 12, 6, 'N2', forecast=(torch.randn(2, 8), 1e-6))
    ref = compress_variant(same, size, score, 12, 6, 'M1')
    check('N2 identical members -> M1', torch.allclose(n2[0], ref[0], atol=2e-6) and torch.allclose(n2[1], ref[1], atol=2e-5))
    # Nonzero common direction, variable magnitudes and channel means.
    aligned = torch.arange(1, 13).view(1, 12, 1) * torch.randn(2, 1, 8)
    a = compress_variant(aligned, size, score, 12, 6, 'N4')
    b = compress_variant(aligned, size, score, 12, 6, 'M1')
    check('N4 collinear vectors -> M1', torch.allclose(a[0], b[0], atol=3e-6))
    n4 = run('N4')
    check('N4 preserves per-token channel mean', torch.allclose(n4[0].mean(-1), m1[0].mean(-1), atol=1e-6))
    # Orthogonal destinations: source identical to the first, zero cosine to second.
    toy = torch.tensor([[[0., 0.], [1., 0.], [0., 1.], [1., 0.]]])
    ts = torch.tensor([[[1.], [2.], [3.], [4.]]]); sc = torch.tensor([[0., 3., 2., 1.]])
    a = compress_variant(toy, ts, sc, 4, 3, 'N5')
    b = compress_variant(toy, ts, sc, 4, 3, 'M1')
    check('N5 cosine 1/0 -> M1', torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]))
    antipodal = torch.tensor([[[0., 0.], [1., -1.], [-1., 1.]]])
    for mode in ('M3', 'N4'):
        cancelled = compress_variant(antipodal, torch.ones(1, 3, 1), torch.tensor([[0., 2., 1.]]), 3, 2, mode)
        check(mode + ': cancellation retains member norm', torch.allclose(cancelled[0][0, 1].norm(), antipodal[0, 1].norm()))
    soft = run('N5')
    check('N5 conserves mass and residual', torch.allclose(soft[1].sum(1), size.sum(1)) and
          torch.allclose((soft[0] * soft[1]).sum(1), (x * size).sum(1), atol=3e-6))
    hard = run('N6', candidate_factor=1)
    check('N6 unexpanded pool -> M0 order', torch.equal(hard[2], run('M0')[2]))
    # Minimum destination count and zero vectors exercise soft fallback and norm guards.
    for mode in MODES[1:]:
        a = compress_variant(torch.zeros_like(x), size, score, 12, 2, mode, keys=x)
        check(mode + ': degenerate finite', torch.isfinite(a[0]).all() and torch.isfinite(a[1]).all())

    vit = VisionTransformer(img_size=16, patch_size=4, embed_dim=24, depth=3,
                            num_heads=3, num_classes=7).eval().requires_grad_(False)
    spec = SimpleNamespace(pool='token')
    cfg = dict(kind='pda', base='pd', prune_keep=[14, 8, 6], merge_keep=[10, 7, 4],
               alpha={'0': .5}, groups={'0': [0, 0, 0, 1, 1, 1]}, rank=4, triton=False)
    img = torch.randn(2, 3, 16, 16)
    pd = build(vit, spec, cfg)
    original_key = config_hash(cfg)
    with torch.no_grad():
        logits = pd(img)
        m0 = build(vit, spec, dict(cfg, base='pdm', merge={'mode': 'M0'}))(img)
        teacher = vit(img)
        check('PDM M0 logits/predictions/KL bit-identical', torch.equal(logits, m0) and
              torch.equal(per_image_kl(logits, teacher), per_image_kl(m0, teacher)))
        for mode in MODES[1:]:
            model = build(vit, spec, dict(cfg, base='pdm', merge={'mode': mode}))
            a, b = model(img), model(img)
            check(mode + ': CPU wrapper repeat/finite', torch.equal(a, b) and torch.isfinite(a).all())
        x0 = pd.embed(img); sz = torch.ones(2, 17, 1)
        a = cls_attention(vit.blocks[0], vit.blocks[0].norm1(x0), sz)
        b = cls_attention_with_keys(vit.blocks[0], vit.blocks[0].norm1(x0), sz)
        check('attention with keys matches original bitwise', torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]))
        score0 = pd.score(0, x0, sz, a[1])
        score1, _ = score_and_forecast(pd.future['0'], x0, sz, a[1], .5)
        check('N2 reuses D vectors with identical score', torch.equal(score0, score1))
    check('config/cache key unchanged', original_key == config_hash(cfg) and 'merge' not in cfg)


if __name__ == '__main__':
    main(); finish()
