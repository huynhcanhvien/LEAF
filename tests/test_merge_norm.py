"""Norm kernel tests. CPU by default; CUDA suite invoked only by the campaign scheduler."""
from common import check, finish
import torch
from types import SimpleNamespace
from timm.models.vision_transformer import VisionTransformer

from leaf.evaluation.runner import build, per_image_kl
from leaf.model.merge_variants import compress_variant
from leaf.model.mlerp_vit import compress_norm


def cpu_tests():
    torch.manual_seed(703); torch.set_num_threads(2)
    x = torch.randn(2, 17, 24); size = torch.rand(2, 17, 1) + .5; scores = torch.randn(2, 17)
    for mode in ('M3', 'N4'):
        a = compress_variant(x, size, scores, 15, 8, mode)
        b = compress_norm(x, size, scores, 15, 8, mode)
        check(mode + ' CPU fallback bitwise', all(torch.equal(v, w) for v, w in zip(a, b)))
    vit = VisionTransformer(img_size=16, patch_size=4, embed_dim=24, depth=3, num_heads=3, num_classes=7).eval().requires_grad_(False)
    cfg = dict(kind='pd', prune_keep=[17, 14, 8], merge_keep=[17, 10, 6], triton=False)
    img = torch.randn(2, 3, 16, 16); spec = SimpleNamespace(pool='token')
    with torch.no_grad():
        a = build(vit, spec, cfg)(img)
        b = build(vit, spec, dict(cfg, kind='pdm', merge=dict(mode='M0', kernel='triton_norm')))(img)
        check('optimized wrapper M0 bitwise', torch.equal(a, b) and torch.equal(per_image_kl(a, a), per_image_kl(b, a)))
        for mode in ('M3', 'N4'):
            ref = build(vit, spec, dict(cfg, kind='pdm', merge=dict(mode=mode)))(img)
            out = build(vit, spec, dict(cfg, kind='pdm', merge=dict(mode=mode, kernel='triton_norm')))(img)
            check(mode + ' CPU wrapper bitwise reference', torch.equal(ref, out))


@torch.no_grad()
def gpu_tests():
    """Pure function returning evidence; caller owns GPU isolation and profiling."""
    torch.manual_seed(703)
    rows = []
    for c in (64, 192, 384, 768, 1024):
        for p, m in ((17, 16), (65, 33), (197, 97), (197, 3)):
            x = torch.randn(2, p, c, device='cuda')
            size = torch.rand(2, p, 1, device='cuda') * 5 + .5
            scores = torch.randn(2, p, device='cuda')
            for mode in ('M3', 'N4'):
                ref = compress_variant(x, size, scores, p, m, mode)
                a = compress_norm(x, size, scores, p, m, mode)
                b = compress_norm(x, size, scores, p, m, mode)
                error = max((a[i] - ref[i]).abs().max().item() for i in (0, 1))
                repeat = torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
                assert error <= 1e-5, (mode, c, p, m, error)
                assert repeat, (mode, 'repeat')
                # Singleton destinations including CLS must be exact.
                untouched = torch.ones_like(a[1], dtype=torch.bool)
                untouched.scatter_(1, a[3][..., None], False)
                ordered = x.gather(1, a[2][..., None].expand(-1, -1, c))[:, :m]
                assert torch.equal(a[0][untouched.expand_as(a[0])], ordered[untouched.expand_as(ordered)])
                rows.append(dict(mode=mode, channels=c, prune=p, merge=m, max_abs_error=error, repeat=repeat))
    toy = torch.tensor([[[0., 0.], [1., -1.], [-1., 1.]]], device='cuda')
    for mode in ('M3', 'N4'):
        for z in (toy, torch.zeros_like(toy)):
            s = torch.ones(1, 3, 1, device='cuda'); sc = torch.tensor([[0., 2., 1.]], device='cuda')
            a = compress_norm(z, s, sc, 3, 2, mode)
            ref = compress_variant(z, s, sc, 3, 2, mode)
            assert torch.allclose(a[0], ref[0], atol=1e-5, rtol=0) and torch.allclose(a[1], ref[1], atol=1e-5, rtol=0), (mode, 'cancellation')
    return rows


if __name__ == '__main__':
    cpu_tests(); finish()
