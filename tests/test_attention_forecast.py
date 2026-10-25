"""Closed-form grouped D: matches the frozen-residual reference; Triton kernels and low rank match the
PyTorch path; construction is deterministic. Needs a GPU.   python tests/test_attention_forecast.py"""
from common import check, finish
import torch

from leaf.backbones import load
from leaf.forecast.future_attention import folded_rows, grouped_log_rows
from leaf.forecast.attention_forecast import AttentionForecast
from leaf.model.token_ops import zscore


def rank_rho(a, b):
    rank = lambda v: v[:, 1:].argsort(1).argsort(1).float()
    ra, rb = rank(a), rank(b); ra, rb = ra - ra.mean(1, keepdim=True), rb - rb.mean(1, keepdim=True)
    return ((ra * rb).sum(1) / (ra.norm(dim=1) * rb.norm(dim=1))).min().item()


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    for name in ('deit_b', 'deit_t'):
        vit, spec = load(name); C, H = spec.dim, spec.heads
        for l, n in ((1, 197), (6, 117), (10, 31)):
            blocks = list(vit.blocks[l + 1:]); P = len(blocks) * H
            torch.manual_seed(0)
            x = torch.randn(4, n, C, device='cuda') * 2 + torch.randn(4, n, 1, device='cuda'); x[..., :4] *= 30
            size = torch.randint(1, 6, (4, n, 1), device='cuda').float()
            for gname, labels in (('G4', [i % 4 for i in range(P)]), ('G1', [0] * P), ('singletons', list(range(P)))):
                m = AttentionForecast(blocks, labels).cuda()
                with torch.no_grad():
                    ref = grouped_log_rows(folded_rows(blocks, x, size.log()), labels); out = m(x, size)
                err = (zscore(ref, 1) - zscore(out, 1)).abs().max().item()
                check(f'{name} block {l} {gname}: closed form == folded rows', err < 2e-3, f'{err:.1e}')
            labels = [i % 16 for i in range(P)]
            ref_m = AttentionForecast(blocks, labels).cuda(); ref_m.prepare_inference(torch.bfloat16)
            tri = AttentionForecast(blocks, labels).cuda(); tri.use_triton(); tri.prepare_inference(torch.bfloat16)
            with torch.no_grad():
                a, b = zscore(ref_m(x, size), 1), tri.zscored(x, size)
            check(f'{name} block {l}: triton == bf16 reference', (a - b).abs().max().item() < 5e-2 and rank_rho(a, b) > .999
                  and bool((b[:, 0] == 0).all()), f'{(a - b).abs().max().item():.1e}')
            for r in (16, 32):
                lo = AttentionForecast(blocks, labels, rank=r).cuda(); lo.prepare_inference(torch.bfloat16)
                lt = AttentionForecast(blocks, labels, rank=r).cuda(); lt.use_triton(); lt.prepare_inference(torch.bfloat16)
                with torch.no_grad():
                    err = (zscore(lo(x, size), 1) - lt.zscored(x, size)).abs().max().item()
                check(f'{name} block {l} rank {r}: triton factors == reconstruction', err < 5e-2, f'{err:.1e}')
            m1, m2 = AttentionForecast(blocks, labels, rank=32).cuda(), AttentionForecast(blocks, labels, rank=32).cuda()
            check(f'{name} block {l}: deterministic construction', torch.equal(m1.A, m2.A) and torch.equal(m1.Bf, m2.Bf))
            full = AttentionForecast(blocks, labels).cuda(); fr = AttentionForecast(blocks, labels, rank=C).cuda()
            with torch.no_grad():
                err = (zscore(full(x, size), 1) - zscore(fr(x, size), 1)).abs().max().item()
            check(f'{name} block {l}: full rank == dense U', err < 1e-3, f'{err:.1e}')
        del vit; torch.cuda.empty_cache()


if __name__ == '__main__':
    main(); finish()
