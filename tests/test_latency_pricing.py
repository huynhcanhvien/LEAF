"""CPU tests for the latency-pricing campaign (run: python tests/test_latency_pricing.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch


def test_lowrank_cpu_matches_reference():
    from leaf.backbones import load
    from leaf.forecast.attention_forecast import AttentionForecast
    from leaf.forecast.cpu_forecast import enable_cpu_lowrank, zscored_lowrank_cpu
    from leaf.reproduction.experiments.low_budget_and_factorial import build
    from leaf.reproduction.experiments.latency_pricing import inventory
    torch.manual_seed(0)
    doc = inventory('deit_s')
    cfg = next(j['config'] for j in doc['jobs'] if j['config']['kind'] == 'pdm' and j['config']['k'])
    vit, spec = load('deit_s', 'cpu')
    m = build(vit, spec, cfg).eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.inference_mode():
        ref = m(x)
        fut = next(f for f in m.modules() if isinstance(f, AttentionForecast))
        xx, s, p = torch.randn(2, 77, spec.dim), torch.rand(2, 77, 1) + 1, torch.rand(2, 77)
        assert (fut.zscored(xx, s, p, 2.) - zscored_lowrank_cpu(fut, xx, s, p, 2.)).abs().max() < 1e-4
        assert enable_cpu_lowrank(m) == len(cfg['k'])
        new = m(x)
    assert torch.equal(ref.argmax(1), new.argmax(1)) and (ref - new).abs().max() < 1e-3


def test_grouped_cv_recovers_linear_law():
    from leaf.reproduction.experiments.latency_pricing import grouped_cv
    rng = np.random.default_rng(0)
    rows = [dict(group=f'g{i}', ms=1. + 2 * a + .3 * k, features=dict(total_gmac=a, original_gmac=a, K=k))
            for i, (a, k) in enumerate(zip(rng.uniform(1, 5, 40), rng.integers(1, 8, 40)))]
    assert grouped_cv(rows, ['original_gmac', 'K'])['mape'] < 1e-6
    assert grouped_cv(rows, ['total_gmac'])['mape'] > 1.


def test_old_sources_unchanged():
    # The campaign recorded the hashes of the research repository's source files at their old paths
    # (results/low_budget_and_factorial/original_sources.json). Those files were ported to leaf/ with new names and
    # imports, so the check cannot hold in this repository; it is kept as a record and skipped.
    print('SKIP old_sources_unchanged: sources were ported from the research repository')


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn(); print('ok', name)
