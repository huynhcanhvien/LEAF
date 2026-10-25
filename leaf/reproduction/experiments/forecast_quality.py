"""Forecast quality (Tab. forecast-quality): how well a token score predicts the true future class-token attention.

For every station block l of the dense model, on 1,024 held-out training images (odd images of calib_bal, disjoint
from the probe images), the Spearman correlation and the top-50% overlap between a score and the target
Phi_j = sum over later blocks of the head-mean class-token attention, for:
- P: the present class-token attention;
- D_G16_r32: the deployed forecast (16 head groups, rank 32, Triton, BF16);
- D_full: the full frozen-residual forecast (singleton groups, full rank, fp32);
- PD_<D>_k<k>: z(log P) + k z(D), k = 1, 2, 3 (k = 1 is the alpha = 1/2 score of the paper).

Image-bootstrap 95% intervals (2,000 resamples). Port of the LEAF-M0 campaign's forecast_quality script and its
evaluation module, computations unchanged; its results are in results/leaf_m0_forecast_quality.

    CUDA_VISIBLE_DEVICES=0 python leaf/reproduction/experiments/forecast_quality.py run --backbone deit_b
    python leaf/reproduction/experiments/forecast_quality.py check --backbone deit_b   # new run vs the paper's
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from leaf.backbones import SPECS                                       # noqa: E402
from leaf.evaluation.data import iterate, load_split                   # noqa: E402
from leaf.evaluation.runner import require_idle_gpu                    # noqa: E402
from leaf.forecast.attention_forecast import AttentionForecast         # noqa: E402
from leaf.forecast.future_attention import folded_rows                 # noqa: E402
from leaf.model.token_ops import cls_attention, zscore                 # noqa: E402
from leaf.planner.configs import ForecastConfig                        # noqa: E402

PAPER = REPO / 'results/leaf_m0_forecast_quality'
BACKBONES = ('deit_s', 'deit_b', 'mae_b')


def quality_scores(current, implemented, full):
    """P/D controls and both proxy variants' kappa mixtures on patch tokens."""
    lp = current.clamp_min(1e-12).log(); zp = zscore(lp, 1)
    out = {'P': current, 'D_G16_r32': implemented, 'D_full': full}
    for name, dd in [('G16_r32', implemented), ('full', full)]:
        for k in (1, 2, 3): out[f'PD_{name}_k{k}'] = zp + k * zscore(dd, 1)
    return {k: v[:, 1:].float().cpu().numpy() for k, v in out.items()}


@torch.no_grad()
def dense_forecasts(vit, images, forecasts):
    """Dense P at every block; forecast x immediately after attention, before MLP."""
    x = vit.norm_pre(vit.patch_drop(vit._pos_embed(vit.patch_embed(images))))
    attention = []; scores = {}
    for l, block in enumerate(vit.blocks):
        out, p = cls_attention(block, block.norm1(x), None)
        x = x + block.drop_path1(block.ls1(out)); attention.append(p)
        if l in forecasts:
            size = x.new_ones(x.shape[0], x.shape[1], 1)
            approx = forecasts[l].zscored(x, size)
            # Full frozen-residual proxy, singleton groups and full rank, in fp32.
            with torch.autocast(x.device.type, enabled=False):
                full = folded_rows(list(vit.blocks[l + 1:]), x.float()).mean(2).sum(1).clamp_min(1e-30).log()
            scores[l] = quality_scores(p, approx, full)
        x = x + block.drop_path2(block.ls2(block.mlp(block.norm2(x))))
    targets = {l: torch.stack(attention[l + 1:]).sum(0)[:, 1:].float().cpu().numpy() for l in scores}
    return scores, targets


def per_image_metrics(score, target):
    """Tie-aware Spearman and deterministic top-half overlap for each image."""
    from scipy.stats import rankdata
    score = np.asarray(score); target = np.asarray(target)
    if score.shape != target.shape or score.ndim != 2: raise ValueError('matched image/token matrices required')
    a = rankdata(score, axis=1); b = rankdata(target, axis=1)
    a -= a.mean(1, keepdims=True); b -= b.mean(1, keepdims=True)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    rho = np.divide((a * b).sum(1), den, out=np.zeros(len(a)), where=den > 0)
    n = score.shape[1] // 2
    if n < 1: raise ValueError('at least two patch tokens required')
    ia = np.argsort(-score, axis=1, kind='stable')[:, :n]; ib = np.argsort(-target, axis=1, kind='stable')[:, :n]
    overlap = np.array([len(np.intersect1d(u, v)) / n for u, v in zip(ia, ib)])
    return dict(spearman=rho, top50_overlap=overlap, constant=(den == 0))


def bootstrap_summary(values, repeats=2000, seed=151):
    """Image bootstrap CI for the image-mean statistic, reproducible across scores."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all(): raise ValueError('finite per-image values required')
    rng = np.random.default_rng(seed); means = np.empty(repeats)
    for start in range(0, repeats, 100):
        n = min(100, repeats - start); means[start:start + n] = values[rng.integers(len(values), size=(n, len(values)))].mean(1)
    return dict(mean=float(values.mean()), ci95=np.quantile(means, [.025, .975]).tolist(), images=len(values))


def default_batch(bb):
    """The batch of the paper's run (its memory profile), so that BF16 results match exactly."""
    p = PAPER / bb / 'profile.json'
    return json.loads(p.read_text())['chosen'] if p.exists() else 32


def run(a):
    from leaf.api import load_backbone
    out = Path(a.out) / a.backbone; out.mkdir(parents=True, exist_ok=True)
    require_idle_gpu(); torch.set_num_threads(4); torch.manual_seed(151)
    vit, spec = load_backbone(a.backbone)
    d = ForecastConfig(a.groups or REPO / f'schedules/leaf/planning/{a.backbone}/groups.json', G=16, rank=32)
    forecasts = {l: AttentionForecast(list(vit.blocks[l + 1:]), d.at(l), rank=32).cuda() for l in range(1, spec.depth - 1)}
    for f in forecasts.values(): f.use_triton(); f.prepare_inference(torch.bfloat16)
    images, labels = load_split('calib_bal', a.data_cache); indices = np.arange(1, len(labels), 2)[:1024]
    im, lb = images[indices], labels[indices]
    batch = a.batch or default_batch(a.backbone)
    chunks = []
    for start in range(0, len(lb), batch):
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            x, _ = next(iterate(im[start:start + batch], lb[start:start + batch], batch, 'cuda', spec.mean, spec.std))
            scores, targets = dense_forecasts(vit, x, forecasts)
        chunk = {}
        for l, rows in scores.items():
            for name, score in rows.items():
                for metric, values in per_image_metrics(score, targets[l]).items(): chunk[f'{l}/{name}/{metric}'] = values
        chunks.append(chunk)
        print(f'[forecast-quality] {a.backbone}: {min(start + batch, len(lb))}/{len(lb)} images', flush=True)
    result = {}
    for key in chunks[0]:
        values = np.concatenate([c[key] for c in chunks])
        result[key] = dict(bootstrap_summary(values), values=values.tolist())
    (out / 'results.json').write_text(json.dumps(dict(
        backbone=a.backbone, images=len(lb), batch=batch, blocks=result,
        target='sum of actual later head-mean CLS attention', location='after attention residual; before MLP',
        precision='bf16 dense/implemented D; fp32 full proxy'), indent=1) + '\n')
    print(f'[forecast-quality] wrote {out / "results.json"}')


def check(a):
    new = json.loads((Path(a.out) / a.backbone / 'results.json').read_text())['blocks']
    ref = json.loads((PAPER / a.backbone / 'results.json').read_text())['blocks']
    keys = [k for k in ref if k.endswith('/spearman')]
    dev = max(abs(new[k]['mean'] - ref[k]['mean']) for k in keys)
    same_table = all(f"{new[k]['mean']:.2f}" == f"{ref[k]['mean']:.2f}" for k in keys)
    print(f'[forecast-quality] {a.backbone}: {len(keys)} Spearman means, max |new - paper| = {dev:.2e}; '
          f'table digits identical: {same_table}')
    if not same_table: raise SystemExit(1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=('run', 'check'))
    ap.add_argument('--backbone', required=True, choices=BACKBONES)
    ap.add_argument('--out', default=str(REPO / 'results/forecast_quality'), help='new runs (the paper\'s run is kept apart)')
    ap.add_argument('--groups', help='groups.json (default: the released one)')
    ap.add_argument('--batch', type=int, help="image batch (default: the batch of the paper's run)")
    ap.add_argument('--data-cache', default='data/cache')
    a = ap.parse_args()
    {'run': run, 'check': check}[a.stage](a)
