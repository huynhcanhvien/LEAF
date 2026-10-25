"""How much the forecast approximations (head grouping G=16, rank r=32) change D at the stations of the LEAF schedules.

python leaf/reproduction/experiments/forecast_approximation.py --gpu 0

For every station block with a forecast in schedules/kappa_form/main_comparison/<bb>.json:
  energy  share of the squared singular values of each group matrix U_g kept by rank 32 (CPU, from the weights);
  r32     per-image Spearman between D(G=16, r=32) and D(G=16, full rank) over the patch tokens;
  approx  per-image Spearman between D(G=16, r=32) and the exact frozen-residual forecast D* (one group per head, full rank),
on the dense trajectory (residual after the attention of the station block) of 256 calibration images.
Writes results/forecast_approximation/report.json.
"""
import argparse
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
import numpy as np

OUT = REPO / 'results/forecast_approximation'


def main(a):
    os.environ['CUDA_VISIBLE_DEVICES'] = a.gpu
    import torch
    from scipy.stats import spearmanr
    from leaf.backbones import load
    from leaf.forecast.attention_forecast import AttentionForecast, _group_matrices
    from leaf.evaluation.data import load_split, iterate
    images, labels = load_split('calib'); report = {}
    for bb in a.backbones:
        vit, spec = load(bb); blocks = list(vit.blocks); L = len(blocks); H = blocks[0].attn.num_heads
        stations = {}
        for s in json.load(open(REPO / f'schedules/kappa_form/main_comparison/{bb}.json'))['schedules'].values():
            for l, lab in s['config']['groups'].items(): stations.setdefault(int(l), lab)
        energy = {}
        for l, lab in stations.items():
            S = _group_matrices(blocks[l + 1:], lab)[2][1].double()
            energy[l] = ((S[:, :a.rank] ** 2).sum(1) / (S ** 2).sum(1)).tolist()
        fut = {l: (AttentionForecast(blocks[l + 1:], lab, a.rank), AttentionForecast(blocks[l + 1:], lab, None),
                   AttentionForecast(blocks[l + 1:], list(range((L - 1 - l) * H)), None)) for l, lab in stations.items()}
        r32 = {l: [] for l in stations}; approx = {l: [] for l in stations}
        with torch.no_grad():
            for x, _ in iterate(images, labels, 32, 'cuda', spec.mean, spec.std, limit=a.images):
                h = vit.norm_pre(vit.patch_drop(vit._pos_embed(vit.patch_embed(x)))).float()
                size = h.new_ones(h.shape[0], h.shape[1], 1)
                for l, blk in enumerate(blocks):
                    mid = h + blk.ls1(blk.attn(blk.norm1(h)))
                    if l in fut:
                        d, full, exact = (f(mid, size)[:, 1:].cpu().numpy() for f in fut[l])
                        r32[l] += [spearmanr(u, v)[0] for u, v in zip(d, full)]
                        approx[l] += [spearmanr(u, v)[0] for u, v in zip(d, exact)]
                    h = mid + blk.ls2(blk.mlp(blk.norm2(mid)))
        report[bb] = {str(l): dict(energy_mean=float(np.mean(energy[l])), energy_min=float(np.min(energy[l])),
                                   spearman_r32_vs_full=float(np.mean(r32[l])), spearman_vs_exact=float(np.mean(approx[l])))
                      for l in stations}
        print(bb, report[bb])
        del vit, fut; torch.cuda.empty_cache()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'report.json').write_text(json.dumps(dict(rank=a.rank, groups=16, images=a.images, backbones=report), indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--gpu', default='0', choices=['0', '1'])
    ap.add_argument('--backbones', nargs='+', default=['deit_s', 'deit_b', 'mae_b'])
    ap.add_argument('--rank', type=int, default=32); ap.add_argument('--images', type=int, default=256)
    main(ap.parse_args())
