"""ImageNet-ReaL re-scoring of the main campaign (CPU only; no new inference).

python leaf/reproduction/experiments/imagenet_real.py

ReaL labels (Beyer et al. 2020, https://github.com/google-research/reassessed-imagenet, data/real/real.json) re-assess the
50k ImageNet val images with multi-label sets. Predictions of every arm are already stored per image
(results/main_comparison/<bb>/val/runs/<sha>.npz, field `pred`). The cache order of val images is the tar
order of the WebDataset validation shards; data/real/val_order_ids.npy maps cache position -> ILSVRC2012 val id
(verified: the original label is in the ReaL set for 90% of the 46,837 images that have one, as published).
ReaL top-1: prediction in the label set, over images with a non-empty set. Pairs are the latency pairs of report.json;
McNemar on per-image ReaL correctness, Holm over all pairs.
"""
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
import numpy as np

from leaf.evaluation.data import mcnemar, holm

MAIN_COMPARISON = REPO / 'results/main_comparison'
OUT = REPO / 'results/imagenet_real'


def real_sets():
    real = json.load(open(REPO / 'data/real/real.json')); ids = np.load(REPO / 'data/real/val_order_ids.npy')
    sets = [set(real[i - 1]) for i in ids]
    return sets, np.array([bool(s) for s in sets])


def main():
    sets, mask = real_sets(); rep = json.load(open(MAIN_COMPARISON / 'report.json')); arms = {}
    def correct(bb, sha):
        key = (bb, sha)
        if key not in arms:
            pred = np.load(MAIN_COMPARISON / bb / 'val' / 'runs' / f'{sha}.npz')['pred']
            arms[key] = np.array([int(p) in s for p, s in zip(pred, sets)])[mask]
        return arms[key]
    sha = {(bb, a['name']): a['sha'] for bb, al in rep['arms'].items() for a in al}
    tests = []
    for t in rep['tests']:
        a, b = correct(t['backbone'], sha[(t['backbone'], t['ours'])]), correct(t['backbone'], sha[(t['backbone'], t['ref'])])
        p, _, _ = mcnemar(a, b)
        tests.append(dict(backbone=t['backbone'], ours=t['ours'], family=t['family'], ref=t['ref'],
                          real=dict(delta=float((a.mean() - b.mean()) * 100), p=p, ours=float(a.mean() * 100), ref=float(b.mean() * 100))))
    holm([t['real'] for t in tests])
    acc = {f'{bb}/{name}': float(correct(bb, s).mean() * 100) for (bb, name), s in sha.items()}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'report.json').write_text(json.dumps(dict(n_images=int(mask.sum()), tests=tests, accuracy=acc), indent=1))
    fam = {}
    for t in tests:
        s = fam.setdefault(t['family'], [0, 0, 0, []]); s[0] += 1; s[3].append(t['real']['delta'])
        if t['real']['holm'] < .05: s[1 if t['real']['delta'] > 0 else 2] += 1
    for f, (n, w, l, d) in sorted(fam.items(), key=lambda kv: -kv[1][0]):
        print(f'{f:18s} pairs {n:3d}  ReaL W/L {w}/{l}  mean d {np.mean(d):+.2f}')
    print('images with ReaL labels:', int(mask.sum()))


if __name__ == '__main__':
    main()
