"""Cached eval splits (uint8, scripts/build_eval_cache.py), dense teacher logits and paired statistics.

Splits in data/cache: calib_bal (10 img/class, train shards 900-999; even images = 'half' for probes and
local search, odd images = 'odd' for choosing K), calib (first 10k of shards 1000+, grouping only),
confirm (shards 1008-1023), val (50k), v2 (ImageNet-V2 matched-frequency).
"""
import glob
import os

import numpy as np
import torch

CACHE = 'data/cache'
SUBSETS = {'half': ('calib_bal', slice(0, None, 2)), 'odd': ('calib_bal', slice(1, None, 2))}


def load_split(name, cache_dir=CACHE):
    """(uint8 images [N, S, S, 3] memmap, labels) of <name>_<S>_u8.npy: S = 224, or the window side of a segmentation
    split (ADE20K: 512, leaf/data/ade20k.py; labels are then the source image of each window)."""
    path = os.path.join(cache_dir, f'{name}_224_u8.npy')
    if not os.path.exists(path):
        found = sorted(glob.glob(os.path.join(glob.escape(cache_dir), f'{glob.escape(name)}_[0-9]*_u8.npy')))
        if len(found) == 1: path = found[0]
    images = np.load(path, mmap_mode='r')
    labels = np.load(os.path.join(cache_dir, f'{name}_labels.npy'))
    return images, labels


def iterate(images, labels, batch_size, device, mean, std, limit=None):
    """Normalized float [B,3,S,S] batches on device with int64 labels."""
    n = len(labels) if limit is None else min(limit, len(labels))
    mean = torch.tensor(mean, device=device).view(1, 3, 1, 1) * 255
    std = torch.tensor(std, device=device).view(1, 3, 1, 1) * 255
    for i in range(0, n, batch_size):
        j = min(i + batch_size, n)
        x = torch.from_numpy(np.array(images[i:j], copy=True, order='C')).to(device, non_blocking=True)
        yield (x.permute(0, 3, 1, 2).float() - mean) / std, torch.from_numpy(labels[i:j]).to(device, non_blocking=True)


def dense_logits(split, vit, spec, cache_dir=CACHE, batch_size=128):
    """Dense logits of the backbone on a split (bf16 autocast), stored once as float16:
    data/cache/<split>_<backbone>_logits.npy."""
    path = os.path.join(cache_dir, f'{split}_{spec.name}_logits.npy')
    if not os.path.exists(path):
        images, labels = load_split(split, cache_dir)
        chunks = []
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            for x, _ in iterate(images, labels, batch_size, 'cuda', spec.mean, spec.std):
                chunks.append(vit(x).float().half().cpu())
        tmp = f'{path}.{os.getpid()}.tmp.npy'          # concurrent workers: write privately, then rename
        np.save(tmp, torch.cat(chunks).numpy()); os.replace(tmp, path)
    return torch.from_numpy(np.load(path).astype(np.float32))


def split_arrays(name, vit, spec, limit=0):
    """(images, labels, teacher logits) of a split or of the calib_bal subsets 'half' / 'odd'."""
    base, sl = SUBSETS.get(name, (name, slice(None)))
    im, lb = load_split(base); te = dense_logits(base, vit, spec)
    im, lb, te = im[sl], lb[sl], te[sl]
    n = limit or len(lb)
    return im[:n], lb[:n], te[:n]


def paired_z(a, b):
    """z of mean(a - b) over paired per-image values (negative: a lower)."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(d.mean() / max(d.std(ddof=1) / np.sqrt(len(d)), 1e-12))


def mcnemar(correct_a, correct_b):
    """Exact-binomial McNemar p-value (normal approximation above 2000 discordant pairs)."""
    from math import comb, erf, sqrt
    b = int(np.sum(correct_a & ~correct_b)); c = int(np.sum(~correct_a & correct_b)); n = b + c
    if n == 0: return 1.0, b, c
    if n > 2000:
        z = max(0, abs(b - c) - 1) / sqrt(n)
        return 2 * (1 - 0.5 * (1 + erf(z / sqrt(2)))), b, c
    return min(1.0, 2 * sum(comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n), b, c


def holm(tests):
    """Adds 'holm' to each dict with a 'p' (step-down adjusted, monotone)."""
    order = sorted(range(len(tests)), key=lambda i: tests[i]['p']); run = 0.
    for r, i in enumerate(order):
        run = max(run, min(1., (len(tests) - r) * tests[i]['p'])); tests[i]['holm'] = run
    return tests
