"""ADE20K (SceneParsing, 150 classes) as fixed 512x512 uint8 windows for segmentation backbones.

The test pipeline is MMSegmentation's for Segmenter on ADE20K: decode with OpenCV (BGR -> RGB), rescale keeping the
aspect ratio into (2048, 512) (short side 512 unless the long side would exceed 2048; cv2 bilinear), then slide a
512x512 window with stride 480 (the last window aligned to the border). Labels: reduce_zero_label (0 -> 255
ignored, class c -> c - 1). Normalization is applied later on the GPU (leaf/evaluation/data.py).

Outputs (in --out-dir, default data/cache), all from data/ade/ADEChallengeData2016:
    ade20k_calib_512_u8.npy [2000, 512, 512, 3]  + ade20k_calib_labels.npy (source image index)
        training images, one centre 512x512 window each: the unlabelled calibration set of the probes ('half')
    ade20k_group_512_u8.npy [1024, ...]           + ade20k_group_labels.npy
        further training images (disjoint), the head-grouping set ('calib')
    ade20k_val_512_u8.npy   [W, ...]              + ade20k_val_labels.npy (image index of each window)
        every sliding window of the 2,000 validation images, in image order
    ade20k_val_windows.npz  per window: image, y, x, h, w (valid part; windows of images smaller than 512 are
        zero-padded at the bottom/right); per image: resized (H', W'), original (H, W); ground truth: gt (flat
        uint8, reduce_zero_label applied) and gt_offset
    ade20k_<split>_files.json  the source file of every image

Training images are chosen by a seeded permutation of the sorted file list (calibration first, then grouping).

Usage:
    python -m leaf data ade20k --download                # fetch and unzip ADEChallengeData2016.zip (~1 GB) first
    python -m leaf data ade20k --split all
"""
import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np

URL = 'http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip'
SCALE = (2048, 512)          # (long edge cap, short edge cap) of MMSegmentation's keep-ratio Resize
WINDOW, STRIDE = 512, 480


def read_rgb(path):
    import cv2
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None: raise OSError(f'cannot decode {path}')
    return np.ascontiguousarray(img[:, :, ::-1])


def rescale(img, scale=SCALE):
    """mmcv.imrescale(img, scale): the largest factor that fits the long edge into max(scale) and the short edge
    into min(scale); size rounded as int(x * f + 0.5); cv2 bilinear."""
    import cv2
    h, w = img.shape[:2]
    f = min(max(scale) / max(h, w), min(scale) / min(h, w))
    size = (int(w * f + .5), int(h * f + .5))
    return cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)


def windows(h, w, window=WINDOW, stride=STRIDE):
    """(y, x, h, w) of MMSegmentation's slide inference, row-major."""
    def starts(n):
        count = max(n - window + stride - 1, 0) // stride + 1
        out = []
        for i in range(count):
            end = min(i * stride + window, n)
            out.append((max(end - window, 0), end))
        return out
    return [(y0, x0, y1 - y0, x1 - x0) for y0, y1 in starts(h) for x0, x1 in starts(w)]


def pad(crop, window=WINDOW):
    out = np.zeros((window, window, 3), np.uint8)
    out[:crop.shape[0], :crop.shape[1]] = crop
    return out


def centre_window(path):
    img = rescale(read_rgb(path)); h, w = img.shape[:2]
    y, x = max(h - WINDOW, 0) // 2, max(w - WINDOW, 0) // 2
    return pad(img[y:y + WINDOW, x:x + WINDOW])


def read_label(path):
    from PIL import Image
    gt = np.array(Image.open(path), dtype=np.uint8)
    out = gt - 1                             # reduce_zero_label: 0 (other) -> 255, c -> c - 1
    out[gt == 0] = 255
    return out


def val_image(paths):
    img_path, ann_path = paths
    original = read_rgb(img_path); img = rescale(original)
    rows = windows(*img.shape[:2])
    crops = [pad(img[y:y + h, x:x + w]) for y, x, h, w in rows]
    return crops, rows, img.shape[:2], original.shape[:2], read_label(ann_path)


def write(out_dir, name, images, labels, files):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    tmp = out / f'{name}_{WINDOW}_u8.{os.getpid()}.tmp.npy'
    mm = np.lib.format.open_memmap(tmp, mode='w+', dtype=np.uint8, shape=(len(images), WINDOW, WINDOW, 3))
    for i, im in enumerate(images): mm[i] = im
    mm.flush(); del mm; tmp.replace(out / f'{name}_{WINDOW}_u8.npy')
    np.save(out / f'{name}_labels.npy', np.asarray(labels, np.int32))
    (out / f'{name}_files.json').write_text(json.dumps(files) + '\n')


def build_train(root, out_dir, calib, group, seed, workers):
    files = sorted(p.name for p in (Path(root) / 'images' / 'training').glob('*.jpg'))
    if len(files) != 20210: raise SystemExit(f'expected 20,210 ADE20K training images, found {len(files)}')
    pick = np.random.default_rng(seed).permutation(len(files))
    for name, idx in (('ade20k_calib', pick[:calib]), ('ade20k_group', pick[calib:calib + group])):
        chosen = [files[i] for i in idx]
        with Pool(workers) as pool:
            images = pool.map(centre_window, [Path(root) / 'images' / 'training' / f for f in chosen], chunksize=16)
        write(out_dir, name, images, idx, chosen)
        print(f'{name}: {len(images)} windows', flush=True)


def build_val(root, out_dir, workers):
    files = sorted(p.stem for p in (Path(root) / 'images' / 'validation').glob('*.jpg'))
    if len(files) != 2000: raise SystemExit(f'expected 2,000 ADE20K validation images, found {len(files)}')
    jobs = [(Path(root) / 'images' / 'validation' / f'{f}.jpg', Path(root) / 'annotations' / 'validation' / f'{f}.png')
            for f in files]
    crops, owner, rows, resized, original, gt, offset = [], [], [], [], [], [], [0]
    with Pool(workers) as pool:
        for i, (c, r, rs, og, label) in enumerate(pool.imap(val_image, jobs, chunksize=8)):
            if label.shape != og: raise SystemExit(f'{files[i]}: label {label.shape} != image {og}')
            crops += c; owner += [i] * len(c); rows += r; resized.append(rs); original.append(og)
            gt.append(label.ravel()); offset.append(offset[-1] + label.size)
    write(out_dir, 'ade20k_val', crops, owner, [f'{f}.jpg' for f in files])
    rows = np.asarray(rows, np.int32)
    np.savez(Path(out_dir) / 'ade20k_val_windows.npz', image=np.asarray(owner, np.int32), y=rows[:, 0], x=rows[:, 1],
             h=rows[:, 2], w=rows[:, 3], resized=np.asarray(resized, np.int32), original=np.asarray(original, np.int32),
             gt=np.concatenate(gt), gt_offset=np.asarray(offset, np.int64))
    print(f'ade20k_val: {len(files)} images, {len(crops)} windows', flush=True)


def download(data_dir):
    import zipfile
    import torch
    dest = Path(data_dir); dest.mkdir(parents=True, exist_ok=True)
    archive = dest / os.path.basename(URL)
    if not archive.exists(): torch.hub.download_url_to_file(URL, str(archive))
    with zipfile.ZipFile(archive) as z: z.extractall(dest)


def main(argv=None):
    ap = argparse.ArgumentParser(prog='leaf data ade20k')
    ap.add_argument('--data-dir', default='data/ade', help='folder holding ADEChallengeData2016/')
    ap.add_argument('--out-dir', default='data/cache')
    ap.add_argument('--split', choices=('train', 'val', 'all'), default='all', help='train: calibration and grouping sets')
    ap.add_argument('--download', action='store_true', help=f'fetch and unzip {URL} first')
    ap.add_argument('--calib', type=int, default=2000)
    ap.add_argument('--group', type=int, default=1024)
    ap.add_argument('--seed', type=int, default=151)
    ap.add_argument('--workers', type=int, default=16)
    args = ap.parse_args(argv)
    if args.download: download(args.data_dir)
    root = Path(args.data_dir) / 'ADEChallengeData2016'
    if args.split in ('train', 'all'): build_train(root, args.out_dir, args.calib, args.group, args.seed, args.workers)
    if args.split in ('val', 'all'): build_val(root, args.out_dir, args.workers)


if __name__ == '__main__':
    main()
