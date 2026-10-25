"""Evaluation of segmentation backbones (task 'segmentation'): per-pixel KL to the dense model and ADE20K mIoU.

Splits are fixed windows (leaf/data/ade20k.py); the runner's split names map to roles of the backbone's dataset:
'half' -> <dataset>_calib (the probes' calibration windows), 'calib' -> <dataset>_group (head grouping),
'val' -> <dataset>_val (every sliding window of the validation images).

Per window, the distortion is the KL(dense || model) of the class distribution at every patch position (decoder
logits on the 32x32 grid, before upsampling), averaged over positions. On 'val', windows are stitched as in
MMSegmentation's slide inference (each window's logits upsampled bilinearly to the window, summed, divided by the
overlap count, resized to the original image, argmax), and every image stores its per-class intersection, union,
predicted and labelled areas, so that mIoU (sum of intersections / sum of unions, mean over classes that occur) and
paired bootstrap intervals can be computed from cached runs.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROLES = {'half': 'calib', 'calib': 'group', 'val': 'val'}
IGNORE = 255


def base_split(spec, split):
    if split not in ROLES: raise ValueError(f'segmentation splits: {", ".join(ROLES)}; got {split!r}')
    return f'{spec.dataset}_{ROLES[split]}'


def pixel_kl(logits, teacher):
    """[B, K, h, w] logits -> [B] mean over positions of KL(teacher || model) over the K classes."""
    kl = F.kl_div(logits.float().log_softmax(1), teacher.float().log_softmax(1), log_target=True, reduction='none')
    return kl.sum(1).flatten(1).mean(1)


def load_windows(name, cache_dir):
    return dict(np.load(Path(cache_dir) / f'{name}_windows.npz'))


class Stitcher:
    """Streams window logits in window order and returns per-image area statistics when each image is complete."""

    def __init__(self, meta, n_classes, window):
        self.meta, self.k, self.window = meta, n_classes, window
        self.current, self.canvas, self.count = -1, None, None
        self.rows = []                       # (image, intersect, union, pred_area, label_area)

    def add(self, logits, first):
        """logits [b, K, h, w] of windows first .. first + b - 1."""
        m = self.meta
        for j in range(logits.shape[0]):
            i = first + j; image = int(m['image'][i])
            if image != self.current:
                self.finish()
                h, w = (int(v) for v in m['resized'][image])
                self.current = image
                self.canvas = logits.new_zeros(self.k, h, w, dtype=torch.float32)
                self.count = logits.new_zeros(1, h, w, dtype=torch.float32)
            y, x, h, w = (int(m[key][i]) for key in ('y', 'x', 'h', 'w'))
            up = F.interpolate(logits[j:j + 1].float(), size=(self.window, self.window), mode='bilinear', align_corners=False)
            self.canvas[:, y:y + h, x:x + w] += up[0, :, :h, :w]
            self.count[:, y:y + h, x:x + w] += 1

    def finish(self):
        if self.current < 0: return
        m, image = self.meta, self.current
        seg = self.canvas / self.count
        size = tuple(int(v) for v in m['original'][image])
        pred = resized_argmax(seg, size)
        gt = torch.from_numpy(m['gt'][m['gt_offset'][image]:m['gt_offset'][image + 1]].reshape(size)).to(pred.device)
        self.rows.append((image, *area_stats(pred, gt.long(), self.k)))
        self.current, self.canvas, self.count = -1, None, None

    def result(self):
        self.finish()
        images = [r[0] for r in self.rows]
        if images != list(range(len(images))): raise RuntimeError('windows are not in image order')
        return {key: np.stack([r[i + 1] for r in self.rows]) for i, key in enumerate(('intersect', 'union', 'pred_area', 'label_area'))}


def resized_argmax(seg, size, chunk=50):
    """argmax over classes of seg [K, h, w] bilinearly resized to size, a chunk of classes at a time (resizing is per
    channel, so this equals resizing all K at once; large images would need K x H x W floats). Ties keep the lowest class."""
    best = idx = None
    for start in range(0, seg.shape[0], chunk):
        up = F.interpolate(seg[None, start:start + chunk], size=size, mode='bilinear', align_corners=False)[0]
        value, arg = up.max(0)
        if best is None: best, idx = value, arg
        else:
            better = value > best
            best = torch.where(better, value, best); idx = torch.where(better, arg + start, idx)
    return idx


def area_stats(pred, gt, k):
    """MMSegmentation's intersect_and_union: per-class areas over labelled pixels (int64 numpy)."""
    keep = gt != IGNORE
    pred, gt = pred[keep], gt[keep]
    inter = torch.bincount(pred[pred == gt], minlength=k)
    area_pred = torch.bincount(pred, minlength=k); area_gt = torch.bincount(gt, minlength=k)
    union = area_pred + area_gt - inter
    return tuple(v[:k].cpu().numpy().astype(np.int64) for v in (inter, union, area_pred, area_gt))


def miou(arr, images=None):
    """mIoU (%) of per-image area arrays, over all images or an index array (bootstrap); classes without union skipped."""
    inter, union = arr['intersect'], arr['union']
    if images is not None: inter, union = inter[images], union[images]
    inter, union = inter.sum(0).astype(np.float64), union.sum(0).astype(np.float64)
    with np.errstate(invalid='ignore', divide='ignore'):
        return 100 * float(np.nanmean(np.where(union > 0, inter / union, np.nan)))


def pixel_accuracy(arr):
    return 100 * float(arr['intersect'].sum() / max(arr['label_area'].sum(), 1))


def summary(arr):
    out = dict(n=int(len(arr['kl'])), kl=float(np.mean(arr['kl'])), agree=100 * float(np.mean(arr['agree'])))
    if 'intersect' in arr: out.update(images=int(len(arr['intersect'])), miou=miou(arr), aacc=pixel_accuracy(arr))
    return out


def paired_bootstrap(a, b, n=10_000, seed=0):
    """mIoU(a) - mIoU(b) over the same images: point estimate, 95% percentile interval and two-sided p (images resampled)."""
    rng = np.random.default_rng(seed); m = len(a['intersect'])
    diffs = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, m, m); diffs[i] = miou(a, idx) - miou(b, idx)
    delta = miou(a) - miou(b)
    p = min(1., 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
    return dict(delta=delta, lo=float(np.percentile(diffs, 2.5)), hi=float(np.percentile(diffs, 97.5)), p=float(p))


@torch.no_grad()
def evaluate(runner, cfg, split):
    """Per-window kl / agree (and per-image area statistics on 'val') of one config: Runner._evaluate for segmentation."""
    from leaf.evaluation.data import iterate
    from leaf.evaluation.gpu_coordination import gpu_work
    spec, device = runner.spec, runner.device
    with gpu_work():
        images, owners, teacher = runner.data(split)
        model = runner.inference_model(cfg)
    stitch = None
    if ROLES[split] == 'val':
        stitch = Stitcher(load_windows(base_split(spec, split), runner.data_cache), spec.num_classes, spec.img_size)
    kls, agree, seen = [], [], 0
    for x, _ in iterate(images, owners, runner.batch, device, spec.mean, spec.std):
        with gpu_work():
            with torch.autocast(device, dtype=torch.bfloat16): logits = model(x).float()
            if not bool(torch.isfinite(logits).all()): raise RuntimeError('nonfinite logits')
            t = torch.from_numpy(np.asarray(teacher[seen:seen + len(x)], dtype=np.float32)).to(device)
            kls.append(pixel_kl(logits, t).cpu().numpy())
            agree.append(logits.argmax(1).eq(t.argmax(1)).float().flatten(1).mean(1).cpu().numpy())
            if stitch is not None: stitch.add(logits, seen)
            seen += len(x)
    if seen != len(owners): raise RuntimeError('incomplete evaluation')
    out = dict(kl=np.concatenate(kls).astype(np.float32), agree=np.concatenate(agree).astype(np.float32))
    if stitch is not None:
        image = stitch.meta['image']
        if seen < len(image) and image[seen] == image[seen - 1]:
            stitch.current = -1             # a limited (smoke) run cut the windows of its last image: drop that image
        out.update(stitch.result())
    with gpu_work():
        del model
        if device == 'cuda': torch.cuda.empty_cache()
    return out
