"""
Pre-process ImageNet evaluation images once into uint8 memmaps, so that every evaluation reads tensors instead
of decoding JPEGs.

Transform = the standard DeiT eval pipeline: Resize(256, bicubic) -> CenterCrop(224).
Normalization is applied later on the GPU, per backbone (leaf/evaluation/data.py).

Outputs (in --out-dir, default data/cache):
    val_224_u8.npy       [50000, 224, 224, 3] uint8 + val_labels.npy       ImageNet-1k validation
    calib_224_u8.npy     [10000, ...]               + calib_labels.npy     first 10k images of train shards 1000..
    calib_bal_224_u8.npy [10000, ...]               + calib_bal_labels.npy 10 images/class from train shards 900..999
                                                      (+ calib_bal_keys.json); even indices = the 5,000 probe images
    confirm_224_u8.npy   [20018, ...]               + confirm_labels.npy   train shards 1008..1023
    v2_224_u8.npy        [10000, ...]               + v2_labels.npy        ImageNet-V2 matched-frequency
    flickr_test_224_u8.npy [1000, ...]  + flickr_test_captions.json   Flickr30k 1k test (Karpathy), CLIP retrieval
    coco_test_224_u8.npy   [5000, ...]  + coco_test_captions.json     MS-COCO 5k test (Karpathy), CLIP retrieval
                                          (labels = image index; captions.json: captions and the image of each)

The calibration split is the first --calib-size images of train shards 1000.. in shard order (deterministic,
no shuffling); the planner uses its first 1,024 images for head grouping. The confirmation split (train shards
1008..1023) is disjoint from calibration (which ends inside shard 1007) and from calib_bal (shards 900..999).
Validation and V2 images are never used for planning.

Usage:
    python -m leaf data prepare --split val            # or: python -m leaf.data.build_cache --split val
    python -m leaf data prepare --split calib_bal
    python -m leaf data prepare --split v2 --v2-tar imagenetv2-matched-frequency.tar.gz
    python -m leaf data prepare --split flickr_test     # --retrieval-dir: the downloaded nlphuji csv + zip files

Retrieval images use CLIP's own transform, Resize(224, bicubic) -> CenterCrop(224), so dense recalls match the
published ones. Their files come from the Hugging Face datasets nlphuji/flickr_1k_test_image_text_retrieval and
nlphuji/mscoco_2014_5k_test_image_text_retrieval (fetched with --download, about 1 GB).
"""
import argparse
import glob
import io
import json
import os
import tarfile
from multiprocessing import Pool

import numpy as np
from PIL import Image
from torchvision import transforms

SIDE = 224          # image side of the cache; set by --side (2026-10-25: 384-px backbones, e.g. ViT-B/16-384)
TRANSFORM = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
])


def configure(side, crop_pct):
    """Resize(side / crop_pct, bicubic) -> CenterCrop(side); the default (224, 0.875) is the DeiT pipeline above."""
    global SIDE, TRANSFORM
    SIDE = int(side)
    TRANSFORM = transforms.Compose([transforms.Resize(int(round(side / crop_pct)), interpolation=transforms.InterpolationMode.BICUBIC),
                                    transforms.CenterCrop(side)])
CLIP_TRANSFORM = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
])
RETRIEVAL = {  # split: (Hugging Face dataset, captions csv, images zip, expected images, expected captions)
    "flickr_test": ("nlphuji/flickr_1k_test_image_text_retrieval", "test_1k_flickr.csv", "images_flickr_1k_test.zip", 1000, 5000),
    "coco_test": ("nlphuji/mscoco_2014_5k_test_image_text_retrieval", "test_5k_mscoco_2014.csv",
                  "images_mscoco_2014_5k_test.zip", 5000, 25010),
}


def read_shard(path):
    """Decode one WebDataset shard -> (uint8 [n,224,224,3], int64 [n]) in tar order."""
    samples = {}
    order = []
    with tarfile.open(path) as tar:
        for member in tar:
            if not member.isfile():
                continue
            key, ext = member.name.rsplit(".", 1)
            if key not in samples:
                samples[key] = {}
                order.append(key)
            samples[key][ext] = tar.extractfile(member).read()
    images, labels = [], []
    for key in order:
        s = samples[key]
        if "jpg" not in s or "cls" not in s:
            continue
        img = Image.open(io.BytesIO(s["jpg"])).convert("RGB")
        images.append(np.asarray(TRANSFORM(img), dtype=np.uint8))
        labels.append(int(s["cls"].decode("utf-8").strip()))
    return np.stack(images), np.asarray(labels, dtype=np.int64)


def read_shard_labels(path):
    """Labels of one shard without decoding images: [(key, label)] in tar order."""
    rows = []
    with tarfile.open(path) as tar:
        for member in tar:
            if member.isfile() and member.name.endswith(".cls"):
                rows.append((member.name[:-4], int(tar.extractfile(member).read().decode("utf-8").strip())))
    return path, rows


def read_shard_selected(job):
    """Decode only the selected keys of one shard, in the requested key order."""
    path, keys = job
    wanted, found = set(keys), {}
    with tarfile.open(path) as tar:
        for member in tar:
            if not member.isfile():
                continue
            key, ext = member.name.rsplit(".", 1)
            if key in wanted and ext == "jpg":
                img = Image.open(io.BytesIO(tar.extractfile(member).read())).convert("RGB")
                found[key] = np.asarray(TRANSFORM(img), dtype=np.uint8)
    missing = wanted - set(found)
    if missing:
        raise RuntimeError(f"{path}: missing jpg for {sorted(missing)[:3]}")
    return path, np.stack([found[k] for k in keys])


def build_balanced(shards, per_class, n_classes, seed, out_dir, name, workers):
    """Exactly `per_class` images of every class, chosen with a fixed seed among the shards.

    The key list (shard, key, label) is written next to the cache so the split is reproducible
    and disjointness from other splits can be audited by shard range."""
    with Pool(workers) as pool:
        listing = dict(pool.imap(read_shard_labels, shards))
    by_class = {c: [] for c in range(n_classes)}
    for path in shards:
        for key, label in listing[path]:
            by_class[label].append((os.path.basename(path), key))
    rng = np.random.default_rng(seed)
    chosen = []
    for c in range(n_classes):
        pool_c = by_class[c]
        if len(pool_c) < per_class:
            raise RuntimeError(f"class {c} has only {len(pool_c)} images in the shard range")
        for i in sorted(rng.choice(len(pool_c), per_class, replace=False)):
            chosen.append((*pool_c[i], c))
    chosen.sort()  # shard, key order: deterministic decode and memmap layout
    per_shard = {}
    for shard, key, _ in chosen:
        per_shard.setdefault(shard, []).append(key)
    root = os.path.dirname(shards[0])
    jobs = [(os.path.join(root, s), keys) for s, keys in sorted(per_shard.items())]
    out_img = os.path.join(out_dir, f"{name}_{SIDE}_u8.npy")
    mm = np.lib.format.open_memmap(out_img, mode="w+", dtype=np.uint8, shape=(len(chosen), SIDE, SIDE, 3))
    offsets, pos = {}, 0
    for path, keys in jobs:
        offsets[path] = pos
        pos += len(keys)
    with Pool(workers) as pool:
        for path, imgs in pool.imap_unordered(read_shard_selected, jobs):
            mm[offsets[path]:offsets[path] + len(imgs)] = imgs
            print(f"[{name}] {os.path.basename(path)} +{len(imgs)}", flush=True)
    mm.flush()
    labels = np.asarray([c for _, _, c in chosen], dtype=np.int64)
    np.save(os.path.join(out_dir, f"{name}_labels.npy"), labels)
    with open(os.path.join(out_dir, f"{name}_keys.json"), "w") as f:
        json.dump(dict(seed=seed, per_class=per_class, shards=[os.path.basename(s) for s in shards],
                       rows=[list(r) for r in chosen]), f)
    print(f"[{name}] wrote {out_img} {mm.shape}, counts per class {np.bincount(labels).min()}-{np.bincount(labels).max()}")


def build(shards, limit, out_dir, name, workers):
    out_img = os.path.join(out_dir, f"{name}_{SIDE}_u8.npy")
    out_lbl = os.path.join(out_dir, f"{name}_labels.npy")
    chunks_img, chunks_lbl, total = [], [], 0
    with Pool(workers) as pool:
        for imgs, lbls in pool.imap(read_shard, shards):
            take = imgs.shape[0] if limit is None else min(imgs.shape[0], limit - total)
            chunks_img.append(imgs[:take])
            chunks_lbl.append(lbls[:take])
            total += take
            print(f"[{name}] {total} images", flush=True)
            if limit is not None and total >= limit:
                pool.terminate()
                break
    mm = np.lib.format.open_memmap(out_img, mode="w+", dtype=np.uint8, shape=(total, SIDE, SIDE, 3))
    pos = 0
    for c in chunks_img:
        mm[pos:pos + len(c)] = c
        pos += len(c)
    mm.flush()
    labels = np.concatenate(chunks_lbl)
    np.save(out_lbl, labels)
    print(f"[{name}] wrote {out_img} {mm.shape}, {len(np.unique(labels))} classes")


def build_v2(tar_path, out_dir):
    """ImageNet-V2 matched-frequency -> v2_224_u8.npy (+ labels); folder name = class index; sorted by class."""
    img, lab = [], []
    with tarfile.open(tar_path, "r:*") as tf:
        for m in tf:
            if not m.isfile() or not m.name.lower().endswith((".jpeg", ".jpg", ".png")): continue
            cls = int(os.path.basename(os.path.dirname(m.name))); im = Image.open(io.BytesIO(tf.extractfile(m).read())).convert("RGB")
            img.append(np.asarray(TRANSFORM(im), dtype=np.uint8)); lab.append(cls)
    order = np.lexsort((np.arange(len(lab)), np.array(lab)))
    np.save(os.path.join(out_dir, f"v2_{SIDE}_u8.npy"), np.stack(img)[order])
    np.save(os.path.join(out_dir, "v2_labels.npy"), np.array(lab, dtype=np.int64)[order])
    print(len(lab), "images", len(set(lab)), "classes")


def build_retrieval(split, retrieval_dir, out_dir, download=False):
    """Karpathy test split of Flickr30k / MS-COCO -> <split>_224_u8.npy, <split>_labels.npy (image index) and
    <split>_captions.json (captions in csv order, the image index of each)."""
    import csv
    import zipfile
    repo, csv_name, zip_name, n_img, n_cap = RETRIEVAL[split]
    root = os.path.join(retrieval_dir, repo.split("/")[1])
    if download:
        from huggingface_hub import hf_hub_download
        for f in (csv_name, zip_name): hf_hub_download(repo, f, repo_type="dataset", local_dir=root)
    with open(os.path.join(root, csv_name), newline="") as f:
        rows = list(csv.DictReader(f))
    with zipfile.ZipFile(os.path.join(root, zip_name)) as zf:
        members = {os.path.basename(m): m for m in zf.namelist() if m.lower().endswith((".jpg", ".jpeg", ".png"))
                   and not os.path.basename(m).startswith(".")}
        images, captions, owner = [], [], []
        for i, row in enumerate(rows):
            with zf.open(members[row["filename"]]) as fh:
                img = Image.open(io.BytesIO(fh.read())).convert("RGB")
            images.append(np.asarray(CLIP_TRANSFORM(img), dtype=np.uint8))
            caps = json.loads(row["raw"])
            captions += [c.strip() for c in caps]; owner += [i] * len(caps)
    if (len(images), len(captions)) != (n_img, n_cap):
        raise RuntimeError(f"{split}: {len(images)} images / {len(captions)} captions, expected {n_img} / {n_cap}")
    np.save(os.path.join(out_dir, f"{split}_224_u8.npy"), np.stack(images))
    np.save(os.path.join(out_dir, f"{split}_labels.npy"), np.arange(len(images), dtype=np.int64))
    with open(os.path.join(out_dir, f"{split}_captions.json"), "w") as f:
        json.dump(dict(source=repo, filenames=[r["filename"] for r in rows], captions=captions, image=owner), f)
    print(f"[{split}] wrote {len(images)} images, {len(captions)} captions")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/imagenet_tar")
    ap.add_argument("--out-dir", default="data/cache")
    ap.add_argument("--split", choices=["val", "calib", "both", "confirm", "calib_bal", "v2", *RETRIEVAL], default="both")
    ap.add_argument("--v2-tar", default="data/imagenetv2-matched-frequency.tar.gz",
                    help="ImageNet-V2 matched-frequency archive (for --split v2)")
    ap.add_argument("--retrieval-dir", default="data/retrieval",
                    help="the nlphuji csv + zip files of the retrieval splits (one folder per dataset)")
    ap.add_argument("--download", action="store_true", help="retrieval splits: fetch the files from the Hugging Face Hub")
    ap.add_argument("--limit", type=int, default=None,
                    help="val/confirm only: stop after this many images (a quick check of the pipeline)")
    ap.add_argument("--per-class", type=int, default=10)
    ap.add_argument("--bal-first-shard", type=int, default=900)
    ap.add_argument("--bal-last-shard", type=int, default=999)
    ap.add_argument("--bal-seed", type=int, default=151)
    ap.add_argument("--bal-name", default="calib_bal")
    ap.add_argument("--calib-size", type=int, default=10_000)
    ap.add_argument("--calib-first-shard", type=int, default=1000)
    ap.add_argument("--confirm-first-shard", type=int, default=1008)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--side", type=int, default=224, help="image side of the cache (224: Resize 256 -> CenterCrop 224)")
    ap.add_argument("--crop-pct", type=float, default=0.875, help="crop fraction: Resize(side / crop_pct) -> CenterCrop(side)")
    args = ap.parse_args(argv)
    if (args.side, args.crop_pct) != (224, 0.875): configure(args.side, args.crop_pct)
    os.makedirs(args.out_dir, exist_ok=True)

    if args.split in ("val", "both"):
        shards = sorted(glob.glob(os.path.join(args.data_dir, "imagenet1k-validation-*.tar")))
        build(shards, args.limit, args.out_dir, "val", args.workers)
    if args.split in ("calib", "both"):
        shards = sorted(glob.glob(os.path.join(args.data_dir, "imagenet1k-train-*.tar")))
        shards = [s for s in shards if int(s.rsplit("-", 1)[1].split(".")[0]) >= args.calib_first_shard]
        build(shards, args.calib_size, args.out_dir, "calib", args.workers)
    if args.split == "confirm":
        shards = sorted(glob.glob(os.path.join(args.data_dir, "imagenet1k-train-*.tar")))
        shards = [s for s in shards if int(s.rsplit("-", 1)[1].split(".")[0]) >= args.confirm_first_shard]
        build(shards, args.limit, args.out_dir, "confirm", args.workers)
    if args.split == "calib_bal":
        # Disjoint by shard range from calib (1000..1007) and confirm (1008..1023).
        if args.bal_last_shard >= min(args.calib_first_shard, args.confirm_first_shard):
            raise ValueError("balanced split must end before the calib/confirm shards")
        shards = sorted(glob.glob(os.path.join(args.data_dir, "imagenet1k-train-*.tar")))
        shards = [s for s in shards
                  if args.bal_first_shard <= int(s.rsplit("-", 1)[1].split(".")[0]) <= args.bal_last_shard]
        build_balanced(shards, args.per_class, 1000, args.bal_seed, args.out_dir, args.bal_name, args.workers)
    if args.split == "v2":
        build_v2(args.v2_tar, args.out_dir)
    if args.split in RETRIEVAL:
        build_retrieval(args.split, args.retrieval_dir, args.out_dir, args.download)


if __name__ == "__main__":
    main()
