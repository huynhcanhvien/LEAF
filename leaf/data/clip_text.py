"""Text embeddings of a CLIP backbone's own (dense, never reduced) text tower, stored once as unit-norm float32:

    <out-dir>/clip_text_<backbone>_imagenet.npy      [1000, D]  ImageNet classes: the 80 OpenAI templates per class,
                                                                each embedding normalised, averaged, normalised again
    <out-dir>/clip_text_<backbone>_<split>.npy       [n_captions, D]  the captions of a retrieval split
                                                                (<split>_captions.json from `leaf data prepare`)

Needs open_clip (pip install 'leaf[clip]'). Runs on the CPU (default) in a few minutes for ViT-B/16 and ~25 min for
ViT-L/14; `--device cuda` takes minutes on an idle card (FP32 either way: the embeddings agree to ~1e-6).

    python -m leaf data clip-text --backbone clip_b16 --task imagenet
    python -m leaf data clip-text --backbone clip_b16 --task flickr_test
"""
import argparse
import json
import os

import numpy as np
import torch

TASKS = ('imagenet', 'flickr_test', 'coco_test')


def text_model(spec, device):
    import open_clip
    if not spec.clip_text: raise ValueError(f'{spec.name} is not a CLIP backbone (no clip_text in backbones.json)')
    name, pretrained = spec.clip_text
    model, _, _ = open_clip.create_model_and_transforms(name, pretrained=pretrained, device=device)
    return model.eval(), open_clip.get_tokenizer(name)


@torch.no_grad()
def encode(model, tokenizer, texts, device, batch_size=256):
    """Unit-norm FP32 embeddings of a list of strings."""
    out = []
    for i in range(0, len(texts), batch_size):
        z = model.encode_text(tokenizer(texts[i:i + batch_size]).to(device)).float()
        out.append((z / z.norm(dim=-1, keepdim=True)).cpu())
        if len(texts) > 1000 and (i // batch_size) % 20 == 19: print(f'[clip-text] {i + batch_size}/{len(texts)} texts', flush=True)
    return torch.cat(out)


def imagenet_classifier(model, tokenizer, device):
    from open_clip import IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES
    rows = []
    for i, name in enumerate(IMAGENET_CLASSNAMES):
        z = encode(model, tokenizer, [t(name) for t in OPENAI_IMAGENET_TEMPLATES], device).mean(0)
        rows.append(z / z.norm())
        if (i + 1) % 100 == 0: print(f'[clip-text] imagenet {i + 1}/{len(IMAGENET_CLASSNAMES)} classes', flush=True)
    return torch.stack(rows)


def captions(split, cache_dir):
    path = os.path.join(cache_dir, f'{split}_captions.json')
    if not os.path.exists(path): raise FileNotFoundError(f'{path}: run `leaf data prepare --split {split}` first')
    with open(path) as f: return json.load(f)['captions']


def build(backbone, task, cache_dir='data/cache', device='cpu'):
    from leaf.backbones import SPECS, resolve
    spec = SPECS[resolve(backbone)]
    model, tokenizer = text_model(spec, device)
    emb = imagenet_classifier(model, tokenizer, device) if task == 'imagenet' else \
        encode(model, tokenizer, captions(task, cache_dir), device)
    path = os.path.join(cache_dir, f'clip_text_{spec.name}_{task}.npy')
    os.makedirs(cache_dir, exist_ok=True)
    tmp = f'{path}.{os.getpid()}.tmp.npy'
    np.save(tmp, emb.numpy().astype(np.float32)); os.replace(tmp, path)
    print(f'[clip-text] wrote {path} {tuple(emb.shape)}', flush=True)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--backbone', default='clip_b16')
    ap.add_argument('--task', choices=TASKS + ('all',), default='all')
    ap.add_argument('--out-dir', default='data/cache')
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args(argv)
    for task in (TASKS if args.task == 'all' else (args.task,)):
        build(args.backbone, task, args.out_dir, args.device)


if __name__ == '__main__':
    main()
