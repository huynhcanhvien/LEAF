"""Zero-shot image-text retrieval of a reduced CLIP image tower (Flickr30k 1k test, MS-COCO 5k test).

The text side is the dense text tower, embedded once (leaf/data/clip_text.py); only the image tower is reduced.
A config is evaluated with the same module as for ImageNet (LeafRunner.inference_model), whose zero-shot head is
pointed at the caption embeddings, so its outputs are image-caption cosines [n_images, n_captions].

Stored per config, like Runner: <root>/<split>/runs/<key>.{json,npz} with the 0-based rank of the best correct
caption of every image (i2t) and of the correct image of every caption (t2i). Recall@k and the paired R@1
hits (McNemar, leaf.evaluation.data.mcnemar) follow from the ranks.
"""
import json
import os
import sys
import time

import numpy as np
import torch

from leaf.evaluation.data import iterate, load_split
from leaf.evaluation.runner import config_hash, save, load_json

SPLITS = ('flickr_test', 'coco_test')
RANKS = (1, 5, 10)


def captions(split, cache_dir='data/cache'):
    """(caption texts, image index of each caption)."""
    with open(os.path.join(cache_dir, f'{split}_captions.json')) as f:
        doc = json.load(f)
    return doc['captions'], np.asarray(doc['image'], dtype=np.int64)


def ranks(sim, owner):
    """0-based ranks from a similarity matrix [n_images, n_captions] and the image of each caption.

    i2t: captions scoring strictly above the best correct caption of the image; t2i: images scoring strictly
    above the correct image of the caption (ties count in favour of the correct item)."""
    sim = torch.as_tensor(sim, dtype=torch.float32); owner = torch.as_tensor(owner, dtype=torch.long)
    n_img, n_cap = sim.shape
    mine = owner[None, :] == torch.arange(n_img)[:, None]                       # [n_img, n_cap]
    best = sim.masked_fill(~mine, -torch.inf).max(1).values
    i2t = (sim > best[:, None]).sum(1)
    t2i = (sim > sim[owner, torch.arange(n_cap)][None, :]).sum(0)
    return i2t.numpy().astype(np.int32), t2i.numpy().astype(np.int32)


def recall(i2t, t2i, ranks=RANKS):
    """Recall@k (%) in both directions and their sum (Rsum, as in PiToMe and CLIP benchmarks)."""
    out = {f'i2t_r{r}': 100 * float(np.mean(np.asarray(i2t) < r)) for r in ranks}
    out.update({f't2i_r{r}': 100 * float(np.mean(np.asarray(t2i) < r)) for r in ranks})
    out['rsum'] = sum(out[f'{d}_r{r}'] for d in ('i2t', 't2i') for r in ranks)
    return out


@torch.no_grad()
def similarities(model, head, images, text, batch, device, mean, std, autocast=True):
    """Cosines of every image to every caption, with the zero-shot head pointed at the caption embeddings."""
    from leaf.model.clip_head import with_classifier
    rows = []
    with with_classifier(head, text):
        for x, _ in iterate(images, np.zeros(len(images), dtype=np.int64), batch, device, mean, std):
            with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=autocast):
                logits = model(x).float()
            if not bool(torch.isfinite(logits).all()): raise RuntimeError('nonfinite logits')
            rows.append((logits / head.scale).cpu())
    return torch.cat(rows)


class RetrievalRunner:
    """Retrieval evaluation of configs on a CLIP backbone, cached per config next to a LeafRunner's results.

        runner = LeafRunner('clip_b16', root='runs/clip_b16')
        ret = RetrievalRunner(runner)
        row = ret.run(cfg, 'coco_test')             # dict with recall@k, rsum and the per-query ranks
    """

    def __init__(self, runner, device='cuda'):
        self.runner, self.device = runner, device
        self._data = {}

    def data(self, split):
        if split not in SPLITS: raise ValueError(f'unknown retrieval split {split!r}; one of {SPLITS}')
        if split not in self._data:
            from leaf.backbones import text_embeddings
            cache = getattr(self.runner, 'data_cache', 'data/cache')
            images, _ = load_split(split, cache)
            _, owner = captions(split, cache)
            text = text_embeddings(self.runner.spec, split, cache)
            if len(text) != len(owner): raise RuntimeError(f'{split}: caption embeddings do not match the captions')
            n = self.runner.limit or None
            if n:                                     # smoke run: the first n images and their captions
                keep = owner < n; images, owner, text = images[:n], owner[keep], text[torch.from_numpy(keep)]
            self._data[split] = (images, owner, text)
        return self._data[split]

    def path(self, cfg, split):
        return self.runner.root / split / 'runs' / config_hash(cfg)

    def evaluate(self, cfg, split):
        _ = self.runner.vit
        images, owner, text = self.data(split)
        model = self.runner.inference_model(cfg)
        head = self.runner._cast.head
        sim = similarities(model, head, images, text, self.runner.batch, self.device, self.runner.spec.mean, self.runner.spec.std)
        i2t, t2i = ranks(sim, owner)
        return dict(i2t=i2t, t2i=t2i)

    def run(self, cfg, split, meta=None):
        """Recall@k, Rsum and per-query ranks (cached); None for an infeasible config (non-finite BF16 logits)."""
        p = self.path(cfg, split)
        if p.with_suffix('.json').exists():
            row = load_json(p.with_suffix('.json')); assert row['config'] == cfg, p
            if row.get('feasible') is False: return None
            return dict(row, **dict(np.load(p.with_suffix('.npz'))))
        start = time.monotonic()
        try:
            arr = self.evaluate(cfg, split)
        except RuntimeError as exc:
            if 'nonfinite' not in str(exc): raise
            save(p.with_suffix('.json'), dict(id=config_hash(cfg), config=cfg, feasible=False, reason=str(exc), **(meta or {})))
            return None
        p.parent.mkdir(parents=True, exist_ok=True); np.savez(p.with_suffix('.npz'), **arr)
        row = dict(id=config_hash(cfg), config=cfg, n_images=len(arr['i2t']), n_captions=len(arr['t2i']),
                   **recall(arr['i2t'], arr['t2i']), batch_size=self.runner.batch, seconds=time.monotonic() - start, **(meta or {}))
        save(p.with_suffix('.json'), row)
        print(json.dumps(dict(split=split, id=row['id'], rsum=round(row['rsum'], 2), i2t_r1=round(row['i2t_r1'], 2),
                              t2i_r1=round(row['t2i_r1'], 2), s=round(row['seconds'], 1))), file=sys.stderr, flush=True)
        return dict(row, **arr)
