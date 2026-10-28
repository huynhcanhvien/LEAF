"""Cached evaluation of model configs on one backbone.

Runner(backbone).run(cfg, split) evaluates once and stores, under <root>/<split>/runs/<key>.{json,npz},
the per-image KL(dense || model), correctness and prediction. key = sha256(config JSON)[:16]; a config
is a JSON dict with 'kind' ('pd' for the method, or a baseline name) and its parameters.
All arms of one comparison use the same batch size (BF16 KL depends on it).
"""
import gc
import hashlib
import json
import sys
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from leaf.backbones import load
from leaf.evaluation.data import iterate, split_arrays
from leaf.evaluation.gpu_coordination import gpu_work

RESULTS = Path('runs')
BATCH = 128


def save(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f'.{os.getpid()}.tmp'); tmp.write_text(json.dumps(obj, indent=1, default=float)); tmp.replace(path)


def load_json(path):
    return json.loads(Path(path).read_text())


def config_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def per_image_kl(logits, teacher):
    return F.kl_div(logits.float().log_softmax(-1), teacher.float().log_softmax(-1), log_target=True, reduction='none').sum(-1)


def metrics(arr):
    """Summary row of per-image arrays: n, top-1 (%) and mean KL; segmentation (per-window 'agree' with the dense argmax):
    n, mean KL, agreement (%) and, with area statistics, mIoU (leaf.evaluation.segmentation)."""
    if 'agree' in arr:
        from leaf.evaluation.segmentation import summary
        return summary(arr)
    return dict(n=len(arr['kl']), top1=100 * float(arr['correct'].mean()), kl=float(arr['kl'].mean()))


def require_idle_gpu(idle=True):
    """Refuse a busy card (timing rule: never share a GPU); idle=True also refuses other processes on it (latency)."""
    index = int(os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0])
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,pci.bus_id', '--format=csv,noheader'], text=True)
    bus = dict(r.split(', ') for r in rows.strip().splitlines())[str(index)]
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_bus_id', '--format=csv,noheader'], text=True)
    others = [r for r in apps.strip().splitlines() if r and r.split(', ')[1] == bus and int(r.split(', ')[0]) != os.getpid()]
    if others and idle: raise RuntimeError(f'GPU {index} shared with {others}')


def build(vit, spec, cfg):
    """nn.Module of a config: the method ('pd') or a baseline (baselines.build)."""
    if cfg['kind'] == 'dense': return vit
    if cfg['kind'] == 'pdm':
        if cfg.get('merge', {}).get('kernel') == 'triton_norm':
            from leaf.model.mlerp_vit import MlerpConfig, MlerpViT
            return MlerpViT(vit, MlerpConfig(**{k: v for k, v in cfg.items() if k != 'kind'}), pool=spec.pool)
        from leaf.model.merge_variant_vit import MergeVariantConfig, MergeVariantViT
        return MergeVariantViT(vit, MergeVariantConfig(**{k: v for k, v in cfg.items() if k != 'kind'}), pool=spec.pool)
    if cfg['kind'] == 'pd':
        from leaf.model.reduced_vit import ReductionConfig, ReducedViT
        return ReducedViT(vit, ReductionConfig(**{k: v for k, v in cfg.items() if k != 'kind'}), pool=spec.pool)
    if cfg['kind'] == 'ipd':     # experimental I roles: archived in the research repository, not part of LEAF
        raise ValueError("kind 'ipd' (I-term roles) is not part of this repository")
    from baselines import build as build_baseline
    return build_baseline(vit, spec, cfg)


class Runner:
    def __init__(self, backbone, root=None, limit=0, batch=BATCH):
        # A limited (smoke) run never shares the cache of full runs: same config key, different image set.
        self.name = backbone; self.root = Path(root or RESULTS / (backbone if not limit else f'{backbone}_smoke{limit}'))
        self.limit = limit; self.batch = batch
        self._vit = None; self._data = {}

    @property
    def vit(self):
        if self._vit is None:
            torch.set_num_threads(4); torch.manual_seed(151)
            torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
            self._vit, self.spec = load(self.name)
        return self._vit

    def inference_model(self, cfg):
        """What evaluation.latency.inference_copy(build(...)) computes, without copying the backbone per config:
        the wrapper is built on the fp32 backbone (D matrices come from fp32 weights), then pointed at one shared
        copy whose Linear/Conv weights are pre-cast to BF16."""
        from leaf.latency.timing import INFERENCE_DTYPE, inference_copy
        if getattr(self, '_cast', None) is None: self._cast = inference_copy(self.vit)
        model = build(self.vit, self.spec, cfg)
        if model is self.vit: return self._cast
        model.vit = self._cast
        for mod in model.modules():
            if mod is not self._cast and hasattr(mod, 'prepare_inference'): mod.prepare_inference(INFERENCE_DTYPE)
        return model.eval()

    def data(self, split):
        if split not in self._data: self._data[split] = split_arrays(split, self.vit, self.spec, self.limit)
        return self._data[split]

    def path(self, cfg, split):
        return self.root / split / 'runs' / f'{config_hash(cfg)}'

    def cached(self, cfg, split):
        p = self.path(cfg, split).with_suffix('.npz')
        return dict(np.load(p)) if p.exists() else None

    @torch.no_grad()
    def _evaluate(self, cfg, split, prefix=None):
        with gpu_work():
            images, labels, teacher = self.data(split)
            model = self.inference_model(cfg)
        kls, correct, preds, seen = [], [], [], 0
        start = 0
        if prefix is not None:              # resume after the shared dense prefix (leaf.evaluation.prefix)
            start = prefix.block
            if prefix.split != split or min(getattr(model, 'stations', [-1]) or [-1]) < start or model.pool == 'dense':
                raise ValueError('the prefix must end at or before the first station, on the same split, without a dense head')
            batches = prefix.batches(labels)
        else:
            batches = iter(iterate(images, labels, self.batch, 'cuda', self.spec.mean, self.spec.std))
        while True:
            with gpu_work():
                try: x, y = next(batches)
                except StopIteration: break
                with torch.autocast('cuda', dtype=torch.bfloat16): logits = (model(x, start=start) if start else model(x)).float()
                if not bool(torch.isfinite(logits).all()): raise RuntimeError('nonfinite logits')
                pred = logits.argmax(-1); preds.append(pred.cpu().numpy()); correct.append(pred.eq(y).cpu().numpy())
                kls.append(per_image_kl(logits, teacher[seen:seen + len(y)].cuda()).cpu().numpy()); seen += len(y)
        with gpu_work():
            del model; gc.collect(); torch.cuda.empty_cache()
        if seen != len(labels): raise RuntimeError('incomplete evaluation')
        return dict(correct=np.concatenate(correct), pred=np.concatenate(preds).astype(np.int16),
                    kl=np.concatenate(kls).astype(np.float32))

    def run(self, cfg, split, meta=None, prefix=None):
        """Per-image arrays (cached); None for an infeasible config (non-finite BF16 logits)."""
        p = self.path(cfg, split)
        if p.with_suffix('.json').exists():
            row = load_json(p.with_suffix('.json')); assert row['config'] == cfg, p
            return None if row.get('feasible') is False else dict(np.load(p.with_suffix('.npz')))
        start = time.monotonic()
        try:
            arr = self._evaluate(cfg, split, prefix) if prefix is not None else self._evaluate(cfg, split)
        except RuntimeError as exc:
            if 'nonfinite' not in str(exc): raise
            gc.collect(); torch.cuda.empty_cache()
            save(p.with_suffix('.json'), dict(id=config_hash(cfg), config=cfg, feasible=False, reason=str(exc), **(meta or {})))
            return None
        p.parent.mkdir(parents=True, exist_ok=True); np.savez(p.with_suffix('.npz'), **arr)
        row = dict(id=config_hash(cfg), config=cfg, **metrics(arr),
                   batch_size=self.batch, seconds=time.monotonic() - start, gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), **(meta or {}))
        save(p.with_suffix('.json'), row)
        shown = {k: round(row[k], 3 if k != 'kl' else 5) for k in ('top1', 'miou', 'agree', 'kl') if k in row}
        print(json.dumps(dict(split=split, id=row['id'], **shown, s=round(row['seconds'], 1))), file=sys.stderr, flush=True)
        return arr

    def kl(self, cfg, split='half'):
        arr = self.run(cfg, split)
        return None if arr is None else arr['kl'].astype(np.float64)
