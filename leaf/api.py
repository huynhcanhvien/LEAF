"""High-level helpers: load released schedules, build the reduced model, evaluate and time it.

    from leaf.api import load_schedule, LeafRunner
    doc, entry = load_schedule('schedules/leaf/deit_b.json', 'latency/DR10.0')
    runner = LeafRunner('deit_b', root='runs/deit_b', timm_id=doc['timm_id'])
    arr = runner.run(entry['config'], 'val')          # per-image correct / pred / kl, cached under root
    model = runner.inference_model(entry['config'])  # BF16 module; leaf.latency.timing.measure_latency(model, 32)

Configs are JSON dicts with a 'kind':
  'pda'    the paper method: stations scored by (1 - alpha) z(log P) + alpha z(D) (leaf.model.alpha_score), on top of
           'base' = 'pdm' (M3 merge, kernel 'triton_norm') or 'pd' (M0 merge)
  'pd', 'pdm'                        the same wrappers with the kappa score z(log P) + k z(D)
  'dense'                            the backbone itself
  'tome', 'pitome', 'evit', 'mctf', 'ppt', 'tofu', 'rapid', 'onemore', 'tome_m3'   re-run baselines
The paper protocol evaluates in BF16 at batch 32 with dense teacher logits computed at the same batch size.
"""
from dataclasses import replace
import json
import os
from pathlib import Path

import numpy as np
import torch

from leaf.backbones import SPECS
from leaf.evaluation.data import SUBSETS, iterate, load_split
from leaf.evaluation.runner import Runner, build as _build_core

DEFAULT_BATCH = 32
SPLITS = ('half', 'odd', 'calib', 'calib_bal', 'confirm', 'val', 'v2')


def load_schedule(path, key=None):
    """(document, entry) of a released schedule file; key=None returns (document, None)."""
    doc = json.loads(Path(path).read_text())
    if key is None: return doc, None
    if key not in doc['schedules']:
        raise KeyError(f'{key!r} not in {path}; available: {", ".join(sorted(doc["schedules"]))}')
    return doc, doc['schedules'][key]


def build_model(vit, spec, cfg):
    """nn.Module of any config kind (see the module docstring); built on the fp32 backbone."""
    kind = cfg['kind']
    if kind == 'pda':
        from leaf.model.alpha_score import build_alpha
        return build_alpha(vit, spec, cfg, _build_extra)
    if kind == 'extra_pdm': return _build_extra(vit, spec, cfg)
    if kind in ('tofu', 'rapid', 'onemore'):
        from baselines.low_budget_posthoc import build_posthoc
        return build_posthoc(vit, spec, cfg)
    if kind == 'tome_m3':
        from baselines.tome_m3 import ToMeM3
        return ToMeM3(vit, spec, tuple(cfg['r']) if isinstance(cfg['r'], list) else cfg['r'])
    return _build_core(vit, spec, cfg)


def _build_extra(vit, spec, cfg):
    """_build_core, plus 'extra_pdm': the forecast-horizon and norm ablations of the paper (leaf/reproduction/ablation_models)."""
    if cfg['kind'] == 'extra_pdm':
        from leaf.reproduction.ablation_models.ablation_vit import ExtraViT
        return ExtraViT(vit, cfg, spec.pool)
    return _build_core(vit, spec, cfg)


def load_backbone(name, timm_id=None, device='cuda'):
    """(vit, spec) of a backbone; timm_id overrides the checkpoint of SPECS[name] (e.g. the one a schedule
    file was planned for). MAE backbones always load the official MAE fine-tuned weights."""
    from leaf.backbones import load, resolve
    name = resolve(name); spec = SPECS[name]
    if timm_id is None or timm_id == spec.timm_id or spec.weights != 'timm':
        return load(name, device)
    import timm
    spec = replace(spec, timm_id=timm_id)
    vit = timm.create_model(timm_id, pretrained=True).to(device).eval().requires_grad_(False)
    if (len(vit.blocks), vit.embed_dim, vit.blocks[0].attn.num_heads) != (spec.depth, spec.dim, spec.heads):
        raise RuntimeError(f'{timm_id} does not match the dimensions of {name}')
    return vit, spec


class LeafRunner(Runner):
    """leaf.evaluation.runner.Runner with the paper protocol: batch 32, explicit data / teacher directories,
    a fixed checkpoint, and every config kind of build_model.

    Results are cached per config and split under <root>/<split>/runs/<key>.{json,npz}. Dense teacher logits
    are computed once at the evaluation batch size and stored as <teacher_cache>/<split>_<backbone>_bs<batch>_logits.npy.
    """

    def __init__(self, backbone, root, batch=DEFAULT_BATCH, data_cache='data/cache', teacher_cache=None, timm_id=None):
        super().__init__(backbone, root=root, batch=batch)
        self.data_cache = Path(data_cache)
        self.teacher_cache = Path(teacher_cache) if teacher_cache else self.data_cache / 'teachers'
        self.timm_id = timm_id
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'    # CPU: tests and smoke runs only

    @property
    def vit(self):
        if self._vit is None:
            torch.set_num_threads(4); torch.manual_seed(151)
            torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
            self._vit, self.spec = load_backbone(self.name, self.timm_id, self.device)
        return self._vit

    @property
    def segmentation(self):
        _ = self.vit
        return self.spec.task == 'segmentation'

    def inference_model(self, cfg):
        """The module evaluated and timed: wrapper built on the fp32 backbone, pointed at a shared BF16 copy."""
        from leaf.latency.timing import INFERENCE_DTYPE, inference_copy
        vit = self.vit
        if getattr(self, '_cast', None) is None: self._cast = inference_copy(vit)
        if cfg['kind'] == 'dense': return self._cast
        model = build_model(vit, self.spec, cfg)
        model.vit = self._cast
        for mod in model.modules():
            if mod is not self._cast and hasattr(mod, 'prepare_inference'): mod.prepare_inference(INFERENCE_DTYPE)
        return model.eval()

    def teacher(self, base):
        """Dense logits of a cached split at the evaluation batch size (computed once, float16 on disk)."""
        path = self.teacher_cache / f'{base}_{self.name}_bs{self.batch}_logits.npy'
        if not path.exists():
            images, labels = load_split(base, self.data_cache)
            self.teacher_cache.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(f'.{os.getpid()}.tmp.npy')     # probe shards may build the same teacher at once
            vit, spec = self.vit, self.spec
            side = spec.img_size // spec.patch
            shape = (spec.num_classes, side, side) if self.segmentation else (spec.num_classes,)
            mm = np.lib.format.open_memmap(tmp, mode='w+', dtype=np.float16, shape=(len(labels), *shape))
            offset = 0
            with torch.no_grad(), torch.autocast(self.device, dtype=torch.bfloat16):
                for x, _ in iterate(images, labels, self.batch, self.device, spec.mean, spec.std):
                    mm[offset:offset + len(x)] = vit(x).float().half().cpu().numpy(); offset += len(x)
            mm.flush(); del mm; tmp.replace(path)
        return np.load(path, mmap_mode='r')

    def data(self, split):
        if self.segmentation:               # windows; the teacher stays a float16 memmap ([N, K, h, w])
            if split not in self._data:
                from leaf.evaluation.segmentation import base_split
                base = base_split(self.spec, split)
                images, labels = load_split(base, self.data_cache)
                teacher = self.teacher(base)
                if len(teacher) != len(labels): raise RuntimeError(f'teacher / image count mismatch for {base}')
                n = self.limit or None
                self._data[split] = (images[:n], labels[:n], teacher[:n])
            return self._data[split]
        if split not in SPLITS: raise ValueError(f'unknown split {split!r}; one of {SPLITS}')
        if split not in self._data:
            _ = self.vit
            base, sl = SUBSETS.get(split, (split, slice(None)))
            images, labels = load_split(base, self.data_cache)
            teacher = self.teacher(base)
            if len(teacher) != len(labels): raise RuntimeError(f'teacher / image count mismatch for {base}')
            n = self.limit or None
            self._data[split] = (images[sl][:n], labels[sl][:n],
                                 torch.from_numpy(np.array(teacher[sl][:n], dtype=np.float32)))
        return self._data[split]

    def _evaluate(self, cfg, split):
        if self.segmentation:
            from leaf.evaluation.segmentation import evaluate
            return evaluate(self, cfg, split)
        return super()._evaluate(cfg, split)


def stations(cfg, spec):
    """Stations of a 'pda' / 'pd' / 'pdm' config: block, tokens after it, merge share theta, forecast weight alpha
    and the equivalent kappa = alpha / (1 - alpha) of the unnormalised score (None for alpha = 1)."""
    from leaf.model.alpha_score import alpha_to_kappa, kappa_to_alpha
    from leaf.planner.alpha import config_details_alpha
    out = []
    for r in config_details_alpha(cfg, spec)['token_path']:
        if r['removed'] <= 0: continue
        a = float(r['alpha']) if 'alpha' in r else kappa_to_alpha(r['kappa'])
        out.append(dict(block=r['block'], tokens_after=r['merge_keep'], theta=r['theta'], alpha=a,
                        kappa=None if a == 1 else alpha_to_kappa(a)))
    return out


def summary(arr):
    """Top-1 (%) and mean KL to the dense model of per-image arrays (segmentation: mIoU, see evaluation.segmentation)."""
    if 'agree' in arr:
        from leaf.evaluation.segmentation import summary as segmentation_summary
        return segmentation_summary(arr)
    return dict(n=int(len(arr['kl'])), top1=100 * float(np.mean(arr['correct'])), kl=float(np.mean(arr['kl'])))
