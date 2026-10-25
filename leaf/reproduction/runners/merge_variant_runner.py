"""CPU-only stage-1 design, immutable rule, cache validation and paired statistics.

No V2 or stage-2 entry point. Historical M0 files are opened read-only and never
passed to Runner.run. New arms live exclusively under the campaign directory.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from leaf.evaluation.data import holm, mcnemar
from leaf.evaluation.runner import Runner, config_hash, load_json, save
from leaf.model.merge_variants import MODES

ROOT = Path('results/merge_variants')
# User decision 2026-10-15: finalize the scalar/group forecast formula before evaluation.
N2_FORMULA_STATUS = 'pending'
SELECTION = (
    ('weak', 'deit_b', 'latency/DR11.5'), ('weak', 'deit_b', 'latency/DR12.5'),
    ('weak', 'mae_b', 'latency/DR10.4'), ('weak', 'mae_b', 'latency/DR11.5'),
    ('weak', 'augreg_b', 'latency/dense0.62'), ('weak', 'deit_s', 'latency/DR3.1'),
    ('strong', 'deit_b', 'latency/DR8.7'), ('strong', 'mae_b', 'latency/DR8.7'),
    ('strong', 'deit_t', 'latency/DR0.6'), ('strong', 'augreg_s', 'latency/dense0.62'),
)
OPTIONAL = ('strong', 'mae_l', 'latency/DR31.0')


def sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def has_merge(cfg):
    n = 197
    for p, m in zip(cfg['prune_keep'], cfg['merge_keep']):
        p, m = min(n, p), min(n, p, m)
        if p > m:
            return True
        n = m
    return False


def design(schedule_dir=Path('schedules/kappa_form/leaf_m0'), include_mae_l=False, exclude_n2=False):
    modes = [mode for mode in MODES[1:] if not (exclude_n2 and 'N2' in mode)]
    arms = []
    for group, backbone, name in SELECTION + ((OPTIONAL,) if include_mae_l else ()):
        source = Path(schedule_dir) / f'{backbone}.json'
        doc = load_json(source)
        cfg = doc['schedules'][name]['config']
        if cfg['kind'] != 'pd':
            raise ValueError('only original locked pd schedules are allowed')
        arms.append(dict(group=group, backbone=backbone, name=name, baseline=cfg,
                         timm_id=doc['timm_id'],
                         source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                         has_merge=has_merge(cfg),
                         variants={mode: dict(cfg, kind='pdm', merge={'mode': mode}) for mode in modes}))
    return dict(version=1, arms=arms, modes=modes,
                deferred_modes=[mode for mode in MODES[1:] if mode not in modes],
                splits={'half': 5000, 'val': 50000}, evaluation_batch=128,
                thresholds=dict(weak_gain=.10, strong_floor=-.05, strong_gain=.15,
                                weak_floor=-.05, alpha=.05, max_extra_ms=.05, max_selected=2),
                holm_family=['backbone', 'mode'], ranking='mean_weak_delta + mean_strong_delta',
                conventions=dict(N2_status=N2_FORMULA_STATUS,
                                 N2='DRAFT ONLY: head-count-weighted mean group logit; M1 size when kappa=0',
                                 N4='size-weighted channel mean + size-weighted centered norm',
                                 N6='cost to destinations outside bottom min(2*delta,p-2) pool; stable source selection',
                                 N5='stable top2; nonpositive cosines fall back to top1',
                                 M3='unweighted mean + maximum norm, M0 size',
                                 M1s='omitted by user: no definition'),
                disclosure='Merge operator selected on ImageNet val; V2 reserved for final pipeline.')


def bind_baselines(rule, baseline_root):
    """Bind exact historical cache locations/batches, without evaluating any model."""
    for arm in rule['arms']:
        refs = {}
        for split, n in rule['splits'].items():
            candidates = list((Path(baseline_root) / arm['backbone']).glob(
                f'**/{split}/runs/{config_hash(arm["baseline"])}.json'))
            valid = []
            for path in candidates:
                meta = load_json(path)
                if meta.get('config') == arm['baseline'] and meta.get('n') == n:
                    valid.append((path, meta['batch_size']))
            if len(valid) != 1:
                raise RuntimeError(f'need one historical {split} cache: {arm["backbone"]}/{arm["name"]}; found {len(valid)}')
            path, batch = valid[0]
            cache_root = path.parents[3]  # parent of backbone/ or bsXX/
            cache_name = path.parents[2].name
            cached_arrays(cache_root, cache_name, arm['baseline'], split, n, batch)
            refs[split] = dict(root=str(cache_root.resolve()), name=cache_name, batch=batch)
        if refs['half']['batch'] != refs['val']['batch']:
            raise RuntimeError('half/val historical batches differ')
        arm['baseline_cache'] = refs
        arm['evaluation_batch'] = refs['val']['batch']
    rule.pop('evaluation_batch', None)
    return rule


class MergeVariantRunner(Runner):
    """Read existing, batch-matched dense teachers; never generate historical data."""
    def __init__(self, backbone, root, batch, data_cache, teacher_cache, timm_id=None):
        super().__init__(backbone, root=root, batch=batch)
        self.data_cache, self.teacher_cache = Path(data_cache), Path(teacher_cache)
        self.locked_timm_id = timm_id

    @property
    def vit(self):
        from leaf.backbones import SPECS
        if self.locked_timm_id is None or self.locked_timm_id == SPECS[self.name].timm_id:
            return super().vit
        # Match the actual paper-final checkpoint without changing any legacy Spec.
        if self._vit is None:
            from dataclasses import replace
            import timm
            import torch
            torch.set_num_threads(4); torch.manual_seed(151)
            torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
            self.spec = replace(SPECS[self.name], timm_id=self.locked_timm_id)
            self._vit = timm.create_model(self.locked_timm_id, pretrained=True).cuda().eval().requires_grad_(False)
            if (len(self._vit.blocks), self._vit.embed_dim, self._vit.blocks[0].attn.num_heads) != (self.spec.depth, self.spec.dim, self.spec.heads):
                raise RuntimeError('locked checkpoint does not match backbone dimensions')
        return self._vit

    def data(self, split):
        import torch
        from leaf.evaluation.data import SUBSETS, load_split
        if split not in ('half', 'val'):
            raise ValueError('stage 1 forbids V2')
        if split not in self._data:
            base, sl = SUBSETS.get(split, (split, slice(None)))
            images, labels = load_split(base, self.data_cache)
            suffix = '' if self.batch == 128 else f'_bs{self.batch}'
            path = self.teacher_cache / f'{base}_{self.name}{suffix}_logits.npy'
            teacher = np.load(path, mmap_mode='r')
            if teacher.shape[0] != len(labels):
                raise RuntimeError('teacher/image count mismatch')
            self._data[split] = images[sl], labels[sl], torch.from_numpy(np.array(teacher[sl], dtype=np.float32))
        return self._data[split]


def assert_ready(rule):
    if any('N2' in mode for mode in rule['modes']) and rule['conventions']['N2_status'] != 'final':
        raise RuntimeError('N2 formula is pending user decision; stage-1 seal/evaluation is disabled')


def seal(root=ROOT, include_mae_l=False, baseline_root=None, exclude_n2=False):
    root = Path(root)
    rule = design(include_mae_l=include_mae_l, exclude_n2=exclude_n2)
    assert_ready(rule)
    if baseline_root is not None:
        bind_baselines(rule, baseline_root)
    # Bind the implementation as well as configs: changed code cannot reuse old results.
    files = ['leaf/model/merge_variants.py', 'leaf/model/merge_variant_vit.py', 'leaf/reproduction/runners/merge_variant_runner.py',
             'leaf/reproduction/experiments/merge_variant_screening.py', 'leaf/backbones.py', 'leaf/evaluation/runner.py',
             'leaf/model/token_ops.py', 'leaf/model/reduced_vit.py', 'leaf/forecast/attention_forecast.py',
             'leaf/forecast/kernels/kappa_score.py', 'leaf/model/kernels/mean_merge.py', 'leaf/evaluation/data.py',
             'leaf/latency/timing.py', 'leaf/evaluation/determinism.py']
    rule['implementation'] = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files}
    sealed = dict(sha256=sha(rule), rule=rule)
    path = root / 'stage1_rule.json'
    if path.exists():
        if load_json(path) != sealed:
            raise RuntimeError('rule/code changed: use a new campaign directory; never overwrite a sealed rule')
    else:
        save(path, sealed)
    return sealed


def read_rule(root=ROOT):
    sealed = load_json(Path(root) / 'stage1_rule.json')
    if sha(sealed['rule']) != sealed['sha256']:
        raise RuntimeError('stage1 rule SHA mismatch')
    for p, expected in sealed['rule']['implementation'].items():
        if hashlib.sha256(Path(p).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'code changed after rule seal: {p}')
    return sealed


def cached_arrays(root, backbone, cfg, split, expected, batch=128, rule_sha=None):
    """Require full same-batch cache with exact config and finite, aligned arrays."""
    if split not in ('half', 'val'):
        raise ValueError('stage 1 never reads V2')
    path = Path(root) / backbone / split / 'runs' / config_hash(cfg)
    meta = load_json(path.with_suffix('.json'))
    if rule_sha is not None and meta.get('stage1_rule_sha') != rule_sha:
        raise RuntimeError(f'cache rule SHA mismatch: {path}')
    if meta.get('config') != cfg or meta.get('batch_size') != batch or meta.get('n') != expected:
        raise RuntimeError(f'incompatible cache metadata: {path}')
    with np.load(path.with_suffix('.npz')) as z:
        arr = {k: z[k] for k in ('correct', 'pred', 'kl')}
    if any(v.shape != (expected,) for v in arr.values()) or not np.isfinite(arr['kl']).all():
        raise RuntimeError(f'invalid cache arrays: {path}')
    return arr


def summarize(rule, rows, costs):
    """Fail closed on incomplete/invalid evidence. Costs are paired per station."""
    expected = {(a['backbone'], a['name'], m) for a in rule['arms'] for m in rule['modes']}
    identities = [(r['backbone'], r['name'], r['mode']) for r in rows]
    if len(identities) != len(set(identities)) or set(identities) != expected:
        raise RuntimeError('complete, unique results required before selection')
    t = rule['thresholds']
    for backbone, mode in {(r['backbone'], r['mode']) for r in rows}:
        holm([r for r in rows if r['backbone'] == backbone and r['mode'] == mode])
    summaries = []
    for mode in rule['modes']:
        group = [r for r in rows if r['mode'] == mode]
        weak = float(np.mean([r['delta'] for r in group if r['group'] == 'weak']))
        strong = float(np.mean([r['delta'] for r in group if r['group'] == 'strong']))
        gain = ((weak >= t['weak_gain'] and strong >= t['strong_floor']) or
                (strong >= t['strong_gain'] and weak >= t['weak_floor']))
        harmful = any(r['delta'] < 0 and r['holm'] < t['alpha'] for r in group)
        relevant = set()
        for arm in rule['arms']:
            n = 197
            for block, (p, m) in enumerate(zip(arm['baseline']['prune_keep'], arm['baseline']['merge_keep'])):
                p, m = min(n, p), min(n, p, m)
                if p > m:
                    relevant.add((arm['backbone'], arm['name'], block))
                n = m
        measured = [c for c in costs if c['mode'] == mode]
        covered = {(c['backbone'], c['name'], c['block']) for c in measured}
        if relevant != covered or any(not np.isfinite(c['extra_ms']) for c in measured):
            raise RuntimeError(f'incomplete merge cost evidence for {mode}')
        cost = max((c['extra_ms'] for c in measured), default=0.)
        passed = bool(gain and not harmful and cost <= t['max_extra_ms'])
        summaries.append(dict(mode=mode, weak=weak, strong=strong, score=weak + strong,
                              harmful=harmful, max_extra_ms=cost, passed=passed))
    winners = sorted((r for r in summaries if r['passed']), key=lambda r: (-r['score'], r['mode']))[:t['max_selected']]
    return dict(rows=rows, variants=summaries, selected=[r['mode'] for r in winners],
                decision='stage2_requires_user_approval' if winners else 'none_passed_stop')


def paired_row(arm, mode, base, arr, calib):
    if len(base['correct']) != len(arr['correct']):
        raise RuntimeError('unaligned paired results')
    p, wins, losses = mcnemar(arr['correct'].astype(bool), base['correct'].astype(bool))
    return dict(backbone=arm['backbone'], name=arm['name'], group=arm['group'], mode=mode,
                delta=100 * float(arr['correct'].mean() - base['correct'].mean()), p=p,
                wins=wins, losses=losses, val_kl=float(arr['kl'].mean()), calib_kl=float(calib['kl'].mean()),
                equal=not arm['has_merge'])
