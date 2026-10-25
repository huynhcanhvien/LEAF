"""Evaluation harness for 'pda' configs (leaf/model/alpha_score.py): the q1_main_v2 ExtraRunner (same data, teachers,
BF16 batch-32 protocol) plus the alpha builder. Additive; the campaign runners are unchanged."""
from pathlib import Path

import torch

from leaf.model.alpha_score import build_alpha


def build_any(vit, spec, cfg):
    """Campaign build (pd/pdm/extra_pdm/baselines) with 'pda' support."""
    import leaf.reproduction.experiments.horizon_norm_seed_ablation as extra
    return build_alpha(vit, spec, cfg, extra.build) if cfg['kind'] == 'pda' else extra.build(vit, spec, cfg)


def alpha_runner(bb, root, design=None, reuse_roots=()):
    """ExtraRunner rooted at root; cache reuse only from the given roots (exact config, batch and n)."""
    import leaf.reproduction.experiments.horizon_norm_seed_ablation as extra
    import leaf.reproduction.experiments.main_comparison as main
    from leaf.latency.timing import inference_copy

    class AlphaRunner(extra.ExtraRunner):
        def inference_model(self, cfg):
            if cfg['kind'] != 'pda': return super().inference_model(cfg)
            _ = self.vit
            if getattr(self, '_cast', None) is None: self._cast = inference_copy(self.vit)
            model = build_any(self.vit, self.spec, cfg)
            model.vit = self._cast
            for mod in model.modules():
                if mod is not self._cast and hasattr(mod, 'prepare_inference'): mod.prepare_inference(torch.bfloat16)
            return model.eval()

    r = AlphaRunner(bb, Path(root), design or main.design())
    r.spec = __import__('leaf.backbones', fromlist=['SPECS']).SPECS[bb]
    r.teacher_cache = main.SOURCE.resolve() / 'teachers'
    r.reuse_roots = [Path(p) for p in reuse_roots]
    return r
