"""Stations scored by the convex combination (1 - alpha_t)*z(log P) + alpha_t*z(D), alpha_t in [0, 1].

Additive: ReducedViT/MergeVariantViT/MlerpViT/ExtraViT and their 'k' configs are unchanged. A config of kind 'pda'
is a legacy config ('base' = 'pd' | 'pdm' | 'extra_pdm') whose 'k' is replaced by 'alpha'. The legacy
wrapper is built with a placeholder weight at every alpha > 0 station (which only creates the forecast
module and its group labels), and a subclass replaces score() by the alpha form.

Equivalence: k = alpha/(1 - alpha) ranks tokens identically (the station only sorts the score), so
legacy k in {0, 1, 2, 3} maps to alpha in {0, 1/2, 2/3, 3/4}; alpha = 1 ranks by D alone.
"""
from copy import deepcopy
from fractions import Fraction

from leaf.forecast.kernels.alpha_score import zscored_alpha

BASES = ('pd', 'pdm', 'extra_pdm')


def kappa_to_alpha(k):
    """Exact k/(1 + k) as a float (k >= 0)."""
    k = Fraction(float(k)).limit_denominator(10 ** 6)
    if k < 0: raise ValueError('k must be >= 0')
    return float(k / (1 + k))


def alpha_to_kappa(a):
    """alpha/(1 - alpha); infinite for alpha = 1."""
    a = float(a)
    if not 0 <= a <= 1: raise ValueError('alpha must lie in [0, 1]')
    return float('inf') if a == 1 else float(Fraction(a).limit_denominator(10 ** 6) / (1 - Fraction(a).limit_denominator(10 ** 6)))


def config_to_alpha(cfg):
    """Legacy 'pd'/'pdm'/'extra_pdm' config -> 'pda' config with the same token order."""
    if cfg['kind'] not in BASES: raise ValueError(f"no forecast weight in kind {cfg['kind']!r}")
    out = deepcopy(cfg)
    out['base'] = out.pop('kind')
    out['alpha'] = {str(l): kappa_to_alpha(v) for l, v in out.pop('k', {}).items() if float(v) > 0}
    return dict(kind='pda', **out)


def alpha_to_kappa_config(cfg):
    """'pda' config -> legacy config used only to build the modules (placeholder k = 1 at alpha > 0)."""
    if cfg['kind'] != 'pda': raise ValueError('expected kind pda')
    out = deepcopy(cfg)
    base, alpha = out.pop('base'), out.pop('alpha')
    out.pop('kind')
    if base not in BASES: raise ValueError(f'unknown base {base!r}')
    if any(not 0 <= float(a) <= 1 for a in alpha.values()): raise ValueError('alpha must lie in [0, 1]')
    out['k'] = {str(l): 1.0 for l, a in alpha.items() if float(a) > 0}
    return dict(kind=base, **out)


class AlphaScore:
    """Mixin: score() of the alpha form at forecast stations, the legacy P-only score elsewhere."""
    alpha = {}

    def score(self, l, x, size, raw_p):
        key = str(l)
        if key in self.future and self.alpha.get(key, 0.) > 0:
            return zscored_alpha(self.future[key], x, size, raw_p, self.alpha[key])
        return super().score(l, x, size, raw_p)


_CLASSES = {}


def build_alpha(vit, spec, cfg, legacy_build):
    """nn.Module of a 'pda' config. legacy_build(vit, spec, cfg) builds the legacy wrapper (e.g. the
    campaign's build function, which knows 'extra_pdm' and the M3 kernels)."""
    model = legacy_build(vit, spec, alpha_to_kappa_config(cfg))
    cls = type(model)
    if cls not in _CLASSES: _CLASSES[cls] = type('Alpha' + cls.__name__, (AlphaScore, cls), {})
    model.__class__ = _CLASSES[cls]
    model.alpha = {str(l): float(a) for l, a in cfg['alpha'].items() if float(a) > 0}
    if set(model.alpha) != set(model.future.keys()): raise ValueError('forecast modules do not match alpha stations')
    if getattr(model, 'ablation', {}).get('variant') in ('shuffle', 'shuffle_size', 'identity'):
        raise ValueError('D placebo variants use the legacy k score; not supported with alpha')
    return model
