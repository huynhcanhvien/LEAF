"""Separate LaPS-M3/N4 planners. Every merging station uses the chosen operator.

Pure-prune probe configs stay exactly pd for read-only paper-final cache reuse.
DP, multiplier sweep and eta projection are planner.merge_final (unchanged copy
of the paper-final source); only measured curves and merge-station latency differ.
"""
from pathlib import Path

from leaf.evaluation.runner import load_json
from leaf.planner.configs import ForecastConfig, config_from_state
from leaf.planner.surrogate import ProbeCurve
from leaf.latency.profile_model import ProfileLatencyModel
from leaf.planner.probes import Planner, probe_jobs, probe_state
from leaf.planner.schedule_caps import canonical


class NormLat(ProfileLatencyModel):
    def __init__(self, model, spec, extra_ms=0.):
        super().__init__(model, spec, 32)
        self.norm_extra = float(extra_ms)

    def station(self, n_in, p, m, active_d=False, groups=16):
        return super().station(n_in, p, m, active_d, groups) + (self.norm_extra if min(n_in, p) > min(n_in, p, m) else 0.)

    def __call__(self, p, m, nd=None, ks=None):
        value = super().__call__(p, m, nd, ks)
        if not self.extra:
            n = self.spec.n_tokens
            for pp, mm in zip(p, m):
                if min(n, pp) > min(n, pp, mm): value += self.norm_extra
                n = min(n, pp, mm)
        return value


def norm_config(state, spec, dcfg, mode):
    cfg = config_from_state(state, spec, dcfg)
    if mode == 'M0': return cfg
    if mode not in ('M3', 'N4'): raise ValueError('one fixed operator per method')
    return dict(cfg, kind='pdm', merge=dict(mode=mode, kernel='triton_norm'))


class NormPlanner(Planner):
    def __init__(self, runner, spec, lat, curves, dcfg, mode):
        self.mode = mode
        super().__init__(runner, spec, lat, curves, dcfg)

    def cfg(self, st):
        return norm_config(st, self.spec, self.dcfg, self.mode)


def curves_from_cache(original, runner, spec, dcfg, mode):
    """Require ALL probes; never silently plan from a partially filled surface."""
    rows={}
    for l,n,theta,k in probe_jobs(spec):
        state=probe_state(l,n,theta,k)
        config=config_from_state(state,spec,dcfg) if theta==0 or mode=='M0' else norm_config(state,spec,dcfg,mode)
        arr=(original if theta==0 or mode=='M0' else runner).cached(config,'half')
        if arr is None: raise RuntimeError(f'Missing probe {mode}/{l}/{n}/{theta}/{k}')
        rows.setdefault((theta,k),{}).setdefault(l,{})[spec.n_tokens-n]=float(arr['kl'].mean())
    return {key:ProbeCurve(value,spec.n_tokens) for key,value in rows.items()}
