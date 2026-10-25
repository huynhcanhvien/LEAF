"""Alpha score (leaf/model/alpha_score.py): conversions, token order equal to the legacy k score, alpha = 0 / 1
limits, and end-to-end logits of a converted sealed schedule. CPU by default; ALPHA_GPU=1 adds the
Triton kernel and a CUDA end-to-end check (only on an idle card)."""
from common import check, finish
import json
import os

import torch

from leaf.model.token_ops import zscore
from leaf.model.alpha_score import alpha_to_kappa, alpha_to_kappa_config, build_alpha, config_to_alpha, kappa_to_alpha

PLANS = 'results/main_comparison/{}/plans.json'


def conversions():
    check('k -> alpha map', [kappa_to_alpha(k) for k in (0, 1, 2, 3)] == [0., .5, 2 / 3, .75])
    check('alpha -> k map', [alpha_to_kappa(a) for a in (0, .5, 2 / 3, .75)] == [0., 1., 2., 3.] and alpha_to_kappa(1) == float('inf'))
    cfg = dict(kind='pdm', prune_keep=[197, 150], merge_keep=[197, 120], k={'1': 3.}, groups={'1': [0, 1]}, rank=32,
               triton=True, fuse_merge=True, merge=dict(mode='M3', kernel='triton_norm'))
    a = config_to_alpha(cfg)
    check('config_to_alpha', a['kind'] == 'pda' and a['base'] == 'pdm' and a['alpha'] == {'1': .75} and 'k' not in a)
    back = alpha_to_kappa_config(a)
    check('alpha_to_kappa_config keeps everything but k', {k: v for k, v in back.items() if k != 'k'} == {k: v for k, v in cfg.items() if k != 'k'}
          and back['k'] == {'1': 1.})
    check('source config unchanged', cfg['k'] == {'1': 3.} and cfg['kind'] == 'pdm')


def order_equivalence():
    g = torch.Generator().manual_seed(0)
    raw_p = torch.rand(64, 197, generator=g).softmax(1); d = torch.randn(64, 197, generator=g)
    zp, zd = zscore(raw_p.log(), 1), zscore(d, 1)
    same = True
    for k in (1., 2., 3.):
        a = kappa_to_alpha(k)
        legacy = (zp + k * zd)[:, 1:].argsort(1, descending=True)
        alpha = ((1 - a) * zp + a * zd)[:, 1:].argsort(1, descending=True)
        same &= bool((legacy == alpha).all())
    check('alpha = k/(1+k) gives the legacy token order (random scores)', same)
    check('alpha = 0 is P only', torch.equal(((1 - 0.) * zp + 0. * zd).argsort(1), zp.argsort(1)))
    check('alpha = 1 ignores P', torch.equal((0. * zp + 1. * zd).argsort(1), zd.argsort(1)))


def end_to_end(device):
    from leaf.backbones import load
    from leaf.reproduction.experiments.horizon_norm_seed_ablation import build as campaign_build
    vit, spec = load('deit_s', device=device)
    jobs = json.load(open(PLANS.format('deit_s')))['jobs']
    name = next(n for n, r in jobs.items() if n.startswith('leaf_alpha4/') and r.get('config', {}).get('k'))
    cfg = jobs[name]['config']
    if device == 'cpu': cfg = dict(cfg, triton=False, fuse_merge=False)
    img = torch.randn(8, 3, 224, 224, generator=torch.Generator().manual_seed(1)).to(device)
    with torch.no_grad():
        legacy = campaign_build(vit, spec, cfg).eval()(img)
        model = build_alpha(vit, spec, config_to_alpha(cfg), campaign_build).eval()
        alpha = model(img)
    diff = (legacy - alpha).abs().max().item()
    check(f'{device}: converted {name} ({type(model).__name__}) logits match legacy', diff < 1e-3 and bool((legacy.argmax(1) == alpha.argmax(1)).all()),
          f'max |diff| {diff:.1e}, alpha {model.alpha}')


def kernel():
    from leaf.forecast.kernels.alpha_score import zscored_future_alpha
    from leaf.forecast.kernels.kappa_score import zscored_future
    g = torch.Generator(device='cuda').manual_seed(0)
    x = torch.randn(4, 197, 384, device='cuda', generator=g); size = torch.rand(4, 197, 1, device='cuda', generator=g) + .5
    u = torch.randn(4, 16, 384, device='cuda', generator=g); lc = torch.zeros(16, device='cuda').log_softmax(0)
    p = torch.rand(4, 197, device='cuda', generator=g).softmax(1)
    for k in (1., 3.):
        a = kappa_to_alpha(k)
        legacy = zscored_future(x, size, u, lc, 6, 1e-6, p, k); new = zscored_future_alpha(x, size, u, lc, 6, 1e-6, p, a)
        check(f'kernel: alpha={a:.3f} == legacy/(1+k)', torch.allclose(new, legacy / (1 + k), atol=1e-5), f'{(new - legacy / (1 + k)).abs().max():.1e}')
    d_only = zscored_future_alpha(x, size, u, lc, 6, 1e-6, p, 1.)
    check('kernel: alpha=1 == z(D)', torch.allclose(d_only, zscored_future(x, size, u, lc, 6, 1e-6, None, 1.), atol=1e-6))


def planner_equivalence(bb='deit_b', n_budgets=2):
    """Re-planning with alpha-relabelled probe curves reproduces the sealed leaf_alpha4 schedules (CPU DP)."""
    import leaf.reproduction.experiments.main_comparison as main
    from leaf.backbones import SPECS
    from leaf.planner.alpha import config_details_alpha, curves_to_alpha, plan_under_latency
    from leaf.reproduction.experiments.low_budget_and_factorial import probe_curves
    spec, d = SPECS[bb], main.dcfg(bb)
    cur = curves_to_alpha(probe_curves(bb, d, 'M3'))
    jobs = json.load(open(PLANS.format(bb)))['jobs']
    for b in main.design()['backbones'][bb]['budgets'][:n_budgets]:
        sealed = jobs[f"leaf_alpha4/{b['id']}"]['config']
        new = plan_under_latency(cur, spec, d, main.design()['models'][bb]['main'], b['latency_ms'], 'M3', True)['config']
        same = (new['prune_keep'] == sealed['prune_keep'] and new['merge_keep'] == sealed['merge_keep']
                and new['alpha'] == {l: kappa_to_alpha(k) for l, k in sealed['k'].items()} and new['groups'] == sealed['groups'])
        stations = [(r['block'], r['alpha']) for r in config_details_alpha(new, spec)['token_path'] if r['removed']]
        check(f"planner: alpha re-plan == sealed leaf_alpha4/{b['id']} ({bb})", same, f'stations (block, alpha) {stations}')


if __name__ == '__main__':
    conversions(); order_equivalence(); end_to_end('cpu'); planner_equivalence()
    if os.environ.get('ALPHA_GPU') == '1':
        kernel(); end_to_end('cuda')
    finish()
