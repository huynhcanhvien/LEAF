"""Re-run off-the-shelf baselines with the vendored official algorithms (baselines/third_party/*/SOURCE).

Config kinds: {'kind': 'tome', 'r': int | [r, inflection]} or {'kind': 'tome', 'r_blocks': [r per block]},
{'kind': 'pitome', 'ratio': float},
{'kind': 'evit', 'keep_rate': float},
{'kind': 'mctf', 'r': int}, {'kind': 'ppt', 'r': int, 'threshold': float}. DiffRate is the method's own config with k = {} (planner.configs.diffrate_config).
"""


def build(vit, spec, cfg):
    from .methods import EViT, MCTF, PPT, PiToMe, ToMe
    kind = cfg['kind']
    if kind == 'tome' and 'r_blocks' in cfg: return ToMe(vit, spec, list(cfg['r_blocks']))    # r per block
    if kind == 'tome': return ToMe(vit, spec, tuple(cfg['r']) if isinstance(cfg['r'], list) else cfg['r'])
    if kind == 'pitome': return PiToMe(vit, spec, cfg['ratio'])
    if kind == 'evit': return EViT(vit, spec, cfg['keep_rate'])
    if kind == 'mctf': return MCTF(vit, spec, cfg['r'])
    if kind == 'ppt': return PPT(vit, spec, cfg['r'], cfg.get('threshold', 7e-5))
    raise ValueError(f'unknown baseline {kind}')
