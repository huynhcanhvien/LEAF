"""leaf probe --prefix-cache: probes resumed from the shared dense prefix (leaf.evaluation.prefix) give the same
per-image KL, predictions and correctness, bit for bit, as the full forward. DeiT-S, 640 probe images. Needs a GPU.

    python tests/test_prefix_cache.py
"""
import tempfile

import numpy as np

from common import check, finish

from leaf.api import LeafRunner
from leaf.backbones import SPECS
from leaf.cli import forecast_config, probe_jobs
from leaf.evaluation.prefix import PrefixCache

BB, P, N = 'deit_s', 'schedules/leaf/planning/deit_s', 640


def runner(root):
    r = LeafRunner(BB, root=root, batch=32); r.limit = N
    return r


def main():
    import os
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    spec = SPECS[BB]
    jobs = list(probe_jobs(spec, forecast_config(f'{P}/groups.json', 16, 32), [0., .5, 1.], [0., .5, 1.], [24, 120, 194]))
    picked = [j for j in jobs if j['block'] in (1, 5, 10, 11)][::5]
    with tempfile.TemporaryDirectory() as tmp:
        full, resumed = runner(f'{tmp}/full'), runner(f'{tmp}/prefix')
        prefix = PrefixCache(resumed, 'half')
        same = 0
        for j in picked:
            a = full.run(j['config'], 'half')
            b = resumed.run(j['config'], 'half', prefix=prefix.at(j['block']))
            same += all(np.array_equal(a[name], b[name]) for name in ('kl', 'pred', 'correct'))
        check(f'prefix cache == full forward ({len(picked)} probes at blocks 1, 5, 10, 11; theta and alpha mixed)',
              same == len(picked), f'{same}/{len(picked)} bit-identical')
        try:
            prefix.at(11); resumed._evaluate(next(j for j in jobs if j['block'] == 5)['config'], 'half', prefix)
            check('a prefix past the station is refused', False)
        except ValueError:
            check('a prefix past the station is refused', True)
    finish()


if __name__ == '__main__':
    main()
