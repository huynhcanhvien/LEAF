"""Offline protocol checks: rule integrity, original caches, selection and no V2."""
from common import check, finish
import copy
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np

from leaf.reproduction.runners.merge_variant_runner import assert_ready, bind_baselines, cached_arrays, design, read_rule, seal, summarize
from leaf.evaluation.runner import config_hash, save


def rejects(fn):
    try: fn()
    except (ValueError, RuntimeError): return True
    return False


def main():
    rule = design()
    check('fixed 10 schedules / 12 specified variants', len(rule['arms']) == 10 and len(rule['modes']) == 12)
    check('V2 absent from allowed splits', set(rule['splits']) == {'half', 'val'})
    check('MAE-L opt-in only', len(design(include_mae_l=True)['arms']) == 11)
    check('original pd config retained', all(a['baseline']['kind'] == 'pd' and 'merge' not in a['baseline'] for a in rule['arms']))
    check('pending N2 prevents evaluation', rejects(lambda: assert_ready(rule)))
    deferred = design(exclude_n2=True)
    assert_ready(deferred)
    check('N2 deferral permits exactly nine finalized modes', len(deferred['modes']) == 9 and
          set(deferred['deferred_modes']) == {'N2', 'N1+N2', 'N2+N5'} and
          all(set(a['variants']) == set(deferred['modes']) for a in deferred['arms']))
    with tempfile.TemporaryDirectory() as temp, patch('leaf.reproduction.runners.merge_variant_runner.N2_FORMULA_STATUS', 'final'):
        sealed = seal(temp)
        check('rule seal is idempotent and SHA verified', seal(temp) == sealed and read_rule(temp) == sealed)
        check('cannot change optional backbone after seal', rejects(lambda: seal(temp, True)))
        tampered = copy.deepcopy(sealed); tampered['rule']['thresholds']['weak_gain'] = -1
        save(Path(temp) / 'stage1_rule.json', tampered)
        check('tampered rule rejected', rejects(lambda: read_rule(temp)))
        cfg = rule['arms'][0]['baseline']
        p = Path(temp) / 'deit_b' / 'val' / 'runs' / config_hash(cfg)
        save(p.with_suffix('.json'), dict(config=cfg, n=3, batch_size=128))
        np.savez(p.with_suffix('.npz'), correct=np.ones(3, dtype=bool), pred=np.zeros(3), kl=np.zeros(3))
        before = p.with_suffix('.npz').read_bytes()
        arr = cached_arrays(temp, 'deit_b', cfg, 'val', 3)
        check('cache reads exact arrays without writes', arr['correct'].all() and before == p.with_suffix('.npz').read_bytes())
        check('wrong cached batch rejected', rejects(lambda: cached_arrays(temp, 'deit_b', cfg, 'val', 3, batch=64)))
        check('wrong image count rejected', rejects(lambda: cached_arrays(temp, 'deit_b', cfg, 'val', 4)))
        check('V2 rejected', rejects(lambda: cached_arrays(temp, 'deit_b', cfg, 'v2', 3)))
        check('wrong experiment SHA rejected', rejects(lambda: cached_arrays(temp, 'deit_b', cfg, 'val', 3, rule_sha='wrong')))
        tiny = dict(arms=[dict(backbone='deit_b', name='synthetic', baseline=cfg)], splits={'val': 3})
        # Use a paired half fixture to verify nested paper-final cache discovery.
        half = Path(temp) / 'deit_b' / 'half' / 'runs' / config_hash(cfg)
        save(half.with_suffix('.json'), dict(config=cfg, n=3, batch_size=128))
        np.savez(half.with_suffix('.npz'), **arr)
        tiny['splits']['half'] = 3
        bound = bind_baselines(tiny, temp)
        check('historical cache discovery binds actual batch', bound['arms'][0]['evaluation_batch'] == 128)
    rows, costs = [], []
    for arm in rule['arms']:
        n = 197
        blocks = []
        for block, (p, m) in enumerate(zip(arm['baseline']['prune_keep'], arm['baseline']['merge_keep'])):
            p, m = min(n, p), min(n, p, m)
            if p > m: blocks.append(block)
            n = m
        for i, mode in enumerate(rule['modes']):
            # M1 and M2 pass; M3 fails strong floor. Others have no gain.
            delta = .2 if i < 2 else (1 if i == 2 and arm['group'] == 'weak' else -.1 if i == 2 else 0.)
            rows.append(dict(backbone=arm['backbone'], name=arm['name'], group=arm['group'], mode=mode, delta=delta, p=1.))
            for block in blocks:
                costs.append(dict(backbone=arm['backbone'], name=arm['name'], block=block, mode=mode, extra_ms=.01))
    result = summarize(rule, copy.deepcopy(rows), costs)
    check('selection thresholds / deterministic top2', result['selected'] == ['M1', 'M2'])
    bad = copy.deepcopy(rows)
    bad[0].update(delta=-.1, p=.00001)
    check('significant degradation excludes variant', 'M1' not in summarize(rule, bad, costs)['selected'])
    expensive = copy.deepcopy(costs)
    for c in expensive:
        if c['mode'] == 'M1': c['extra_ms'] = .051
    check('merge cost threshold enforced', 'M1' not in summarize(rule, copy.deepcopy(rows), expensive)['selected'])
    check('incomplete results rejected', rejects(lambda: summarize(rule, rows[:-1], costs)))
    check('duplicate results rejected', rejects(lambda: summarize(rule, rows + rows[:1], costs)))
    check('incomplete station timing rejected', rejects(lambda: summarize(rule, copy.deepcopy(rows), costs[1:])))
    for r in rows: r['delta'] = 0
    check('no winners -> stop', summarize(rule, rows, costs)['decision'] == 'none_passed_stop')


if __name__ == '__main__':
    main(); finish()
