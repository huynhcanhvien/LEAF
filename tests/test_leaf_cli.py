"""The public entry points reproduce the released paper results on the CPU (no GPU, no ImageNet needed):
`leaf plan` gives the released latency- and GMAC-planned schedules exactly (Lagrangian DP, SALT, alpha grid),
and `leaf fit-latency` gives the released SALT model exactly.

    python tests/test_leaf_cli.py [backbone ...]      (default: deit_t deit_b)
"""
import contextlib
import io
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import check, finish

from leaf.cli import main


def run(*argv):
    with contextlib.redirect_stdout(io.StringIO()):
        main(list(argv))


def test_plan(bb, tmp):
    P = f'schedules/leaf/planning/{bb}'
    released = json.load(open(f'schedules/leaf/{bb}.json'))['schedules']
    run('plan', '--backbone', bb, '--probes', f'{P}/probes.json', '--groups', f'{P}/groups.json', '--salt', f'{P}/salt.json',
        '--budgets', f'{P}/budgets.json', '--out', f'{tmp}/{bb}.json')
    got = json.load(open(f'{tmp}/{bb}.json'))['schedules']
    keys = [k for k in released if k.startswith('latency/')]
    same = [k for k in keys if got.get(k, {}).get('config') == released[k]['config']]
    check(f'{bb}: leaf plan reproduces the {len(keys)} released latency schedules', len(same) == len(keys), f'{len(same)}/{len(keys)}')
    gm = {k: v for k, v in released.items() if k.startswith('gmac/')}
    if gm:
        run('plan', '--backbone', bb, '--probes', f'{P}/probes.json', '--groups', f'{P}/groups.json', '--alphas', '0', '1/2', '2/3', '3/4',
            '--gmac', *[f"{v['budget']['value']:.12g}" for v in gm.values()], '--out', f'{tmp}/{bb}_gmac.json')
        got = list(json.load(open(f'{tmp}/{bb}_gmac.json'))['schedules'].values())
        same = sum(g.get('config') == v['config'] for g, v in zip(got, gm.values()))
        check(f'{bb}: leaf plan --gmac reproduces the {len(gm)} released GMAC schedules', same == len(gm), f'{same}/{len(gm)}')


def test_fit_salt(bb, tmp):
    P = f'schedules/leaf/planning/{bb}'
    run('fit-latency', '--backbone', bb, '--timings', f'{P}/salt_workloads.json', '--out', f'{tmp}/salt_{bb}.json')
    a, b = json.load(open(f'{tmp}/salt_{bb}.json')), json.load(open(f'{P}/salt.json'))
    # Exact on the machine that fitted it; other numpy / scipy / BLAS builds change the last bits of the NNLS solution.
    x, y = a['coef'] + [a['intercept']], b['coef'] + [b['intercept']]
    dev = max(abs(u - v) / max(abs(v), 1e-9) for u, v in zip(x, y))
    check(f'{bb}: leaf fit-latency reproduces the released SALT model (same ridge, coefficients within 1e-6)',
          a['alpha'] == b['alpha'] and len(x) == len(y) and dev < 1e-6, f'max relative deviation {dev:.1e}')


def main_():
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    bbs = sys.argv[1:] or ['deit_t', 'deit_b']
    with tempfile.TemporaryDirectory() as tmp:
        for bb in bbs:
            test_plan(bb, tmp); test_fit_salt(bb, tmp)
    finish()


if __name__ == '__main__':
    main_()
