"""The search commands on the CPU (no GPU, no ImageNet needed):
- `leaf.planner.budgets` gives the paper's budgets from the anchor latencies (all seven backbones);
- `leaf plan --budgets` reads the output format of `leaf budgets` and gives the released schedules;
- `leaf plan --operator M0` and the calibration-seed probe tables give the paper's ablation schedules;
- `leaf workloads` plans distinct timing workloads within their GMAC points;
- `leaf probe` keeps the probes of earlier grids in probes.json and appends new ones (cached runs, no GPU);
- `leaf search --dry-run` prints the steps for one GPU and for two;
- `leaf search` runs its steps as processes and writes schedules.json and search.json (cached probes, released SALT).

    python tests/test_search_cli.py
"""
import collections
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import check, finish

from leaf.backbones import SPECS
from leaf.reproduction.experiments.campaign_reports import BBS as PAPER_BBS
from leaf.cli import forecast_config, main, probe_jobs, load_probe_curves
from leaf.evaluation.runner import config_hash


def run(*argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):     # results and progress lines
        main(list(argv))
    return out.getvalue()


def test_budgets():
    from leaf.planner.budgets import anchor_budgets, dense_budgets
    for bb in PAPER_BBS:          # the paper's backbones (registered extra ones, e.g. clip_b16, have no release)
        ref = json.load(open(f'schedules/leaf/planning/{bb}/budgets.json'))
        anchors = {b['id']: dict(ms=b['latency_ms'] / .97, gmac=b['gmac']) for b in ref if b['anchor']}
        got = anchor_budgets(anchors) if anchors else dense_budgets(ref[0]['latency_ms'] / .5)
        err = max(abs(a['latency_ms'] - b['latency_ms']) for a, b in zip(got, ref))
        check(f'{bb}: budgets from the anchor latencies equal the paper budgets', [b['id'] for b in got] == [b['id'] for b in ref]
              and err < 1e-9, f'max |error| {err:.1e} ms')


def test_plan_budgets_file(tmp):
    bb, P = 'deit_t', 'schedules/leaf/planning/deit_t'
    wrapped = dict(backbone=bb, budgets=json.load(open(f'{P}/budgets.json')))
    Path(f'{tmp}/budgets.json').write_text(json.dumps(wrapped))
    run('plan', '--backbone', bb, '--probes', f'{P}/probes.json', '--groups', f'{P}/groups.json', '--salt', f'{P}/salt.json',
        '--budgets', f'{tmp}/budgets.json', '--out', f'{tmp}/plan.json')
    got = json.load(open(f'{tmp}/plan.json'))['schedules']
    released = {k: v for k, v in json.load(open(f'schedules/leaf/{bb}.json'))['schedules'].items() if k.startswith('latency/')}
    same = sum(got.get(k, {}).get('config') == v['config'] for k, v in released.items())
    check(f'{bb}: leaf plan --budgets <leaf budgets output> gives the released schedules', same == len(released), f'{same}/{len(released)}')


def test_alpha_tables(tmp):
    bb, P = 'deit_b', 'schedules/leaf/planning/deit_b'
    ref = json.load(open(f'schedules/main_comparison/{bb}.json'))['schedules']
    for arm, probes, op in (('ablation/m0_operator', 'probes_m0.json', 'M0'), ('seed251', 'probes_seed251.json', 'M3'),
                            ('seed351', 'probes_seed351.json', 'M3')):
        run('plan', '--backbone', bb, '--probes', f'{P}/{probes}', '--groups', f'{P}/groups.json', '--salt', f'{P}/salt.json',
            '--budgets', f'{P}/budgets.json', '--operator', op, '--alphas', '0', '1/2', '2/3', '3/4', '--out', f'{tmp}/{op}.json')
        got = json.load(open(f'{tmp}/{op}.json'))['schedules']
        keys = [k for k in ref if k.startswith(arm + '/')]
        same = sum(config_hash(got['latency/' + k.split('/')[-1]]['config']) == config_hash(ref[k]['config']) for k in keys)
        check(f'{bb}: leaf plan --operator {op} --probes {probes} gives the paper schedules {arm}/*', keys and same == len(keys),
              f'{same}/{len(keys)}')


def test_workloads(tmp):
    from leaf.planner.alpha import config_gmac
    bb, P = 'deit_t', 'schedules/leaf/planning/deit_t'
    run('workloads', '--backbone', bb, '--probes', f'{P}/probes.json', '--groups', f'{P}/groups.json', '--out', f'{tmp}/w.json')
    rows = json.load(open(f'{tmp}/w.json'))['rows']
    kinds = collections.Counter(r['name'].split('/')[0] for r in rows)
    within = all(r['gmac_target'] is None or config_gmac(r['config'], SPECS[bb]) <= r['gmac_target'] + 1e-9 for r in rows)
    check(f'{bb}: leaf workloads plans {len(rows)} distinct workloads of every kind within their GMAC points',
          len(rows) >= 40 and len({config_hash(r['config']) for r in rows}) == len(rows) and within
          and set(kinds) == {'diffrate', 'm3', 'm0', 'no_forecast', 'prune_only', 'merge_only'}, dict(kinds))


def test_probe_merge(tmp):
    """Fake cached runs stand in for the GPU: probe only writes probes.json when every run of the grid exists."""
    bb, groups = 'deit_t', 'schedules/leaf/planning/deit_t/groups.json'
    spec, d = SPECS[bb], forecast_config(groups, 16, 32)
    out = Path(tmp) / 'probes'

    def fake(alphas):
        for j in probe_jobs(spec, d, [0., .5], alphas, [24, 96]):
            p = out / 'runs/half/runs' / f"{config_hash(j['config'])}.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(dict(kl=j['block'] + j['count'] / 1000 + j['theta'] + j['alpha'] / 10)))

    def probe(*alphas, dry=False):
        return run('probe', '--backbone', bb, '--groups', groups, '--out', str(out), '--thetas', '0', '1/2',
                   '--deltas', '24', '96', '--alphas', *alphas, *(['--dry-run'] if dry else []))
    fake([0.]); probe('0')
    first = json.load(open(out / 'probes.json'))['records']
    log = probe('0', '1/3', dry=True)
    n_new = len(list(probe_jobs(spec, d, [0., .5], [1 / 3], [24, 96])))
    check('leaf probe: only the new grid points are left to run (cached runs are reused)', f'{n_new} to run' in log, log.strip())
    fake([0., 1 / 3]); probe('0', '1/3'); fake([1.]); probe('1')
    doc = json.load(open(out / 'probes.json'))
    keys = [(r['block'], r['count'], r['theta'], r['alpha']) for r in doc['records']]
    curves = load_probe_curves([out / 'probes.json'], spec)
    check('leaf probe: probes.json keeps earlier grids in order and appends new grid points once',
          doc['records'][:len(first)] == first and len(keys) == len(set(keys)) and len(doc['grids']) == 3
          and sorted({a for _, a in curves}) == [0., 1 / 3, 1.], f"{len(keys)} probes, {len(doc['grids'])} grids")


def test_search_dry_run(tmp):
    one = run('search', '--backbone', 'deit_t', '--workspace', f'{tmp}/ws1', '--dry-run')
    two = run('search', '--backbone', 'deit_t', '--workspace', f'{tmp}/ws2', '--gpus', '0,1', '--dry-run')
    steps = lambda log: [line.split(']')[0][1:] for line in log.splitlines() if line.startswith('[')]
    check('leaf search --dry-run: every step, in order, on one GPU',
          steps(one) == ['group-heads', 'probe', 'budgets', 'workloads', 'time-workloads', 'fit-latency', 'plan'] and '--shard' not in one)
    timing = [line for line in two.splitlines() if line.startswith(('[budgets]', '[time-workloads]'))]
    check('leaf search --dry-run --gpus 0,1: one probe shard per GPU, timing on GPU 0 only',
          'CUDA_VISIBLE_DEVICES=0' in two and 'CUDA_VISIBLE_DEVICES=1' in two and '--shard 1/2' in two
          and all('CUDA_VISIBLE_DEVICES=0 ' in line for line in timing))


def test_search_run(tmp):
    """A real search without a GPU: every probe of a small grid is cached (fake KL), SALT and budgets are the released ones."""
    bb, P = 'deit_t', 'schedules/leaf/planning/deit_t'
    spec, d = SPECS[bb], forecast_config(f'{P}/groups.json', 16, 32)
    ws = Path(tmp) / 'ws'
    for j in probe_jobs(spec, d, [0., .5], [0., .5], [24, 96]):
        p = ws / 'probes/runs/half/runs' / f"{config_hash(j['config'])}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(dict(kl=(j['block'] + 1) * (spec.n_tokens - j['count']) / 1e4 * (1.2 - j['alpha'] / 2 - j['theta'] / 5))))
    log = run('search', '--backbone', bb, '--workspace', str(ws), '--groups', f'{P}/groups.json', '--thetas', '0', '1/2',
              '--alphas', '0', '1/2', '--deltas', '24', '96', '--salt', f'{P}/salt.json', '--budget-ms', '4.0', '4.5')
    out = json.load(open(ws / 'schedules.json')); hist = json.load(open(ws / 'search.json'))
    check('leaf search: runs probe and plan as processes, writes schedules.json and search.json',
          len(out['schedules']) == 2 and [s['step'] for s in hist['runs'][-1]['steps']] == ['probe', 'plan'], log.strip().splitlines()[-1])


def main_():
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    test_budgets()
    with tempfile.TemporaryDirectory() as tmp:
        test_plan_budgets_file(tmp); test_alpha_tables(tmp); test_workloads(tmp); test_probe_merge(tmp); test_search_dry_run(tmp); test_search_run(tmp)
    finish()


if __name__ == '__main__':
    main_()
