"""Alpha form of the probe results that the campaigns wrote with the forecast weight k (CPU only, additive).

The campaigns before the alpha score probed alpha in {0, 1/2, 2/3, 3/4} as k = alpha / (1 - alpha) in {0, 1, 2, 3}
(score z(log P) + k z(D), the same token order as (1 - alpha) z(log P) + alpha z(D)). Those runs are kept as they
are in results/; this tool writes their alpha form next to them.

    python tools/alpha_form.py probe-tables   M0-operator and calibration-seed probe means in alpha
                                             -> schedules/leaf/planning/<bb>/probes_m0.json, probes_seed<s>.json (tracked)
    python tools/alpha_form.py probe-cache    alpha-keyed probe-run cache for `leaf probe` (needs the per-probe runs under results/)
                                             -> results/probe_runs_alpha/<bb>/runs/half/runs/<alpha config_sha>.{json,npz}
                                                (relative symbolic links) and results/probe_runs_alpha/<bb>/sources.json
    python tools/alpha_form.py check          `leaf probe --out results/probe_runs_alpha/<bb>` must rebuild the released
                                             probes.json (same probes, same KL) without a GPU

A link named after an alpha config can point to a run of the equivalent k config; sources.json lists, for every
link, the run it points to and whether that run was made in the k form.
"""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.chdir(REPO)

from leaf.backbones import SPECS                                    # noqa: E402
from leaf.evaluation.runner import config_hash                           # noqa: E402
from leaf.model.alpha_score import config_to_alpha, kappa_to_alpha      # noqa: E402

RESULTS = REPO / 'results'
PLANNING = REPO / 'schedules/leaf/planning'
CACHE = RESULTS / 'probe_runs_alpha'
SEEDS = {'deit_b': (251, 351), 'mae_b': (251, 351), 'augreg_s': (251, 351)}
ALPHA_RUNS = 'ablation_and_alpha_grid/{bb}/half/runs'      # runs made in the alpha form (pda configs)


def load(p): return json.loads(Path(p).read_text())


def save(p, obj):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(obj, indent=1) + '\n')


def alpha_records(rows, source):
    return [dict(block=r['block'], count=r['count'], theta=r['theta'], alpha=kappa_to_alpha(r['kappa']), kl=r['kl'], source=source)
            for r in rows]


def probe_tables(args):
    for bb in SPECS:
        src = RESULTS / f'low_budget_and_factorial/{bb}/probe_means_M0.json'
        save(PLANNING / bb / 'probes_m0.json', dict(
            backbone=bb, images='calib_bal even indices (5,000 training images)', batch=32,
            operator='M0 (average merge; pure pruning: plain operator)', forecast=dict(G=16, rank=32),
            note='probed as k = alpha / (1 - alpha); plan with: leaf plan --operator M0 --alphas 0 1/2 2/3 3/4',
            records=alpha_records(load(src), str(src.relative_to(REPO)))))
        for seed in SEEDS.get(bb, ()):
            src = RESULTS / f'horizon_norm_seed_ablation/{bb}/probe_means_seed{seed}.json'
            save(PLANNING / bb / f'probes_seed{seed}.json', dict(
                backbone=bb, images=f'calib_bal, calibration draw with seed {seed} (5,000 training images)', batch=32,
                operator='M3 (pure pruning: plain operator)', forecast=dict(G=16, rank=32),
                note='probed as k = alpha / (1 - alpha); the calibration-seed ablation of the paper',
                records=alpha_records(load(src), str(src.relative_to(REPO)))))
        print(bb, 'probe tables written', flush=True)


def link(target, name, root):
    name.parent.mkdir(parents=True, exist_ok=True)
    if name.is_symlink() or name.exists(): name.unlink()
    os.symlink(os.path.relpath(target, name.parent), name)


def probe_cache(args):
    for bb in args.backbones:
        dest = CACHE / bb / 'runs/half/runs'
        sources = {}
        # 1. runs made in the alpha form: every 'pda' run on the 5,000 probe images at batch 32
        for meta in sorted((RESULTS / ALPHA_RUNS.format(bb=bb)).glob('*.json')):
            row = load(meta)
            if row['config']['kind'] != 'pda' or row.get('n') != 5000 or row.get('batch_size') != 32: continue
            sources[meta.stem] = dict(run=str(meta.relative_to(REPO)), form='alpha')
        # 2. runs made in the k form (M3 probes of the GMAC-planning campaign), named after the alpha config
        for r in load(RESULTS / f'low_budget_and_factorial/{bb}/probe_means_M3.json'):
            sha = config_hash(config_to_alpha(r['config']))
            if sha in sources: continue
            sources[sha] = dict(run=r['metadata'], form='k', k_config_sha=Path(r['metadata']).stem)
        for sha, s in sources.items():
            meta = REPO / s['run']
            if not meta.exists() or not meta.with_suffix('.npz').exists():
                raise SystemExit(f'{meta} missing: the per-probe runs are not under results/ (raw results, see results/README.md)')
            link(meta, dest / f'{sha}.json', CACHE); link(meta.with_suffix('.npz'), dest / f'{sha}.npz', CACHE)
        save(CACHE / bb / 'sources.json', dict(backbone=bb, note=__doc__.split('\n\n')[-1].strip(), runs=sources))
        print(bb, f"{sum(s['form'] == 'alpha' for s in sources.values())} alpha-form runs, "
                  f"{sum(s['form'] == 'k' for s in sources.values())} k-form runs linked", flush=True)


def check(args):
    from leaf.cli import main
    ok = True
    for bb in args.backbones:
        out = CACHE / bb
        (out / 'probes.json').unlink(missing_ok=True)
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            main(['probe', '--backbone', bb, '--groups', str(PLANNING / bb / 'groups.json'), '--out', str(out)])
        if not (out / 'probes.json').exists():
            print(bb, 'FAIL', log.getvalue().strip().splitlines()[-1]); ok = False; continue
        key = lambda r: (r['block'], r['count'], r['theta'], r['alpha'])
        got = {key(r): r['kl'] for r in load(out / 'probes.json')['records']}
        ref = {key(r): r['kl'] for r in load(PLANNING / bb / 'probes.json')['records']}
        same = got == ref
        ok &= same
        print(bb, 'PASS' if same else 'FAIL', f'{len(got)} probes rebuilt from the cache, released {len(ref)}, KL identical: {same}')
    if not ok: raise SystemExit(1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=('probe-tables', 'probe-cache', 'check'))
    ap.add_argument('--backbones', nargs='+', default=list(SPECS))
    a = ap.parse_args()
    {'probe-tables': probe_tables, 'probe-cache': probe_cache, 'check': check}[a.stage](a)
