"""Does the alpha score reproduce the sealed k schedules on ImageNet val? (plan alpha-ablation-paper-v3-plan-2026-10-19)

Stages:
  tasks    (CPU) every q1_main_v2 schedule with a forecast weight: leaf_alpha4/* (7 backbones), abl/* and seed*/*
           arms and ref/M3/gmac/* with k > 0; deduplicated by config; D-placebo variants skipped (they keep
           the k score). Written once to ROOT/tasks.json with its sha256.
  worker   (GPU, one process per idle card) val 50k at BF16 batch 32 for the converted 'pda' config
           ('alpha' arm) and, with --control, the legacy config re-run in the same session without cache
           reuse ('control' arm), so drift of the environment is separated from the alpha change.
  report   (CPU) per-image comparison with the sealed legacy arrays (and the control): changed predictions,
           delta top-1, max |delta KL|, McNemar; writes ROOT/summary.json and the report markdown.
Usage: python leaf/reproduction/experiments/alpha_kappa_equivalence.py tasks|report
       CUDA_VISIBLE_DEVICES=0 python leaf/reproduction/experiments/alpha_kappa_equivalence.py worker [--control]
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from leaf.model.alpha_score import BASES, config_to_alpha

ROOT = REPO / 'results/alpha_kappa_equivalence'
REPORT = REPO / 'results/reports/alpha-equivalence-2026-10-19.md'
SPLIT, N_VAL = 'val', 50000
PLACEBO = ('shuffle', 'shuffle_size', 'identity')


def load(p): return json.loads(Path(p).read_text())


def save(p, obj):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp'); tmp.write_text(json.dumps(obj, indent=1) + '\n'); tmp.replace(p)


def selected(name, cfg):
    if cfg.get('kind') not in BASES or not cfg.get('k'): return False
    if cfg.get('ablation', {}).get('variant') in PLACEBO: return False
    return name.startswith(('leaf_alpha4/', 'ablation/', 'seed', 'ref/M3/gmac/'))


def tasks():
    dest = ROOT / 'tasks.json'
    if dest.exists(): return load(dest)
    import leaf.reproduction.experiments.main_comparison as main
    from leaf.evaluation.runner import config_hash
    out = {}
    for bb in main.BBS:
        for name, row in load(main.ROOT / bb / 'plans.json')['jobs'].items():
            cfg = row.get('config')
            if cfg is None or not selected(name, cfg): continue
            tid = f'{bb}-{config_hash(cfg)}'
            t = out.setdefault(tid, dict(id=tid, backbone=bb, legacy=cfg, alpha=config_to_alpha(cfg),
                                         legacy_sha=config_hash(cfg), names=[]))
            t['names'].append(name)
    doc = dict(tasks=sorted(out.values(), key=lambda t: (main.BBS.index(t['backbone']), t['id'])), time=time.time())
    doc['sha256'] = hashlib.sha256(json.dumps(doc['tasks'], sort_keys=True).encode()).hexdigest()
    save(dest, doc)
    return doc


def runner(bb):
    """ExtraRunner of q1_main_v2 (same harness, data, teachers), rooted here, with no cache reuse."""
    from leaf.reproduction.runners.alpha_runner import alpha_runner
    return alpha_runner(bb, ROOT)


def claim(arm, tid):
    p = ROOT / 'claims' / arm / f'{tid}.lock'; p.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.close(os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)); return True
    except FileExistsError:
        return False


def worker(control=False):
    import gc
    import torch
    from leaf.reproduction.runners.exclusive_timing import guard_foreign
    guard_foreign()
    doc = tasks(); r, bb_last = None, None
    for arm in ('alpha', 'control') if control else ('alpha',):
        for t in doc['tasks']:
            done = ROOT / 'completed' / arm / f"{t['id']}.json"
            if done.exists() or not claim(arm, t['id']): continue
            if t['backbone'] != bb_last:
                r = None; gc.collect(); torch.cuda.empty_cache(); r = runner(t['backbone']); bb_last = t['backbone']
            cfg = t['alpha'] if arm == 'alpha' else t['legacy']
            o = r.run(cfg, SPLIT)
            save(done, dict(status='ok' if o is not None else 'infeasible', path=str(r.path(cfg, SPLIT).with_suffix('.npz')),
                            top1=None if o is None else float(o['correct'].mean() * 100), names=t['names'],
                            gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), time=time.time()))
            print(arm, t['id'], t['names'][0], flush=True)


def legacy_arrays(bb, cfg):
    """Sealed per-image val arrays of a legacy config from the q1_main_v2 campaign or its reuse roots."""
    import leaf.reproduction.experiments.main_comparison as main
    from leaf.evaluation.runner import config_hash
    roots = [main.ROOT / bb] + [main.SOURCE.resolve() / bb, main.VALIDATION.resolve() / bb, main.LOW.resolve() / bb,
                                main.EXTRA_ABL.resolve() / bb]
    for base in roots:
        p = base / SPLIT / 'runs' / config_hash(cfg)
        if p.with_suffix('.json').exists() and p.with_suffix('.npz').exists():
            row = load(p.with_suffix('.json'))
            if row.get('config') == cfg and row.get('n') == N_VAL:
                path = row.get('reused_from')
                return dict(np.load(p.with_suffix('.npz'))), str(path or p)
    return None, None


def compare(a, b):
    from leaf.evaluation.data import mcnemar
    p, nb, nc = mcnemar(a['correct'].astype(bool), b['correct'].astype(bool))
    return dict(changed_pred=int((a['pred'] != b['pred']).sum()), delta_top1=100 * float(b['correct'].mean() - a['correct'].mean()),
                max_abs_dkl=float(np.abs(a['kl'].astype(np.float64) - b['kl'].astype(np.float64)).max()), mcnemar_p=p,
                only_a=nb, only_b=nc)


def report():
    doc = tasks(); rows = []
    for t in doc['tasks']:
        row = dict(id=t['id'], backbone=t['backbone'], names=t['names'], alpha=t['alpha']['alpha'])
        legacy, src = legacy_arrays(t['backbone'], t['legacy'])
        row['legacy_source'] = src
        arrs = {}
        for arm in ('alpha', 'control'):
            done = ROOT / 'completed' / arm / f"{t['id']}.json"
            if done.exists() and load(done)['status'] == 'ok': arrs[arm] = dict(np.load(load(done)['path']))
        if legacy is not None:
            row['legacy_top1'] = 100 * float(legacy['correct'].mean())
            if 'alpha' in arrs: row['alpha_vs_legacy'] = compare(legacy, arrs['alpha'])
            if 'control' in arrs: row['control_vs_legacy'] = compare(legacy, arrs['control'])
        if 'alpha' in arrs and 'control' in arrs: row['alpha_vs_control'] = compare(arrs['control'], arrs['alpha'])
        rows.append(row)
    cmp = [r['alpha_vs_legacy'] for r in rows if 'alpha_vs_legacy' in r]
    summary = dict(tasks_sha=doc['sha256'], n_tasks=len(rows), n_compared=len(cmp),
                   identical=sum(c['changed_pred'] == 0 for c in cmp),
                   max_changed_pred=max((c['changed_pred'] for c in cmp), default=None),
                   max_abs_delta_top1=max((abs(c['delta_top1']) for c in cmp), default=None),
                   min_mcnemar_p=min((c['mcnemar_p'] for c in cmp), default=None),
                   missing_legacy=[r['id'] for r in rows if r['legacy_source'] is None], rows=rows)
    save(ROOT / 'summary.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k != 'rows'}, indent=1))
    return summary


def changed_ids():
    """Tasks whose alpha arm differs from the sealed legacy arrays (or has none): only these need the control."""
    rows = load(ROOT / 'summary.json')['rows']
    return {r['id'] for r in rows if r.get('alpha_vs_legacy', {}).get('changed_pred', 1) != 0}


def worker_control_changed():
    """Control arm (legacy config re-run in this session) for the changed tasks only."""
    import gc
    import torch
    from leaf.reproduction.runners.exclusive_timing import guard_foreign
    guard_foreign()
    doc = tasks(); want = changed_ids(); r, bb_last = None, None
    for t in doc['tasks']:
        done = ROOT / 'completed' / 'control' / f"{t['id']}.json"
        if t['id'] not in want or done.exists() or not claim('control', t['id']): continue
        if t['backbone'] != bb_last:
            r = None; gc.collect(); torch.cuda.empty_cache(); r = runner(t['backbone']); bb_last = t['backbone']
        o = r.run(t['legacy'], SPLIT)
        save(done, dict(status='ok' if o is not None else 'infeasible', path=str(r.path(t['legacy'], SPLIT).with_suffix('.npz')),
                        top1=None if o is None else float(o['correct'].mean() * 100), names=t['names'],
                        gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), time=time.time()))
        print('control', t['id'], t['names'][0], flush=True)


def status():
    doc = tasks(); n = len(doc['tasks'])
    c = {arm: len(list((ROOT / 'completed' / arm).glob('*.json'))) for arm in ('alpha', 'control')}
    print(f"equivalence: alpha {c['alpha']}/{n} control {c['control']}")


def release():
    n = 0
    for p in (ROOT / 'claims').rglob('*.lock'):
        if not (ROOT / 'completed' / p.parent.name / (p.stem + '.json')).exists(): p.unlink(); n += 1
    print(f'released {n} stale claims')


def write_markdown(summary):
    rows = summary['rows']; cmp = [r for r in rows if 'alpha_vs_legacy' in r]
    lines = ['# Equivalence check of the α score against κ schedules on ImageNet val (2026-10-19)', '',
             'Plan: `results/notes/plan/alpha-ablation-paper-v3-plan-2026-10-19.md`. Raw: `' + str(ROOT.relative_to(REPO)) + '`.', '',
             'Each sealed κ schedule of `q1_main_v2` (κ > 0) is converted to a `pda` config with α = κ/(1+κ), val 50k is rerun '
             '(BF16, bs32, same harness), and each image is compared with the sealed per-image κ array.', '',
             f"- Schedules: {summary['n_tasks']} (tasks sha `{summary['tasks_sha'][:12]}`), compared: {summary['n_compared']}.",
             f"- Identical (0 images with changed prediction): **{summary['identical']}/{summary['n_compared']}**.",
             f"- At most {summary['max_changed_pred']} images changed; max |Δtop-1| = {summary['max_abs_delta_top1']} points; "
             f"min McNemar p = {summary['min_mcnemar_p']}.",
             f"- Missing κ arrays: {len(summary['missing_legacy'])}.", '']
    diff = [r for r in cmp if r['alpha_vs_legacy']['changed_pred']]
    if diff:
        lines += ['## Schedules with differences', '', '| Schedule | top-1 κ | images changed | Δtop-1 | max ΔKL | p | rerun κ control: images changed |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        for r in diff:
            a = r['alpha_vs_legacy']; c = r.get('control_vs_legacy', {})
            lines.append(f"| {r['backbone']} {r['names'][0]} | {r['legacy_top1']:.2f} | {a['changed_pred']} | {a['delta_top1']:+.3f} | "
                         f"{a['max_abs_dkl']:.2e} | {a['mcnemar_p']:.3f} | {c.get('changed_pred', '')} |")
        lines.append('')
    by = {}
    for r in cmp: by.setdefault(r['backbone'], []).append(r['alpha_vs_legacy']['changed_pred'])
    lines += ['## By backbone', '', '| Backbone | schedules | identical | total images changed |', '|---|---:|---:|---:|']
    for bb, v in by.items(): lines.append(f'| {bb} | {len(v)} | {sum(x == 0 for x in v)} | {sum(v)} |')
    REPORT.write_text('\n'.join(lines) + '\n')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('stage', choices=('tasks', 'worker', 'report', 'status', 'release'))
    ap.add_argument('--control', action='store_true')
    ap.add_argument('--control-changed', action='store_true', help='control arm for tasks that differ from legacy only')
    a = ap.parse_args()
    if a.stage == 'tasks':
        d = tasks(); print(len(d['tasks']), 'tasks', d['sha256'])
    elif a.stage == 'worker':
        worker_control_changed() if a.control_changed else worker(a.control)
    elif a.stage == 'status':
        status()
    elif a.stage == 'release':
        release()
    else:
        write_markdown(report())


if __name__ == '__main__':
    main()
