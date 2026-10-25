"""Latency pricing: does pricing schedules by latency matter where latency departs from GMAC (GPU bs4/8, CPU)?

Append-only campaign (results/notes/plan/q1-revision-plan-2026-10-18.md, PRICING). Stages:
  inventory   (CPU) locked workload per backbone: historical M0/M3 configs, near-GMAC pairs, stress, DR anchors, dense
  gpu_time    (GPU0, GPU1 idle) three interleaved rounds at the GPU regimes
  cpu_time    (CPU, both GPUs idle) three interleaved rounds, pinned cores, low-rank D on CPU
  analyze     (CPU) per regime: distinguishable near-GMAC pairs, grouped-CV MAPE of GMAC-only vs SALT formulas, gate
Run detached: setsid nohup python leaf/reproduction/experiments/latency_pricing.py <stage> > log 2>&1 &
Existing code is only imported; nothing old is modified.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
from leaf.reproduction.runners.exclusive_timing import cast_exclusive, cpu_exclusive

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / 'results/latency_pricing'
SOURCE = REPO / 'results/gmac_planning'
COMPARISON = REPO / 'results/salt_fit'
EXTRA = REPO / 'results/salt_validation/extra'
BBS = ('deit_s', 'deit_b', 'mae_b')
REGIMES = {'gpu_bs4': ('cuda', 4), 'gpu_bs8': ('cuda', 8), 'gpu_bs32': ('cuda', 32),
           'cpu_bs1': ('cpu', 1), 'cpu_bs8': ('cpu', 8)}
ROUNDS = 3
GATE = dict(min_distinguishable_pairs=.30, gmac_mape_ratio=2.0)


def save(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=1, default=float)); tmp.replace(path)


def load(path):
    return json.loads(Path(path).read_text())


def immutable(path, obj):
    obj = json.loads(json.dumps(obj, default=float))
    if Path(path).exists():
        assert load(path) == obj, f'refusing to overwrite {path}'
    else:
        save(path, obj)


def key(cfg):
    from leaf.evaluation.runner import config_hash
    return config_hash(cfg)


def gpus_idle():
    out = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
    return not [int(s) for s in out.split() if s.strip() and int(s) != os.getpid()]


def card_idle():
    """True when no other process uses the card this process times on (timing rule: only the timed card must be free;
    the other card may run accuracy work at the same time)."""
    index = os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0]
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,pci.bus_id', '--format=csv,noheader'], text=True)
    bus = dict(r.split(', ') for r in rows.strip().splitlines())[index]
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_bus_id', '--format=csv,noheader'], text=True)
    return not [r for r in apps.strip().splitlines() if r and r.split(', ')[1] == bus and int(r.split(', ')[0]) != os.getpid()]


# CPU timing on CCD1 (physical cores 8-15, SMT siblings 24-31 left idle); GPU workers are pinned to CCD0
# (CPUs 0-7,16-23) by the orchestrator, so the two never share cores or L3.
CPU_CORES = tuple(range(8, 16))


# ---------------------------------------------------------------- inventory
def inventory(bb):
    """Locked list of workloads; chosen from historical inventories only, never from accuracy."""
    dest = ROOT / bb / 'workload.json'
    if dest.exists(): return load(dest)
    jobs = {}
    def add(name, cfg, group, source):
        k = key(cfg)
        row = jobs.setdefault(k, dict(id=k, config=cfg, names=[], groups=[], sources=[]))
        row['names'].append(name); row['groups'].append(group); row['sources'].append(source)
    add('dense', dict(kind='dense'), 'dense', 'reference')
    for r in load(COMPARISON / 'dataset.json')['rows']:
        if r['backbone'] == bb: add('history/' + r['names'][0], r['config'], 'history/' + r['group'], str(COMPARISON / 'dataset.json'))
    extra = load(EXTRA / bb / 'design.json')
    for name, row in extra['jobs'].items():
        add('extra/' + name, row['config'], 'extra/' + name, str(EXTRA / bb / 'design.json'))
    frozen = load(SOURCE / bb / 'frozen.json')['jobs']
    for name, row in frozen.items():
        if name.startswith('baseline/'):
            add(name, row['config'], name, str(SOURCE / bb / 'frozen.json'))
    pairs = [dict(a=key(extra['jobs'][p['a']]['config']) if p['a'] in extra['jobs'] else None,
                  b=key(extra['jobs'][p['b']]['config']) if p['b'] in extra['jobs'] else None,
                  relative_gmac_gap=p['relative_gmac_gap'], names=[p['a'], p['b']]) for p in extra['pairs']]
    doc = dict(backbone=bb, jobs=list(jobs.values()), pairs=pairs, created=time.time(),
               rule='historical M0/M3 workload set + preselected near-GMAC pairs/stress + DiffRate/baseline grids; no accuracy used')
    immutable(dest, doc)
    return doc


def reduction_jobs(doc):
    """Configs of our pd/pdm family (formula fitting); baselines are timed but not used for the fit."""
    return [j for j in doc['jobs'] if j['config']['kind'] in ('pd', 'pdm', 'dense')]


# ---------------------------------------------------------------- GPU timing
def gpu_time(bb, regimes=('gpu_bs4', 'gpu_bs8', 'gpu_bs32')):
    import torch
    from leaf.latency.timing import measure_latency
    from leaf.reproduction.experiments.low_budget_and_factorial import context
    doc = inventory(bb); done = ROOT / bb / 'gpu_done.json'
    if done.exists(): return
    r, _ = context(bb)          # read-only use of the low-budget runner: model building only
    batches = [REGIMES[g][1] for g in regimes]
    jobs = doc['jobs']; dest = ROOT / bb / 'timing_gpu'
    if dest.exists(): dest.rename(ROOT / bb / f'timing_gpu_interrupted_{time.time_ns()}')
    for rnd in range(ROUNDS):
        order = jobs if rnd % 2 == 0 else jobs[::-1]
        for job in order:
            if not card_idle(): raise RuntimeError('another process appeared on the timed card; session discarded')
            try:
                model = r.inference_model(job['config'])
                meas = {str(bs): cast_exclusive(model, bs, iters=50, warmup=10) for bs in batches}
            except torch.cuda.OutOfMemoryError:
                meas = dict(status='OOM')
            except Exception as e:     # baseline adapters may refuse a config; keep the reason
                meas = dict(status='error', error=repr(e))
            save(dest / f"{job['id']}#{rnd}.json", dict(id=job['id'], round=rnd, measurements=meas,
                 clean=card_idle(), device=torch.cuda.get_device_name(0), time=time.time()))
            model = None; gc.collect(); torch.cuda.empty_cache()
    save(done, dict(rounds=ROUNDS, batches=batches, n=len(jobs), time=time.time()))


# ---------------------------------------------------------------- CPU timing
def cpu_build(vit, spec, cfg):
    from leaf.latency.cpu_timing import cpu_inference_copy
    from leaf.forecast.cpu_forecast import enable_cpu_lowrank
    from leaf.reproduction.experiments.low_budget_and_factorial import build
    model = vit if cfg['kind'] == 'dense' else build(vit, spec, cfg)
    m = cpu_inference_copy(model)
    enable_cpu_lowrank(m)
    return m


def cpu_time(bb, regimes=('cpu_bs1', 'cpu_bs8'), only_family=True):
    from leaf.latency.cpu_timing import pin, measure_latency_cpu
    info = pin(CPU_CORES)
    import torch
    from leaf.backbones import load as load_backbone
    doc = inventory(bb); done = ROOT / bb / 'cpu_done.json'
    if done.exists(): return
    vit, spec = load_backbone(bb, 'cpu')
    jobs = reduction_jobs(doc) + ([] if only_family else [j for j in doc['jobs'] if j not in reduction_jobs(doc)])
    dest = ROOT / bb / 'timing_cpu'
    if dest.exists(): dest.rename(ROOT / bb / f'timing_cpu_interrupted_{time.time_ns()}')
    batches = [REGIMES[g][1] for g in regimes]
    for rnd in range(ROUNDS):
        order = jobs if rnd % 2 == 0 else jobs[::-1]
        for job in order:
            load1 = os.getloadavg()[0]
            try:
                m = cpu_build(vit, spec, job['config'])
                meas = {str(bs): cpu_exclusive(m, bs, rounds=3, iters=20 if bs == 1 else 8, warmup=5) for bs in batches}
            except Exception as e:
                meas = dict(status='error', error=repr(e))
            save(dest / f"{job['id']}#{rnd}.json", dict(id=job['id'], round=rnd, measurements=meas, pin=info,
                 loadavg_before=load1, loadavg_after=os.getloadavg()[0], cpu=platform.processor(), time=time.time()))
            m = None; gc.collect()
    save(done, dict(rounds=ROUNDS, batches=batches, n=len(jobs), pin=info, time=time.time()))


# ---------------------------------------------------------------- analysis
def measured(bb, regime):
    """{id: (mean ms, sd over round means)} for a regime; missing/failed configs omitted."""
    device, bs = REGIMES[regime]
    folder = ROOT / bb / ('timing_gpu' if device == 'cuda' else 'timing_cpu')
    rows = {}
    for p in sorted(folder.glob('*.json')):
        rec = load(p); m = rec['measurements'].get(str(bs)) if isinstance(rec['measurements'], dict) else None
        if isinstance(m, dict) and 'ms_mean' in m:
            rows.setdefault(rec['id'], []).append(m['ms_mean'])
    return {k: (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.) for k, v in rows.items() if len(v) == ROUNDS}


def grouped_cv(rows, names, alphas=(0., 1e-3, 1e-2, 1e-1, 1.), folds=5, seed=0):
    """Outer workload-grouped CV MAPE; inner grouped CV picks the ridge alpha on training folds only."""
    from leaf.latency.salt import fit, predict
    groups = sorted({r['group'] for r in rows}); rng = np.random.default_rng(seed); rng.shuffle(groups)
    fold_of = {g: i % folds for i, g in enumerate(groups)}
    x = np.array([[r['features'].get(k, 0.) for k in names] for r in rows]); y = np.array([r['ms'] for r in rows])
    f = np.array([fold_of[r['group']] for r in rows]); pred = np.zeros_like(y)
    def inner_alpha(idx):
        best = None
        for a in alphas:
            err = []
            for j in range(folds):
                tr, te = idx[f[idx] != j], idx[f[idx] == j]
                if len(te) == 0 or len(tr) < len(names) + 2: continue
                m = fit(x[tr], y[tr], a); err += list(abs(predict(m, x[te]) - y[te]) / y[te])
            score = np.mean(err) if err else np.inf
            if best is None or score < best[0]: best = (score, a)
        return best[1]
    for i in range(folds):
        tr, te = np.where(f != i)[0], np.where(f == i)[0]
        if len(te) == 0: continue
        m = fit(x[tr], y[tr], inner_alpha(tr)); pred[te] = predict(m, x[te])
    full = fit(x, y, inner_alpha(np.arange(len(y))))
    full.update(feature_names=list(names))
    return dict(mape=float(np.mean(abs(pred - y) / y) * 100), pred=pred.tolist(), model=full)


def features_of(cfg, spec):
    from leaf.latency.salt import features
    if cfg['kind'] == 'dense':
        return features(dict(kind='pd', prune_keep=[spec.n_tokens] * spec.depth, merge_keep=[spec.n_tokens] * spec.depth,
                             k={}, groups={}, rank=32), spec)
    return features(cfg, spec)


def analyze_regime(bb, regime):
    from leaf.backbones import SPECS
    from leaf.latency.salt import FORMULAS
    spec = SPECS[bb]; doc = inventory(bb); ms = measured(bb, regime)
    fam = {j['id']: j for j in reduction_jobs(doc)}
    rows = [dict(id=i, group=fam[i]['groups'][0], ms=ms[i][0], sd=ms[i][1], features=features_of(fam[i]['config'], spec))
            for i in fam if i in ms]
    out = dict(backbone=bb, regime=regime, n=len(rows), formulas={})
    for name, cols in FORMULAS.items():
        cv = grouped_cv(rows, cols)
        out['formulas'][name] = dict(mape=cv['mape'], model=cv['model'], formula=name)
    # Pair test: distinguishable latency differences between near-GMAC pairs, and ranking by each formula's CV prediction.
    pred = {name: dict(zip([r['id'] for r in rows], grouped_cv(rows, cols)['pred'])) for name, cols in FORMULAS.items()}
    pairs = []
    for p in doc['pairs']:
        a, b = p['a'], p['b']
        if a not in ms or b not in ms or a not in pred['gmac_only'] or b not in pred['gmac_only']: continue
        tau = max(.05 * ms[a][0] / 20, 2 * np.sqrt((ms[a][1] ** 2 + ms[b][1] ** 2) / 2))   # 0.05 ms at ~20 ms scale, relative
        diff = ms[a][0] - ms[b][0]
        pairs.append(dict(a=a, b=b, diff_ms=diff, tau=tau, distinguishable=abs(diff) > tau,
                          correct={n: bool(np.sign(pred[n][a] - pred[n][b]) == np.sign(diff)) for n in pred}))
    dist = [q for q in pairs if q['distinguishable']]
    out['pairs'] = dict(n=len(pairs), distinguishable=len(dist),
                        share=len(dist) / max(1, len(pairs)),
                        correct={n: sum(q['correct'][n] for q in dist) for n in pred}, rows=pairs)
    best = min((n for n in FORMULAS if n != 'gmac_only'), key=lambda n: out['formulas'][n]['mape'])
    ratio = out['formulas']['gmac_only']['mape'] / out['formulas'][best]['mape']
    out['gate'] = dict(best_formula=best, gmac_mape=out['formulas']['gmac_only']['mape'],
                       best_mape=out['formulas'][best]['mape'], ratio=ratio,
                       distinguishable_share=out['pairs']['share'],
                       passed=bool(out['pairs']['share'] >= GATE['min_distinguishable_pairs'] and ratio >= GATE['gmac_mape_ratio']),
                       rule=GATE)
    dense = ms.get(key(dict(kind='dense')))
    out['dense_ms'] = dense[0] if dense else None
    return out


def analyze():
    summary = {}
    for bb in BBS:
        for regime in REGIMES:
            dev = REGIMES[regime][0]
            if not (ROOT / bb / ('gpu_done.json' if dev == 'cuda' else 'cpu_done.json')).exists(): continue
            res = analyze_regime(bb, regime)
            save(ROOT / bb / f'analysis_{regime}.json', res)
            summary[f'{bb}/{regime}'] = res['gate'] | dict(n=res['n'], dense_ms=res['dense_ms'],
                                                           pairs=res['pairs']['distinguishable'], pairs_n=res['pairs']['n'],
                                                           correct=res['pairs']['correct'])
            g = res['gate']
            print(f"{bb:7s} {regime:8s} n={res['n']:3d} dense={res['dense_ms'] or 0:7.2f}ms  GMAC MAPE {g['gmac_mape']:.2f}%  "
                  f"best {g['best_formula']} {g['best_mape']:.2f}% (x{g['ratio']:.2f})  pairs {res['pairs']['distinguishable']}/{res['pairs']['n']}  "
                  f"gate {'PASS' if g['passed'] else 'fail'}", flush=True)
    save(ROOT / 'analysis_summary.json', summary)


# ---------------------------------------------------------------- planning (regimes that passed the gate)
LOW = REPO / 'results/low_budget_and_factorial'
VALIDATION = REPO / 'results/salt_validation'
EXTRA_ABL = REPO / 'results/horizon_norm_seed_ablation'
GMAC_GRID = (.80, .85, .90, .95, 1.00, 1.05, 1.10)
SPLITS = dict(confirm=20018, val=50000, v2=10000)


def design():
    """Own runner binding (copied fields of the audited low-budget design); results never go to old roots."""
    dest = ROOT / 'design.json'
    if dest.exists(): return load(dest)
    from leaf.reproduction.experiments.gmac_planning import sha
    low = load(LOW / 'design.json')
    doc = dict(backbones={bb: low['backbones'][bb] for bb in low['backbones']}, data_cache=low['data_cache'],
               original_root=low['original_root'], merge_root=low['merge_root'], parent_design_sha=low['sha256'],
               rules=dict(accuracy='BF16 deterministic batch32, exact cache reuse only', planner='historical M3 probes, unchanged DP',
                          selection='no confirm/val/V2 feedback; all planned arms evaluated', gate=GATE))
    doc['sha256'] = sha(doc)
    immutable(dest, doc)
    return doc


def runner(bb):
    from leaf.reproduction.runners.campaign_runner import CampaignRunner
    from leaf.reproduction.experiments.low_budget_and_factorial import LowRunner
    r = LowRunner(bb, ROOT, design())
    r.teacher_cache = SOURCE.resolve() / 'teachers'
    r.reuse_roots = [SOURCE.resolve() / bb, VALIDATION.resolve() / bb, LOW.resolve() / bb, EXTRA_ABL.resolve() / bb]
    return r


def dcfg(bb):
    from leaf.planner.configs import ForecastConfig
    return ForecastConfig(Path(design()['backbones'][bb]['root']) / 'groups.json', G=16, rank=32)


def passed_regimes():
    s = load(ROOT / 'analysis_summary.json')
    return sorted({k.split('/')[1] for k, v in s.items() if v['passed'] and not k.endswith('gpu_bs32')})


def plan(bb, regimes):
    """SALT-device and GMAC-only DP on the same historical M3 probes; budgets from measured DR anchors."""
    from leaf.backbones import SPECS
    from leaf.planner.gmac import GmacPlanner, config_cost
    from leaf.planner.plan import plan_latency, config_details
    from leaf.reproduction.experiments.low_budget_and_factorial import probe_curves
    from leaf.reproduction.experiments.gmac_planning import sha
    dest = ROOT / bb / 'plans.json'
    if dest.exists(): return load(dest)
    spec, d = SPECS[bb], dcfg(bb); curves = probe_curves(bb, d, 'M3')   # read-only historical cache
    doc = inventory(bb); ids = {n: j for j in doc['jobs'] for n in j['names']}
    jobs = {}
    for name in ('dense',) + tuple(n for n in ids if n.startswith('baseline/')):
        jobs['ref/' + name] = dict(config=ids[name]['config'], group='reference')
    gp = GmacPlanner(spec, d, curves, 'M3'); gmac_jobs = {}
    anchors = sorted(n for n in ids if n.startswith('baseline/DR'))
    for a in anchors:
        g0 = config_cost(ids[a]['config'], spec)
        for f in GMAC_GRID:
            r = gp.plan(g0 * f); r = dict(r) if isinstance(r, dict) else dict(status='no_result')
            gmac_jobs[f'gmac/{a[9:]}/x{f:.2f}'] = dict(r, group='gmac', cap=g0 * f, anchor=a)
    jobs.update(gmac_jobs)
    for regime in regimes:
        analysis = load(ROOT / bb / f'analysis_{regime}.json'); best = analysis['gate']['best_formula']
        model = analysis['formulas'][best]['model']; ms = measured(bb, regime)
        for a in anchors:
            budget = .97 * ms[ids[a]['id']][0]
            started = time.monotonic()
            r = plan_latency(curves, spec, d, model, budget, 'M3', True)
            jobs[f'salt/{regime}/{a[9:]}'] = dict(r, group='salt', regime=regime, formula=best, budget_ms=budget,
                                                   anchor=a, search_seconds=time.monotonic() - started)
        # Cross-regime transfer control: the bs32 station-rich model planned for bs32, deployed here.
    for name, row in jobs.items():
        if 'config' in row:
            row['config_sha'] = key(row['config'])
            if row['config']['kind'] in ('pd', 'pdm'):
                row['details'] = config_details(row['config'], spec); row['gmac'] = config_cost(row['config'], spec)
    out = dict(backbone=bb, regimes=regimes, jobs=jobs, sha=sha(jobs), time=time.time())
    immutable(dest, out)
    return out


def tasks(bbs=BBS):
    out = {}
    for bb in bbs:
        if not (ROOT / bb / 'plans.json').exists(): continue
        for name, row in load(ROOT / bb / 'plans.json')['jobs'].items():
            if 'config' not in row: continue
            t = out.setdefault(bb + '-' + key(row['config']), dict(id=bb + '-' + key(row['config']), backbone=bb, config=row['config'], names=[]))
            t['names'].append(name)
    return sorted(out.values(), key=lambda t: (t['backbone'], t['id']))


def claim(split, tid):
    p = ROOT / 'claims' / split / f'{tid}.lock'; p.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.close(os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)); return True
    except FileExistsError:
        return False


def worker():
    """Accuracy worker: one per GPU (CUDA_VISIBLE_DEVICES=0 / =1), dynamic claims, cached reuse."""
    import torch
    from leaf.reproduction.runners.exclusive_timing import guard_foreign
    guard_foreign()
    current = None; r = None
    for split in SPLITS:
        for t in tasks():
            done = ROOT / 'completed' / split / f"{t['id']}.json"
            if done.exists() or not claim(split, t['id']): continue
            if current != t['backbone']:
                r = None; gc.collect(); torch.cuda.empty_cache(); r = runner(t['backbone']); current = t['backbone']
            try:
                out = r.run(t['config'], split)
                save(done, dict(status='ok' if out is not None else 'infeasible', task=t['id'],
                                top1=float(out['correct'].mean()) if out is not None else None,
                                kl=float(out['kl'].mean()) if out is not None else None, time=time.time()))
            except torch.cuda.OutOfMemoryError:
                save(done, dict(status='OOM', task=t['id']))
            print(split, t['id'], flush=True)


def final_time(bb, regimes, devices=('cuda', 'cpu')):
    """One session per backbone and regime family; every planned arm and reference, three interleaved rounds."""
    devs = {REGIMES[g][0] for g in regimes} & set(devices)
    ts = [t for t in tasks((bb,))]
    for dev in sorted(devs):
        done = ROOT / bb / f'final_{dev}_done.json'
        if done.exists(): continue
        batches = sorted({REGIMES[g][1] for g in regimes if REGIMES[g][0] == dev} | ({32} if dev == 'cuda' else set()))
        dest = ROOT / bb / f'final_{dev}'
        if dest.exists(): dest.rename(ROOT / bb / f'final_{dev}_interrupted_{time.time_ns()}')
        if dev == 'cuda':
            import torch
            from leaf.latency.timing import measure_latency
            r = runner(bb)
            for rnd in range(ROUNDS):
                for t in (ts if rnd % 2 == 0 else ts[::-1]):
                    if not card_idle(): raise RuntimeError('timed card shared during final timing')
                    try:
                        m = r.inference_model(t['config']); meas = {str(b): cast_exclusive(m, b, iters=50, warmup=10) for b in batches}
                    except Exception as e:
                        meas = dict(status='error', error=repr(e))
                    save(dest / f"{t['id']}#{rnd}.json", dict(id=t['id'], round=rnd, measurements=meas, clean=card_idle(), time=time.time()))
                    m = None; gc.collect(); torch.cuda.empty_cache()
                print(bb, 'final round', rnd, 'done', flush=True)
        else:
            from leaf.latency.cpu_timing import pin, measure_latency_cpu
            from leaf.backbones import load as load_backbone
            info = pin(CPU_CORES); vit, spec = load_backbone(bb, 'cpu')
            for rnd in range(ROUNDS):
                for t in (ts if rnd % 2 == 0 else ts[::-1]):
                    try:
                        m = cpu_build(vit, spec, t['config'])
                        meas = {str(b): cpu_exclusive(m, b, rounds=3, iters=20 if b == 1 else 8, warmup=5) for b in batches}
                    except Exception as e:
                        meas = dict(status='error', error=repr(e))
                    save(dest / f"{t['id']}#{rnd}.json", dict(id=t['id'], round=rnd, measurements=meas, pin=info,
                         loadavg=os.getloadavg()[0], time=time.time()))
                    m = None; gc.collect()
        save(done, dict(rounds=ROUNDS, batches=batches, n=len(ts), time=time.time()))


def final_ms(bb, regime):
    dev, bs = REGIMES[regime]; rows = {}
    for p in sorted((ROOT / bb / f'final_{dev}').glob('*.json')):
        rec = load(p); m = rec['measurements'].get(str(bs)) if isinstance(rec['measurements'], dict) else None
        if isinstance(m, dict) and 'ms_mean' in m: rows.setdefault(rec['id'], []).append(m['ms_mean'])
    return {k: (float(np.mean(v)), float(np.std(v, ddof=1))) for k, v in rows.items() if len(v) == ROUNDS}


def compare(regimes):
    """Pre-declared PRICING test: SALT-device vs GMAC-only at matched measured latency on the target regime."""
    from leaf.evaluation.data import mcnemar, holm
    tests = []; fidelity = []
    for bb in BBS:
        if not (ROOT / bb / 'plans.json').exists(): continue
        jobs = load(ROOT / bb / 'plans.json')['jobs']
        for regime in regimes:
            ms = final_ms(bb, regime)
            gm = [(n, j) for n, j in jobs.items() if j.get('group') == 'gmac' and 'config' in j and bb + '-' + j['config_sha'] in ms]
            for n, j in jobs.items():
                if j.get('group') != 'salt' or j.get('regime') != regime or 'config' not in j: continue
                a = bb + '-' + j['config_sha']
                if a not in ms: continue
                fidelity.append(dict(backbone=bb, regime=regime, name=n, budget=j['budget_ms'], measured=ms[a][0],
                                     predicted=j.get('predicted_ms'), meets=ms[a][0] <= j['budget_ms']))
                best = None
                for gn, g in gm:
                    b = bb + '-' + g['config_sha']; tau = max(.0025 * ms[a][0], 2 * np.sqrt((ms[a][1] ** 2 + ms[b][1] ** 2) / 2))
                    gap = abs(ms[a][0] - ms[b][0])
                    if gap <= tau and (best is None or gap < best[0]): best = (gap, gn, b, tau)
                if best is None:
                    tests.append(dict(backbone=bb, regime=regime, salt=n, matched=False)); continue
                row = dict(backbone=bb, regime=regime, salt=n, gmac=best[1], matched=True, salt_ms=ms[a][0],
                           gmac_ms=ms[best[2]][0], tau=best[3], same_config=a == best[2])
                for split in ('val', 'v2'):
                    ca = np.load(ROOT / bb / split / 'runs' / f"{j['config_sha']}.npz")['correct']
                    cb = np.load(ROOT / bb / split / 'runs' / f"{jobs[best[1]]['config_sha']}.npz")['correct']
                    p, b_, c_ = mcnemar(ca, cb)
                    row[split] = dict(delta=float(100 * (ca.mean() - cb.mean())), p=p)
                tests.append(row)
    for split in ('val', 'v2'):
        fam = [t[split] for t in tests if t.get('matched') and not t['same_config']]
        holm(fam)
    out = dict(tests=tests, fidelity=fidelity, time=time.time())
    save(ROOT / 'compare.json', out)
    m = [t for t in tests if t.get('matched') and not t['same_config']]
    print('matched', len(m), 'of', len(tests), 'identical', sum(t.get('same_config', False) for t in tests))
    for t in m:
        print(t['backbone'], t['regime'], t['salt'], 'vs', t['gmac'], f"{t['salt_ms']:.2f}/{t['gmac_ms']:.2f} ms",
              f"val {t['val']['delta']:+.2f} (holm {t['val']['holm']:.3g})  v2 {t['v2']['delta']:+.2f} (holm {t['v2']['holm']:.3g})")
    print('budget met', sum(f['meets'] for f in fidelity), '/', len(fidelity))


def claim(split, tid):   # overrides the earlier definition: crash-safe claims (leaf/reproduction/runners/exclusive_timing.py)
    from leaf.reproduction.runners.exclusive_timing import claim as common_claim
    return common_claim(ROOT, split, tid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['inventory', 'gpu_time', 'cpu_time', 'analyze', 'plan', 'worker', 'final_time', 'compare'])
    ap.add_argument('--backbones', default=','.join(BBS))
    ap.add_argument('--devices', default='cuda,cpu')
    a = ap.parse_args()
    bbs = a.backbones.split(',')
    if a.stage == 'inventory':
        for bb in bbs: d = inventory(bb); print(bb, len(d['jobs']), 'jobs', len(d['pairs']), 'pairs', flush=True)
    elif a.stage == 'gpu_time':
        for bb in bbs: gpu_time(bb); print('GPU done', bb, flush=True)
    elif a.stage == 'cpu_time':
        for bb in bbs: cpu_time(bb); print('CPU done', bb, flush=True)
    elif a.stage == 'analyze':
        analyze()
    elif a.stage == 'plan':
        regimes = passed_regimes(); print('regimes passing the gate:', regimes)
        for bb in bbs:
            p = plan(bb, regimes); print('PLANNED', bb, len(p['jobs']), flush=True)
    elif a.stage == 'worker':
        worker()
    elif a.stage == 'final_time':
        for bb in bbs: final_time(bb, passed_regimes(), a.devices.split(',')); print('final timing done', bb, a.devices, flush=True)
    else:
        compare(passed_regimes())


if __name__ == '__main__':
    main()
