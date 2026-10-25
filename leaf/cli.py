"""Command line of LEAF:  leaf <command> [options]   (or python -m leaf <command>)

  backbones        the backbones and their CLI names (leaf/backbones.json; add yours with LEAF_BACKBONES=file.json)

Use a schedule (schedules/leaf/<bb>.json, assets/schedules/<bb>.json or the output of `leaf plan`):
  schedules        list the schedules of a file: stations, budget, measured accuracy and latency
  evaluate         top-1 and KL to the dense model on a cached split (prints the paper's number when known)
  benchmark        latency as one CUDA graph (BF16), interleaved rounds

Search schedules for a backbone, GPU or budget:
  search           the whole search in one workspace folder: group-heads -> probe -> budgets -> workloads ->
                   time-workloads -> fit-latency -> plan (each step resumes; see: leaf search --help)
  data             download ImageNet-1k; prepare the uint8 evaluation caches; CLIP text embeddings
  group-heads      head groups of the forecast D (1,024 calibration images)
  probe            single-station probes on the 5,000 calibration images (GPU, shardable)
  budgets          time anchor schedules (DiffRate, GMAC points, dense) on this GPU -> latency budgets
  workloads        timing workloads for fit-latency, planned from the probes at several GMAC points
  time-workloads   time a list of configs on this GPU
  fit-latency      fit the station-aware latency model SALT to timed workloads
  plan             Lagrangian DP under a latency (SALT) or GMAC budget -> schedules

Reproduce the paper:
  experiments      list | run <experiment> <stage>       the paper's experiments (campaign stages)
  report           analyze | tables | figures | assets | all   analyses, tables, figures and assets from results/

Every GPU command refuses a card that another process uses (--allow-busy to override); select the card
with CUDA_VISIBLE_DEVICES.
"""
import argparse
from fractions import Fraction
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

from leaf.console import Progress, duration, log

DEFAULT_DELTAS = (8, 24, 48, 72, 96, 120, 144, 168, 194)     # 197 tokens (patch 16 at 224)
DEFAULT_THETAS = (0., .5, 1.)
DEFAULT_ALPHAS = (0., .25, 1 / 3, .5, 2 / 3, .75, 1.)


def default_deltas(spec):
    """Tokens removed by a probed station: the paper's grid for 197 tokens; on other token counts the same fractions
    of the patch tokens, the largest probe again leaving 3 tokens (257 tokens: 10, 31, ..., 219, 254)."""
    if spec.n_tokens == 197: return list(DEFAULT_DELTAS)
    return [round(d * (spec.n_tokens - 1) / 196) for d in DEFAULT_DELTAS[:-1]] + [spec.n_tokens - 3]


# ------------------------------------------------------------------------------------------------ helpers
def parse_number(text):
    """Float or exact fraction ('1/3')."""
    return float(Fraction(text))


def load_json(path):
    return json.loads(Path(path).read_text())


def save_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(obj, indent=1, default=float) + '\n'); tmp.replace(path)


def require_idle_card(args):
    """Refuse a GPU that another process uses (accuracy and timing both assume an idle card)."""
    if getattr(args, 'allow_busy', False): return
    from leaf.evaluation.runner import require_idle_gpu
    require_idle_gpu(idle=True)


def resolve_target(args):
    """(backbone, config, timm_id, label) selected by --schedule/--key, --config or --dense."""
    if args.schedule:
        from leaf.api import load_schedule
        if not args.key: raise SystemExit('--key is required with --schedule (see: python -m leaf schedules FILE)')
        doc, entry = load_schedule(args.schedule, args.key)
        if args.backbone and args.backbone != doc['backbone']:
            raise SystemExit(f"--backbone {args.backbone} does not match the schedule file ({doc['backbone']})")
        return doc['backbone'], entry['config'], args.timm_id or doc.get('timm_id'), args.key
    if not args.backbone: raise SystemExit('--backbone is required with --config / --dense')
    if args.dense: return args.backbone, dict(kind='dense'), args.timm_id, 'dense'
    if args.config:
        cfg = load_json(args.config) if Path(args.config).exists() else json.loads(args.config)
        return args.backbone, cfg, args.timm_id, 'config'
    raise SystemExit('give --schedule FILE --key KEY, --config JSON or --dense')


def add_target_arguments(p):
    g = p.add_argument_group('what to run')
    g.add_argument('--schedule', help='schedule file: schedules/leaf/<bb>.json (released) or assets/schedules/<bb>.json (every schedule of the paper)')
    g.add_argument('--key', help='schedule key in the file, e.g. latency/DR10.0')
    g.add_argument('--config', help='a config as JSON text or a JSON file')
    g.add_argument('--dense', action='store_true', help='the dense backbone')
    g.add_argument('--backbone', help='a name of `leaf backbones` (deit_b, deit-b, DeiT-B, a timm id, ...); implied by --schedule')
    g.add_argument('--timm-id', help='override the checkpoint (default: the one recorded in the schedule file)')


def add_data_arguments(p):
    p.add_argument('--data-cache', default='data/cache', help='directory of the uint8 caches (default data/cache)')
    p.add_argument('--teacher-cache', help='directory for dense teacher logits (default <data-cache>/teachers)')


def make_runner(args, backbone, timm_id, root):
    from leaf.api import LeafRunner
    return LeafRunner(backbone, root=root, batch=args.batch, data_cache=args.data_cache,
                      teacher_cache=args.teacher_cache, timm_id=timm_id)


def forecast_config(groups, G, rank):
    from leaf.planner.configs import ForecastConfig
    return ForecastConfig(groups, G=G, rank=None if str(rank).lower() == 'full' else int(rank))


# ------------------------------------------------------------------------------------------------ commands
def cmd_backbones(args):
    from leaf.backbones import REGISTRY, SPECS, _registry_files
    if args.json:
        print(json.dumps({n: dict(timm_id=s.timm_id, aliases=list(s.aliases), depth=s.depth, dim=s.dim, heads=s.heads, pool=s.pool,
                                  weights=s.weights, diffrate=s.diffrate_key) for n, s in SPECS.items()}, indent=1)); return
    print(f"{'name':10s} {'timm checkpoint':45s} {'blocks':>6s} {'dim':>5s} {'heads':>5s}  aliases")
    for n, s in SPECS.items():
        print(f'{n:10s} {s.timm_id:45s} {s.depth:6d} {s.dim:5d} {s.heads:5d}  {", ".join(s.aliases)}')
    print(f"\nregistry: {', '.join(str(p) for p in _registry_files())}")


def cmd_schedules(args):
    from leaf.api import load_schedule
    doc, _ = load_schedule(args.file)
    keys = [args.key] if args.key else list(doc['schedules'])
    if args.json:
        print(json.dumps({k: doc['schedules'][k] for k in keys}, indent=1)); return
    print(f"{doc['backbone']} ({doc.get('timm_id')})  dense: {doc.get('dense')}")
    width = max(18, *(len(k) for k in keys))
    print(f"{'key':{width}s} {'budget':>9s} {'ms bs32':>8s} {'val':>7s} {'V2':>7s}  stations (block: tokens after, theta, alpha)")
    num = lambda v, d: '' if not isinstance(v, (int, float)) else f'{v:.{d}f}'
    for k in keys:
        e = doc['schedules'][k]; m = e.get('measured', {})
        if 'stations' in e:
            st = '  '.join(f"{s['block']}:{s['tokens_after']},{s['theta']:.2g},{s['alpha']:.3g}" for s in e['stations'])
        else:
            st = json.dumps({a: b for a, b in e['config'].items() if a != 'kind'}) + f"  ({e['config']['kind']})"
        b = e.get('budget') or {}; bv = b.get('value')
        ms = m.get('ms_bs32', (m.get('latency_gpu') or {}).get('ms_bs32'))
        val = m.get('val', m.get('val_miou'))      # segmentation schedules: mIoU on the validation set
        print(f"{k:{width}s} {num(bv, 3):>9s} {num(ms, 3):>8s} {num(val, 2):>7s} {num(m.get('v2'), 2):>7s}  {st}")


def cmd_evaluate(args):
    from leaf.api import summary
    backbone, cfg, timm_id, label = resolve_target(args)
    require_idle_card(args)
    r = make_runner(args, backbone, timm_id, Path(args.out) / backbone)
    r.limit = args.limit
    t0 = time.monotonic()
    log('evaluate', f'{backbone} {label}: {args.split} at batch {args.batch}' + (f', first {args.limit} images' if args.limit else ''))
    arr = r.run(cfg, args.split)
    if arr is None: raise SystemExit('non-finite logits: the config is infeasible in BF16')
    row = dict(backbone=backbone, schedule=label, split=args.split, batch=args.batch, **summary(arr),
               cache=str(r.path(cfg, args.split).with_suffix('.npz')))
    paper = paper_numbers(args).get(args.split)
    if paper is not None and args.batch == 32 and not args.limit: row['paper_top1'] = paper
    if 'top1' in row:
        shown = f"top-1 {row['top1']:.3f}%" + (f" (paper {paper:.3f}%)" if 'paper_top1' in row else '')
    else:                                     # segmentation: agreement with the dense argmax, mIoU on val
        shown = f"agreement with dense {row['agree']:.2f}%" + (f", mIoU {row['miou']:.2f}" if 'miou' in row else '')
    log('evaluate', shown + f", KL {row['kl']:.4f}, {duration(time.monotonic() - t0)}")
    print(json.dumps(row))


def paper_numbers(args):
    """The paper's numbers of a --schedule entry (released files and assets/schedules books), or {}."""
    if not args.schedule: return {}
    from leaf.api import load_schedule
    return load_schedule(args.schedule, args.key)[1].get('measured') or {}


def cmd_benchmark(args):
    import gc
    import torch
    from leaf.latency.timing import measure_latency
    backbone, cfg, timm_id, label = resolve_target(args)
    require_idle_card(args)
    r = make_runner(args, backbone, timm_id, Path(args.out) / backbone)
    model = r.inference_model(cfg)
    log('benchmark', f'{backbone} {label} on {torch.cuda.get_device_name(0)}: batch {", ".join(map(str, args.batches))}, '
                     f'{args.rounds} rounds of {args.iters} CUDA-graph replays, each after a warm-up')
    rounds = []
    for i in range(args.rounds):
        rounds.append({str(bs): measure_latency(model, bs, iters=args.iters) for bs in args.batches})
        log('benchmark', f'round {i + 1}/{args.rounds}: ' + ', '.join(f"bs{bs} {m['ms_mean']:.3f} ms" for bs, m in rounds[-1].items()))
    out = dict(backbone=backbone, schedule=label, device=torch.cuda.get_device_name(0), rounds=rounds,
               ms_mean={bs: round(sum(x[bs]['ms_mean'] for x in rounds) / len(rounds), 3) for bs in rounds[0]})
    paper = paper_numbers(args)
    paper_ms = paper.get('ms_bs32', (paper.get('latency_gpu') or {}).get('ms_bs32'))
    if paper_ms is not None and '32' in out['ms_mean']:
        out['paper_ms_bs32'] = paper_ms                   # an RTX 5060 Ti, the whole machine idle
        log('benchmark', f"bs32 {out['ms_mean']['32']:.3f} ms, paper {paper_ms:.3f} ms ({100 * (out['ms_mean']['32'] / paper_ms - 1):+.1f}%)")
    print(json.dumps(out))
    del model; gc.collect(); torch.cuda.empty_cache()


def cmd_data(args, rest):
    if args.action == 'download':
        from leaf.data.download import main
    elif args.action == 'clip-text':
        from leaf.data.clip_text import main
    elif args.action == 'ade20k':
        from leaf.data.ade20k import main
    else:
        from leaf.data.build_cache import main
    main(rest)


def cmd_group_heads(args):
    import functools
    import leaf.forecast.head_groups as grouping
    from leaf.api import load_backbone
    from leaf.evaluation.data import load_split
    require_idle_card(args)
    grouping.load_split = functools.partial(load_split, cache_dir=args.data_cache)   # the cache of this command
    vit, spec = load_backbone(args.backbone, args.timm_id)
    log('group-heads', f'{args.backbone}: grouping (block, head) rows on {args.n} calibration images'); t0 = time.monotonic()
    split = 'calib'
    if spec.task == 'segmentation':           # the dataset's grouping windows (leaf/data/ade20k.py)
        from leaf.evaluation.segmentation import base_split
        split = base_split(spec, 'calib')
    grouping.group_heads(vit, spec, args.out, n=args.n, split=split)
    log('group-heads', f'wrote {args.out} ({duration(time.monotonic() - t0)})')
    print(json.dumps(dict(backbone=args.backbone, groups=args.out, images=args.n)))


def probe_jobs(spec, dcfg, thetas, alphas, deltas):
    """Single-station 'pda' configs, built like the paper's probes (pure pruning keeps the plain operator)."""
    from leaf.planner.alpha import to_alpha_config
    from leaf.planner.configs import config_from_state
    from leaf.planner.plan import operator_config
    from leaf.planner.probes import probe_state
    for l in range(1, spec.depth):
        for delta in deltas:
            n = spec.n_tokens - delta
            if n < 3: continue
            for th in thetas:
                for a in alphas:
                    if l == spec.depth - 1 and a > 0: continue          # no later block, no forecast
                    cfg = config_from_state(probe_state(l, n, th, a), spec, dcfg)   # 'k' field holds alpha
                    if th != 0: cfg = operator_config(cfg, 'M3')
                    yield dict(block=l, count=n, theta=th, alpha=a, config=to_alpha_config(cfg))


def cmd_probe(args):
    from leaf.backbones import SPECS
    from leaf.evaluation.runner import load_json as read
    spec = SPECS[args.backbone]
    args.deltas = args.deltas or default_deltas(spec)
    jobs = list(probe_jobs(spec, forecast_config(args.groups, args.G, args.rank), args.thetas, args.alphas, args.deltas))
    i, n = (int(v) for v in args.shard.split('/'))
    mine = jobs[i::n]
    from leaf.api import LeafRunner
    paths = LeafRunner(args.backbone, root=Path(args.out) / 'runs')          # cache paths only (no model, no GPU)
    for src in args.reuse or ():        # link runs of these probe folders instead of running them again
        src_paths = LeafRunner(args.backbone, root=Path(src) / 'runs')
        for job in jobs:
            new, old = paths.path(job['config'], 'half'), src_paths.path(job['config'], 'half')
            if new.with_suffix('.json').exists() or not old.with_suffix('.json').exists(): continue
            new.parent.mkdir(parents=True, exist_ok=True)
            for ext in ('.npz', '.json'):           # metadata last: it marks the run as done
                if old.with_suffix(ext).exists(): os.symlink(old.with_suffix(ext).resolve(), new.with_suffix(ext))
    todo = [job for job in mine if not paths.path(job['config'], 'half').with_suffix('.json').exists()]
    log('probe', f'{args.backbone}: {len(jobs)} probes in the grid, {len(mine)} in shard {args.shard}, '
                 f'{len(mine) - len(todo)} cached, {len(todo)} to run')
    if args.dry_run: return
    if todo:
        require_idle_card(args); r = make_runner(args, args.backbone, args.timm_id, Path(args.out) / 'runs')
        progress = Progress('probe', len(todo), 'probes')
        for job in todo: r.run(job['config'], 'half'); progress.step()
    # Write probes.json once every probe is done (by any shard).
    records = []
    for job in jobs:
        p = paths.path(job['config'], 'half').with_suffix('.json')
        if not p.exists():
            log('probe', f'not complete yet ({p.name} missing); rerun when every shard has finished'); return
        row = read(p)
        if row.get('feasible') is False: raise SystemExit(f'infeasible probe {job}')
        records.append(dict({k: v for k, v in job.items() if k != 'config'}, kl=row['kl']))
    # Probes of earlier grids in the same folder are kept, in their order; new probes are appended.
    dest = Path(args.out) / 'probes.json'
    forecast = dict(G=args.G, rank=args.rank)
    grids = [dict(thetas=args.thetas, alphas=args.alphas, deltas=args.deltas)]
    if dest.exists():
        old = load_json(dest)
        if old.get('backbone') != args.backbone or old.get('forecast') != forecast or old.get('batch') != args.batch:
            raise SystemExit(f'{dest} holds probes of another backbone, forecast or batch; use another --out')
        have = {(r['block'], r['count'], r['theta'], r['alpha']) for r in old['records']}
        records = old['records'] + [r for r in records if (r['block'], r['count'], r['theta'], r['alpha']) not in have]
        grids = old.get('grids', []) + [g for g in grids if g not in old.get('grids', [])]
    images = 'calib_bal even indices' if spec.task == 'classification' else f'{spec.dataset}_calib windows (per-pixel KL)'
    save_json(dest, dict(backbone=args.backbone, images=images, batch=args.batch,
                         forecast=forecast, grids=grids, records=records))
    log('probe', f'wrote {dest} ({len(records)} probes)')


def time_configs(r, configs, batch, rounds, iters):
    """[{ms, sd, rounds}] per config: every round times every config once (each with its own warm-up, inside
    measure_latency); odd rounds run in reverse order, as in the paper's timing sessions."""
    import gc
    import numpy as np
    import torch
    from leaf.latency.timing import measure_latency
    times = {i: [] for i in range(len(configs))}
    log('timing', f'{len(configs)} configs x {rounds} rounds at batch {batch} on {torch.cuda.get_device_name(0)} '
                  f'(odd rounds reversed; every measurement warms up first)')
    progress = Progress('timing', len(configs) * rounds, 'measurements', every=max(1, len(configs) // 2))
    for rnd in range(rounds):
        order = list(range(len(configs)))
        if rnd % 2: order.reverse()                   # middle round reversed, as in the paper
        for i in order:
            model = r.inference_model(configs[i])
            times[i].append(measure_latency(model, batch, iters=iters)['ms_mean'])
            del model; gc.collect(); torch.cuda.empty_cache(); progress.step()
    return [dict(ms=float(np.mean(times[i])), sd=float(np.std(times[i], ddof=1)) if len(times[i]) > 1 else 0.,
                 rounds=times[i]) for i in range(len(configs))]


def cmd_time_workloads(args):
    import torch
    require_idle_card(args)
    rows = load_json(args.workloads)
    rows = rows['rows'] if isinstance(rows, dict) else rows
    r = make_runner(args, args.backbone, args.timm_id, Path('runs') / args.backbone)
    timed = time_configs(r, [row['config'] for row in rows], args.batch, args.rounds, args.iters)
    out = [dict(dict(name=row['name']) if 'name' in row else {}, config=row['config'], group=row.get('group'), **t)
           for row, t in zip(rows, timed)]
    save_json(args.out, dict(backbone=args.backbone, batch=args.batch, rounds=args.rounds,
                             device=torch.cuda.get_device_name(0), time=time.time(), rows=out))
    log('time-workloads', f'wrote {args.out} ({len(out)} workloads)')


def cmd_budgets(args):
    import torch
    from leaf.backbones import SPECS
    from leaf.planner.alpha import config_gmac, to_alpha_config
    from leaf.planner.budgets import anchor_budgets, dense_budgets
    from leaf.planner.configs import diffrate_config, diffrate_schedules
    spec = SPECS[args.backbone]
    kind = args.anchors
    if kind == 'auto': kind = 'diffrate' if diffrate_schedules(spec) else 'dense'
    anchors = {}                                     # id -> config
    if kind == 'diffrate':
        if not diffrate_schedules(spec): raise SystemExit(f'DiffRate has no released schedules for {args.backbone}; use --anchors dense or gmac')
        anchors = {f'DR{b}': diffrate_config(spec, b) for b in sorted(diffrate_schedules(spec), key=float)}
    elif kind == 'gmac':
        if not (args.gmac_points and args.probes and args.groups):
            raise SystemExit('--anchors gmac needs --gmac-points, --probes and --groups')
        from leaf.planner.gmac import GmacPlanner
        planner = GmacPlanner(spec, forecast_config(args.groups, args.G, args.rank), load_probe_curves(args.probes, spec), 'M3')
        for g in args.gmac_points:
            res = planner.plan(g)
            if 'config' not in res: raise SystemExit(f'no schedule within {g:g} GMAC')
            anchors[f'gmac{g:g}'] = to_alpha_config(res['config'])
    elif kind == 'configs':
        if not args.anchor_configs: raise SystemExit('--anchors configs needs --anchor-configs FILE ({id: config})')
        anchors = load_json(args.anchor_configs)
    require_idle_card(args)
    r = make_runner(args, args.backbone, args.timm_id, Path('runs') / args.backbone)
    ids = ['dense'] + list(anchors)
    timed = dict(zip(ids, time_configs(r, [dict(kind='dense')] + list(anchors.values()), args.batch, args.rounds, args.iters)))
    measured = {i: dict(timed[i], gmac=config_gmac(anchors[i], spec) if i in anchors else None,
                        config=anchors.get(i, dict(kind='dense'))) for i in ids}
    if kind == 'dense':
        budgets = dense_budgets(timed['dense']['ms'], args.dense_fractions)
    else:
        budgets = anchor_budgets({i: dict(ms=timed[i]['ms'], gmac=measured[i]['gmac']) for i in anchors}, args.factor, args.extras)
    save_json(args.out, dict(backbone=args.backbone, batch=args.batch, rounds=args.rounds, device=torch.cuda.get_device_name(0),
                             anchors=kind, factor=None if kind == 'dense' else args.factor,
                             dense_fractions=list(args.dense_fractions) if kind == 'dense' else None,
                             measured=measured, budgets=budgets, time=time.time()))
    for i in ids: print(f"{i:12s} {timed[i]['ms']:8.3f} ms (sd {timed[i]['sd']:.3f})" + (f"  {measured[i]['gmac']:.2f} GMAC" if measured[i]['gmac'] else ''))
    for b in budgets: print(f"budget {b['id']:12s} {b['latency_ms']:8.3f} ms")
    log('budgets', f'wrote {args.out} ({len(budgets)} budgets)')


def cmd_workloads(args):
    from leaf.backbones import SPECS
    from leaf.latency.workloads import GMAC_FRACTIONS, make_workloads
    spec = SPECS[args.backbone]
    close = lambda grid: None if grid is None else [float(v) for v in grid]
    curves = load_probe_curves(args.probes, spec, close(args.thetas), close(args.alphas))
    rows = make_workloads(spec, forecast_config(args.groups, args.G, args.rank), curves,
                          fractions=args.gmac_fractions or GMAC_FRACTIONS, diffrate=not args.no_diffrate)
    save_json(args.out, dict(backbone=args.backbone, probes=[str(p) for p in args.probes], groups=str(args.groups),
                             note='GMAC-planned schedules (5 variants) and DiffRate schedules; time with leaf time-workloads',
                             rows=rows))
    log('workloads', f'wrote {args.out} ({len(rows)} workloads)')


def cmd_fit_latency(args):
    from leaf.backbones import SPECS
    from leaf.evaluation.runner import config_hash
    from leaf.latency.salt import FORMULAS, features, fit_salt
    from leaf.planner.alpha import kappa_view
    spec = SPECS[args.backbone]
    doc = load_json(args.timings); rows = doc['rows'] if isinstance(doc, dict) else doc
    data = []
    for row in rows:
        cfg = kappa_view(row['config'])
        group = row.get('group') or config_hash({k: v for k, v in cfg.items() if k not in ('kind', 'merge')})
        data.append(dict(config=cfg, ms=row['ms'], group=group, features=features(cfg, spec)))
    model = fit_salt(data, args.formula)
    model.update(backbone=args.backbone, source=str(args.timings),
                 training_range={k: [min(r['features'].get(k, 0) for r in data), max(r['features'].get(k, 0) for r in data)]
                                 for k in FORMULAS[args.formula]})
    save_json(args.out, model)
    print(json.dumps(dict(out=args.out, n=model['n'], groups=model['groups'], ridge=model['alpha'],
                          coef=dict(zip(model['feature_names'], [round(c, 5) for c in model['coef']])),
                          intercept=round(model['intercept'], 5))))


def load_probe_curves(paths, spec, thetas=None, alphas=None):
    """{(theta, alpha): ProbeCurve} from probe files, in file / record order (the DP breaks ties by this order)."""
    from leaf.planner.surrogate import ProbeCurve
    rows = {}
    for path in paths:
        doc = load_json(path); records = doc['records'] if isinstance(doc, dict) else doc
        for a in records:
            th, al = float(a['theta']), float(a['alpha'])
            if thetas is not None and th not in thetas: continue
            if alphas is not None and al not in alphas: continue
            rows.setdefault((th, al), {}).setdefault(a['block'], {})[spec.n_tokens - a['count']] = a['kl']
    return {k: ProbeCurve(v, spec.n_tokens) for k, v in rows.items()}


def cmd_plan(args):
    from leaf.api import stations
    from leaf.backbones import SPECS
    from leaf.evaluation.runner import config_hash
    from leaf.planner.alpha import config_gmac, to_alpha_config, plan_under_latency
    spec = SPECS[args.backbone]
    close = lambda grid: None if grid is None else [float(v) for v in grid]
    curves = load_probe_curves(args.probes, spec, close(args.thetas), close(args.alphas))
    d = forecast_config(args.groups, args.G, args.rank)
    budgets = []
    if args.budget_ms: budgets += [dict(kind='latency', id=f'{v:g}ms', value=v) for v in args.budget_ms]
    if args.gmac: budgets += [dict(kind='gmac', id=f'{v:g}gmac', value=v) for v in args.gmac]
    if args.budgets:
        doc = load_json(args.budgets)
        for b in doc['budgets'] if isinstance(doc, dict) else doc:
            budgets.append(dict(kind='latency', id=b['id'], value=b['latency_ms']))
    if not budgets: raise SystemExit('give --budget-ms, --gmac or --budgets')
    salt = load_json(args.salt) if args.salt else None
    out = dict(backbone=args.backbone, probes=[str(p) for p in args.probes], groups=str(args.groups),
               salt=None if salt is None else str(args.salt), forecast=dict(G=args.G, rank=args.rank),
               operator=args.operator, schedules={})
    for b in budgets:
        t0 = time.monotonic()
        if b['kind'] == 'latency':
            if salt is None: raise SystemExit('a latency budget needs --salt')
            res = plan_under_latency(curves, spec, d, salt, b['value'], args.operator, True)
            key = f"latency/{b['id']}"
        else:
            from leaf.planner.gmac import GmacPlanner
            res = GmacPlanner(spec, d, curves, args.operator).plan(b['value'])
            if 'config' in res: res = dict(res, config=to_alpha_config(res['config']))
            key = f"gmac/{b['id']}"
        if 'config' not in res:
            out['schedules'][key] = dict(budget=b, status='no_feasible_candidate'); log('plan', f'{key}: no feasible schedule'); continue
        cfg = res['config']
        out['schedules'][key] = dict(budget=dict(kind=b['kind'], value=b['value'], id=b['id']), stations=stations(cfg, spec),
                                     config=cfg, config_sha=config_hash(cfg),
                                     planner=dict(fhat=res.get('fhat'), predicted_ms=res.get('predicted_ms'),
                                                  gmac=config_gmac(cfg, spec), seconds=round(time.monotonic() - t0, 2)))
        st = ' '.join(f"{s['block']}:{s['tokens_after']}" for s in out['schedules'][key]['stations'])
        pred = res.get('predicted_ms')
        log('plan', f"{key:20s} K={len(out['schedules'][key]['stations'])} {st:40s} F^={res['fhat']:.4f}"
              + (f' predicted {pred:.2f} ms' if pred else '') + f" gmac {out['schedules'][key]['planner']['gmac']:.2f}")
    out.update(timm_id=args.timm_id)
    save_json(args.out, out)
    log('plan', f"wrote {args.out} ({len(out['schedules'])} schedules)")


STEPS = ('group-heads', 'probe', 'budgets', 'workloads', 'time-workloads', 'fit-latency', 'plan')


def cmd_search(args):
    """Run the search as separate `python -m leaf <command>` processes, so that each GPU step starts on a clean card
    and the log lists commands one can run by hand."""
    ws = Path(args.workspace); ws.mkdir(parents=True, exist_ok=True)
    gpus = [g.strip() for g in args.gpus.split(',') if g.strip()]
    steps = [s.strip() for s in args.steps.split(',')] if args.steps else list(STEPS)
    for s in steps + list(args.redo or ()):
        if s not in STEPS: raise SystemExit(f'unknown step {s}; steps: {",".join(STEPS)}')
    if args.salt: steps = [s for s in steps if s not in ('workloads', 'time-workloads', 'fit-latency')]
    if args.budgets or args.budget_ms:
        steps = [s for s in steps if s != 'budgets']
    files = dict(groups=ws / 'groups.json', probes=ws / 'probes' / 'probes.json', budgets=ws / 'budgets.json',
                 workloads=ws / 'salt_workloads.json', timings=ws / 'salt_timings.json', salt=ws / 'salt.json',
                 schedules=ws / 'schedules.json')
    groups = Path(args.groups) if args.groups else files['groups']
    salt = Path(args.salt) if args.salt else files['salt']
    budgets_file = Path(args.budgets) if args.budgets else files['budgets']
    grid = ['--thetas', *map(str, args.thetas), '--alphas', *map(str, args.alphas)]
    from leaf.backbones import SPECS, resolve
    deltas = args.deltas or default_deltas(SPECS[resolve(args.backbone)])
    forecast = ['--G', str(args.G), '--rank', str(args.rank)]
    common = (['--timm-id', args.timm_id] if args.timm_id else []) + (['--allow-busy'] if args.allow_busy else [])
    data = ['--data-cache', args.data_cache] + (['--teacher-cache', args.teacher_cache] if args.teacher_cache else [])
    timing = ['--batch', str(args.timing_batch), '--rounds', str(args.rounds), '--iters', str(args.iters)]
    log_path = ws / 'search.json'
    history = load_json(log_path) if log_path.exists() else dict(backbone=args.backbone, runs=[])
    if history['backbone'] != args.backbone: raise SystemExit(f'{ws} is a workspace of {history["backbone"]}')
    run = dict(time=time.time(), argv=sys.argv[1:], steps=[]); history['runs'].append(run)

    t_start = time.monotonic()

    def call(step, argv, gpu=None, wait=True):
        cmd = [sys.executable, '-m', 'leaf', *argv]
        env = dict(os.environ) if gpu is None else dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
        shown = ('' if gpu is None else f'CUDA_VISIBLE_DEVICES={gpu} ') + 'python -m leaf ' + shlex.join(argv)
        print(f'[{step}] {shown}', flush=True)
        run['steps'].append(dict(step=step, command=shown, time=time.time())); save_json(log_path, history)
        if args.dry_run: return None
        proc = subprocess.Popen(cmd, env=env)
        if not wait: return proc
        t0 = time.monotonic()
        if proc.wait(): raise SystemExit(f'step {step} failed (exit {proc.returncode}): {shown}')
        log('search', f'{step} done in {duration(time.monotonic() - t0)} (total {duration(time.monotonic() - t_start)})')

    def todo(step, output):
        if step not in steps: return False
        if output.exists() and step not in (args.redo or ()):
            print(f'[{step}] {output} exists, skipped (--redo {step} to run it again)', flush=True); return False
        return True

    if todo('group-heads', groups):
        call('group-heads', ['group-heads', '--backbone', args.backbone, '--out', str(groups), '--n', str(args.group_images),
                       '--data-cache', args.data_cache, *common], gpus[0])
    if 'probe' in steps:          # always run: finished probes are cached, so a rerun only adds new grid points
        argv = ['probe', '--backbone', args.backbone, '--groups', str(groups), '--out', str(files['probes'].parent),
                *grid, '--deltas', *map(str, deltas), *forecast, '--batch', str(args.batch), *data, *common]
        if args.reuse_probes: argv += ['--reuse', *args.reuse_probes]
        if len(gpus) > 1:         # one shard per GPU, then one pass that writes probes.json
            procs = [call('probe', argv + ['--shard', f'{i}/{len(gpus)}'], g, wait=False) for i, g in enumerate(gpus)]
            for proc in procs:
                if proc is not None and proc.wait(): raise SystemExit('a probe shard failed')
        call('probe', argv, gpus[0])
    probes = [str(files['probes'])] + list(args.extra_probes or ())
    if todo('budgets', budgets_file):
        argv = ['budgets', '--backbone', args.backbone, '--out', str(budgets_file), '--anchors', args.anchors,
                '--factor', str(args.factor), '--extras', str(args.extras), '--dense-fractions', *map(str, args.dense_fractions),
                *timing, *data, *common]
        if args.anchors == 'gmac':
            argv += ['--gmac-points', *map(str, args.gmac_points or ()), '--probes', *probes, '--groups', str(groups), *forecast]
        if args.anchors == 'configs': argv += ['--anchor-configs', str(args.anchor_configs)]
        call('budgets', argv, gpus[0])
    workloads = Path(args.workloads) if args.workloads else files['workloads']
    if not args.workloads and todo('workloads', workloads):
        call('workloads', ['workloads', '--backbone', args.backbone, '--probes', *probes, '--groups', str(groups),
                           *grid, *forecast, '--out', str(workloads)])
    if todo('time-workloads', files['timings']):
        call('time-workloads', ['time-workloads', '--backbone', args.backbone, '--workloads', str(workloads), '--out', str(files['timings']),
                      *timing, *data, *common], gpus[0])
    if todo('fit-latency', salt):
        call('fit-latency', ['fit-latency', '--backbone', args.backbone, '--timings', str(files['timings']), '--out', str(salt)])
    if todo('plan', files['schedules']):
        argv = ['plan', '--backbone', args.backbone, '--probes', *probes, '--groups', str(groups), *grid, *forecast,
                '--out', str(files['schedules'])] + (['--timm-id', args.timm_id] if args.timm_id else [])
        if args.budget_ms: argv += ['--budget-ms', *map(str, args.budget_ms)]
        if args.gmac: argv += ['--gmac', *map(str, args.gmac)]
        if not args.budget_ms and (args.budgets or 'budgets' in steps or budgets_file.exists()):
            argv += ['--budgets', str(budgets_file)]
        if '--budget-ms' in argv or '--budgets' in argv: argv += ['--salt', str(salt)]
        call('plan', argv)
    if not args.dry_run and files['schedules'].exists():
        print(f"schedules: {files['schedules']}  (leaf schedules {files['schedules']}; "
              f"leaf evaluate | benchmark --schedule {files['schedules']} --key <key>)")


# ------------------------------------------------------------------------------------------------ parser
def build_parser():
    ap = argparse.ArgumentParser(prog='python -m leaf', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True, metavar='command')

    p = sub.add_parser('backbones', help='list the backbones and the names the CLI accepts (leaf/backbones.json)')
    p.add_argument('--json', action='store_true')
    sub.add_parser('experiments', help="the paper's experiments: list | run <experiment> <stage>")
    sub.add_parser('report', help="the paper's analyses, tables, figures and assets from results/: analyze | tables | figures | assets | all")
    p = sub.add_parser('schedules', help='list the schedules of a schedule file')
    p.add_argument('file'); p.add_argument('--key'); p.add_argument('--json', action='store_true', help='print the full entries')

    p = sub.add_parser('evaluate', help='top-1 and KL to the dense model of a schedule on a cached split')
    add_target_arguments(p); add_data_arguments(p)
    p.add_argument('--split', default='val', choices=('val', 'v2', 'half', 'odd', 'calib', 'calib_bal', 'confirm'))
    p.add_argument('--batch', type=int, default=32, help='evaluation batch (the paper uses 32; BF16 KL depends on it)')
    p.add_argument('--limit', type=int, default=0, help='first N images only (smoke test, cached separately)')
    p.add_argument('--out', default='runs/leaf', help='result cache (per backbone, split and config)')
    p.add_argument('--allow-busy', action='store_true')

    p = sub.add_parser('benchmark', help='latency of a schedule (CUDA graph, BF16)')
    add_target_arguments(p); add_data_arguments(p)
    p.add_argument('--batches', type=int, nargs='+', default=[32])
    p.add_argument('--rounds', type=int, default=3); p.add_argument('--iters', type=int, default=200)
    p.add_argument('--batch', type=int, default=32, help=argparse.SUPPRESS)
    p.add_argument('--out', default='runs/leaf', help=argparse.SUPPRESS)
    p.add_argument('--allow-busy', action='store_true')

    p = sub.add_parser('data', help='download ImageNet / prepare the evaluation caches (other options pass through)')
    p.add_argument('action', choices=('download', 'prepare', 'clip-text', 'ade20k'),
                   help='download: --out DIR; prepare: --split {val,calib,calib_bal,confirm,v2,flickr_test,coco_test} [--data-dir] '
                        '[--out-dir] [--v2-tar] [--retrieval-dir]; clip-text: --backbone clip_b16 --task {imagenet,flickr_test,coco_test,all}; '
                        'ade20k: [--download] --split {train,val,all} (segmentation windows)')

    p = sub.add_parser('group-heads', help='head groups of the forecast D')
    p.add_argument('--backbone', required=True); p.add_argument('--out', required=True, help='groups.json to write')
    p.add_argument('--n', type=int, default=1024, help='calibration images (default 1024)')
    p.add_argument('--timm-id'); p.add_argument('--data-cache', default='data/cache'); p.add_argument('--allow-busy', action='store_true')

    p = sub.add_parser('probe', help='single-station probes (KL to dense on 5,000 calibration images)')
    p.add_argument('--backbone', required=True); p.add_argument('--groups', required=True, help='groups.json of `leaf group-heads`')
    p.add_argument('--out', required=True, help='directory: runs/ cache and probes.json')
    p.add_argument('--thetas', type=parse_number, nargs='+', default=list(DEFAULT_THETAS))
    p.add_argument('--alphas', type=parse_number, nargs='+', default=list(DEFAULT_ALPHAS), help="fractions allowed, e.g. 1/3")
    p.add_argument('--deltas', type=int, nargs='+', help='tokens removed by the probed station (default: 8 24 ... 194 for '
                                                           '197 tokens, scaled to the backbone\'s token count)')
    p.add_argument('--G', type=int, default=16); p.add_argument('--rank', default='32', help="integer or 'full'")
    p.add_argument('--shard', default='0/1', help='i/n: this process runs every n-th probe (one process per GPU)')
    p.add_argument('--reuse', nargs='+', help='other probe folders (--out of an earlier leaf probe, or results/probe_runs_alpha/<bb>): '
                                             'their runs of the same configs are linked, not run again')
    p.add_argument('--batch', type=int, default=32); p.add_argument('--timm-id'); add_data_arguments(p)
    p.add_argument('--dry-run', action='store_true', help='only count the probes'); p.add_argument('--allow-busy', action='store_true')

    p = sub.add_parser('budgets', help='time anchor schedules on this GPU and derive latency budgets')
    p.add_argument('--backbone', required=True); p.add_argument('--out', required=True, help='budgets file (input of leaf plan --budgets)')
    p.add_argument('--anchors', default='auto', choices=('auto', 'diffrate', 'gmac', 'dense', 'configs'),
                   help="auto: DiffRate's released schedules when the backbone has them (the paper), else dense fractions; "
                        'gmac: LEAF schedules GMAC-planned at --gmac-points; configs: --anchor-configs')
    p.add_argument('--gmac-points', type=float, nargs='+', help='GMAC points of the gmac anchors')
    p.add_argument('--probes', nargs='+', help='probe files (gmac anchors)'); p.add_argument('--groups', help='groups.json (gmac anchors)')
    p.add_argument('--G', type=int, default=16); p.add_argument('--rank', default='32')
    p.add_argument('--anchor-configs', help='JSON file {id: config} of anchor schedules')
    p.add_argument('--factor', type=float, default=.97, help='budget = factor x measured anchor latency (default 0.97)')
    p.add_argument('--extras', type=int, default=2, help='budgets below the fastest anchor, spaced by the mean anchor gap (default 2)')
    p.add_argument('--dense-fractions', type=float, nargs='+', default=[.5, .56, .62, .68, .74, .8, .86],
                   help='budgets as fractions of the dense latency (--anchors dense)')
    p.add_argument('--batch', type=int, default=32, help='timing batch (default 32, as in the paper)')
    p.add_argument('--rounds', type=int, default=3); p.add_argument('--iters', type=int, default=200)
    p.add_argument('--timm-id'); add_data_arguments(p); p.add_argument('--allow-busy', action='store_true')

    p = sub.add_parser('workloads', help='timing workloads for fit-latency, GMAC-planned from the probes')
    p.add_argument('--backbone', required=True); p.add_argument('--probes', required=True, nargs='+')
    p.add_argument('--groups', required=True); p.add_argument('--out', required=True)
    p.add_argument('--gmac-fractions', type=float, nargs='+', help='GMAC points as fractions of the dense GMAC (default 0.35 ... 0.84, 8 points)')
    p.add_argument('--no-diffrate', action='store_true', help="leave out DiffRate's released schedules")
    p.add_argument('--thetas', type=parse_number, nargs='+'); p.add_argument('--alphas', type=parse_number, nargs='+')
    p.add_argument('--G', type=int, default=16); p.add_argument('--rank', default='32')

    p = sub.add_parser('time-workloads', help='time configs on this GPU for fit-latency')
    p.add_argument('--backbone', required=True)
    p.add_argument('--workloads', required=True, help='e.g. schedules/leaf/planning/<bb>/salt_workloads.json, or a JSON list of {config}')
    p.add_argument('--out', required=True); p.add_argument('--batch', type=int, default=32)
    p.add_argument('--rounds', type=int, default=3); p.add_argument('--iters', type=int, default=200)
    p.add_argument('--timm-id'); add_data_arguments(p); p.add_argument('--allow-busy', action='store_true')

    p = sub.add_parser('fit-latency', help='fit the latency model SALT (non-negative least squares, grouped-CV ridge) to timed workloads')
    p.add_argument('--backbone', required=True); p.add_argument('--timings', required=True, help='output of time-workloads')
    p.add_argument('--out', required=True); p.add_argument('--formula', default='station_rich')

    p = sub.add_parser('plan', help='plan schedules under latency (SALT) or GMAC budgets')
    p.add_argument('--backbone', required=True)
    p.add_argument('--probes', required=True, nargs='+', help='probe files, in order (ties are broken by this order)')
    p.add_argument('--groups', required=True); p.add_argument('--salt', help='SALT model (leaf fit-latency) for latency budgets')
    p.add_argument('--budget-ms', type=float, nargs='+', help='latency budgets (ms per batch, the SALT batch size)')
    p.add_argument('--budgets', help="budgets file: [{id, latency_ms}, ...] (e.g. schedules/leaf/planning/<bb>/budgets.json)")
    p.add_argument('--gmac', type=float, nargs='+', help='GMAC budgets (no latency model)')
    p.add_argument('--thetas', type=parse_number, nargs='+', help='restrict the merge-share grid')
    p.add_argument('--alphas', type=parse_number, nargs='+', help='restrict the forecast-weight grid (fractions allowed)')
    p.add_argument('--operator', default='M3', choices=('M3', 'M0'),
                   help='merge operator of the schedules: M3 (MLERP, the method) or M0 (average); use probes of the same '
                        'operator, e.g. schedules/leaf/planning/<bb>/probes_m0.json for M0')
    p.add_argument('--G', type=int, default=16); p.add_argument('--rank', default='32')
    p.add_argument('--timm-id', help='recorded in the output'); p.add_argument('--out', required=True)

    p = sub.add_parser('search', help='the whole schedule search in one workspace folder',
                       description='Search LEAF schedules for a backbone on this machine, in one workspace folder:\n'
                                   '  group -> probe -> budgets -> workloads -> time -> fit -> plan\n'
                                   'Outputs: groups.json, probes/ (runs/ cache + probes.json), budgets.json, salt_workloads.json,\n'
                                   'salt_timings.json, salt.json, schedules.json, and search.json (the commands that were run).\n'
                                   'A step whose output exists is skipped (--redo STEP runs it again). Probes are cached one by\n'
                                   'one: a new --thetas/--alphas/--deltas grid only probes the new points, and probes.json keeps\n'
                                   'every grid probed in the workspace. Timing steps run on the first of --gpus; keep the card\n'
                                   'to themselves.', formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--backbone', required=True); p.add_argument('--workspace', required=True, help='folder for every output')
    p.add_argument('--gpus', default='0', help='GPU indices, e.g. 0,1: probes are sharded over them; timing uses the first')
    p.add_argument('--steps', help=f"comma-separated subset of {','.join(STEPS)} (default: all)")
    p.add_argument('--redo', nargs='+', choices=STEPS, help='run these steps even if their output exists')
    p.add_argument('--dry-run', action='store_true', help='print the commands only')
    g = p.add_argument_group('grid and forecast')
    g.add_argument('--thetas', type=parse_number, nargs='+', default=list(DEFAULT_THETAS), help='merge shares (default 0 1/2 1)')
    g.add_argument('--alphas', type=parse_number, nargs='+', default=list(DEFAULT_ALPHAS), help='forecast weights (default 0 1/4 1/3 1/2 2/3 3/4 1)')
    g.add_argument('--deltas', type=int, nargs='+', help='tokens removed by a probed station (default: 8 24 ... 194 for '
                                                           '197 tokens, scaled to the backbone\'s token count)')
    g.add_argument('--G', type=int, default=16); g.add_argument('--rank', default='32')
    g.add_argument('--group-images', type=int, default=1024); g.add_argument('--groups', help='use this groups.json (skips the group step)')
    g.add_argument('--extra-probes', nargs='+', help='more probe files for workloads and plan, after the workspace probes')
    g.add_argument('--reuse-probes', nargs='+', help='probe folders whose runs are linked instead of run again, e.g. '
                                                    'results/probe_runs_alpha/<bb> (the paper backbones, same groups.json)')
    g = p.add_argument_group('budgets (default: leaf budgets --anchors auto)')
    g.add_argument('--anchors', default='auto', choices=('auto', 'diffrate', 'gmac', 'dense', 'configs'))
    g.add_argument('--gmac-points', type=float, nargs='+', help='GMAC points of --anchors gmac')
    g.add_argument('--anchor-configs'); g.add_argument('--factor', type=float, default=.97); g.add_argument('--extras', type=int, default=2)
    g.add_argument('--dense-fractions', type=float, nargs='+', default=[.5, .56, .62, .68, .74, .8, .86])
    g.add_argument('--budgets', help='use this budgets file instead of timing anchors')
    g.add_argument('--budget-ms', type=float, nargs='+', help='plan at these latency budgets instead')
    g.add_argument('--gmac', type=float, nargs='+', help='also plan at these GMAC budgets')
    g = p.add_argument_group('latency model')
    g.add_argument('--salt', help='use this SALT model (skips workloads, time and fit; it must come from the same GPU)')
    g.add_argument('--workloads', help='time these workloads instead of generating them, e.g. schedules/leaf/planning/<bb>/salt_workloads.json')
    g.add_argument('--timing-batch', type=int, default=32, help='batch of the budgets and SALT (default 32)')
    g.add_argument('--rounds', type=int, default=3); g.add_argument('--iters', type=int, default=200)
    g = p.add_argument_group('data')
    g.add_argument('--batch', type=int, default=32, help='probe batch (the paper uses 32)')
    g.add_argument('--timm-id'); g.add_argument('--data-cache', default='data/cache'); g.add_argument('--teacher-cache')
    g.add_argument('--allow-busy', action='store_true')
    return ap


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    ap = build_parser()
    if argv and argv[0] in ('experiments', 'report'):   # paper reproduction (leaf/reproduction/cli.py)
        from leaf.reproduction import cli as reproduction
        return (reproduction.experiments if argv[0] == 'experiments' else reproduction.report)(argv[1:])
    if argv and argv[0] == 'data':                     # pass the remaining options to the data tools
        args, rest = ap.parse_known_args(argv)
        return cmd_data(args, rest)
    args = ap.parse_args(argv)
    if getattr(args, 'backbone', None):                  # names, aliases and timm ids of leaf/backbones.json
        from leaf.backbones import resolve
        try: args.backbone = resolve(args.backbone)
        except KeyError as e: raise SystemExit(str(e).strip('"'))
    {'backbones': cmd_backbones, 'schedules': cmd_schedules, 'evaluate': cmd_evaluate, 'benchmark': cmd_benchmark, 'group-heads': cmd_group_heads,
     'probe': cmd_probe, 'budgets': cmd_budgets, 'workloads': cmd_workloads, 'time-workloads': cmd_time_workloads,
     'fit-latency': cmd_fit_latency, 'plan': cmd_plan, 'search': cmd_search}[args.command](args)


if __name__ == '__main__':
    main()
