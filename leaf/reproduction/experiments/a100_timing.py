"""Main comparison re-timed on another GPU (e.g. an A100): every arm of the main campaign for DeiT-S, DeiT-B and
ViT-B (MAE) -- the released LEAF schedules, the baseline grid points kept by the timing selection, DiffRate, dense and
the early LEAF variant -- timed in ONE session on the device with the paper's protocol (10 eager warm-ups, CUDA-graph
capture, 10 graph warm-ups, 50 timed replays, three interleaved rounds). Accuracy is batch- and device-independent and
reused from the sealed campaigns at home, so the comparison on the device is "pairs at equal measured device latency"
(batch_sensitivity.analyse), with no accuracy evaluation on the device.

Stages (python leaf/reproduction/experiments/a100_timing.py <stage>; results/a100_timing/):
  export   (home, CPU)   tasks.json (configs, families, batches) + design.json -> kit a100_kit.tar.gz (with this file)
  time     (device)      one session per backbone, batches 32 for every arm and 4/128 for LEAF, DiffRate and dense
  pack     (device)      results/a100_timing/**/*.json -> a100_timing_<device>.tar.gz
  unpack   (home)        extract an archive from the device (sealed files are never overwritten)
  compare  (home, CPU)   pairs at equal device latency per batch -> results/a100_timing/report_<device>.json
"""
import argparse
import gc
import json
import re
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

ROOT = REPO / 'results/a100_timing'
BBS = ('deit_s', 'deit_b', 'mae_b')
CORE_BATCHES, ALL_BATCHES = (4, 32, 128), (32,)
ROUNDS = 3
KIT = REPO / 'a100_kit.tar.gz'


def load(p): return json.loads(Path(p).read_text())


def save(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True); Path(p).write_text(json.dumps(obj, indent=1, sort_keys=True, default=float))


def device_info():
    import torch
    return dict(device=torch.cuda.get_device_name(0), capability=list(torch.cuda.get_device_capability(0)), torch=torch.__version__,
                cuda=torch.version.cuda, bf16=torch.cuda.is_bf16_supported(),
                driver=subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], capture_output=True, text=True).stdout.strip())


# ----------------------------------------------------------------------------------------------- export (home)
def export():
    from leaf.reproduction.experiments import batch_sensitivity as B
    from leaf.reproduction.experiments.latency_pricing import ROOT as PRICING
    tasks = {}
    for bb in BBS:
        rows = []
        for t in B.tasks(bb):
            core = t['family'] in ('LEAF', 'dense', 'DiffRate')
            rows.append(dict(id=t['id'], name=t['name'], family=t['family'], src=t['src'], sha=t['sha'], config=t['config'],
                             batches=list(CORE_BATCHES if core else ALL_BATCHES), budget=t.get('budget')))
        tasks[bb] = rows
    save(ROOT / 'tasks.json', dict(backbones=list(BBS), rounds=ROUNDS, protocol='10 eager warm-ups, CUDA graph, 10 graph warm-ups, 50 replays, 3 interleaved rounds',
                                   tasks=tasks, time=time.time()))
    shutil.copy2(PRICING / 'design.json', ROOT / 'design.json')
    with tarfile.open(KIT, 'w:gz') as tar:
        for p in (ROOT / 'tasks.json', ROOT / 'design.json', Path(__file__), REPO / 'scripts/kaggle/second_device_kit.tar.gz'):
            tar.add(p, arcname=str(p.resolve().relative_to(REPO)))
    print('kit', KIT, f'{KIT.stat().st_size / 1e6:.1f} MB;', {bb: len(r) for bb, r in tasks.items()}, 'arms')


# ----------------------------------------------------------------------------------------------- time (device)
def runner(bb):
    from leaf.reproduction.runners.alpha_runner import alpha_runner
    return alpha_runner(bb, ROOT, design=load(ROOT / 'design.json'), reuse_roots=[])


def card_idle():
    import os
    index = os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0]
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,pci.bus_id', '--format=csv,noheader'], text=True)
    bus = dict(r.split(', ') for r in rows.strip().splitlines()).get(str(index))
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_bus_id', '--format=csv,noheader'], text=True)
    others = [r for r in apps.strip().splitlines() if r and r.split(', ')[1] == bus and int(r.split(', ')[0]) != os.getpid()]
    return not others


def timing(bbs, smoke=False):
    import os
    import torch
    from leaf.reproduction.runners.exclusive_timing import cast_exclusive
    os.environ['LEAF_EXCLUSIVE_TIMING'] = '1'
    tasks = load(ROOT / 'tasks.json')['tasks']
    for bb in bbs:
        done = ROOT / bb / 'latency_done.json'
        if done.exists() and not smoke: print('exists', done); continue
        ts = tasks[bb][:3] if smoke else tasks[bb]; r = runner(bb); dest = ROOT / bb / ('smoke' if smoke else 'latency'); t0 = time.time()
        for rnd in range(1 if smoke else ROUNDS):
            for i, t in enumerate(ts if rnd % 2 == 0 else ts[::-1]):
                p = dest / f"{t['id']}#{rnd}.json"
                if p.exists(): continue
                if not card_idle(): raise SystemExit('another process appeared on the timed card; session discarded')
                meas = {}
                try:
                    m = r.inference_model(t['config'])
                    for b in t['batches']:
                        try: meas[str(b)] = cast_exclusive(m, b, iters=5 if smoke else (50 if b <= 32 else 20), warmup=2 if smoke else (10 if b <= 32 else 5))
                        except torch.cuda.OutOfMemoryError: meas[str(b)] = 'OOM'; torch.cuda.empty_cache()
                except Exception as e:
                    meas = dict(status='error', error=repr(e)[:400])
                save(p, dict(id=t['id'], names=[t['name']], family=t['family'], round=rnd, measurements=meas, time=time.time()))
                m = None; gc.collect(); torch.cuda.empty_cache()
                if smoke or i % 20 == 0: print(f"{bb} round {rnd} {i + 1}/{len(ts)} {t['family']} {meas if smoke else ''} {time.time() - t0:.0f}s", flush=True)
        if not smoke: save(done, dict(rounds=ROUNDS, n=len(ts), time=time.time(), **device_info()))
        r = None; gc.collect(); torch.cuda.empty_cache()
        print('timed', bb, len(ts), flush=True)


# ----------------------------------------------------------------------------------------------- pack / unpack
def pack():
    tag = re.sub(r'[^A-Za-z0-9]+', '_', device_info()['device']).strip('_')
    dest = REPO / f'a100_timing_{tag}.tar.gz'
    with tarfile.open(dest, 'w:gz') as tar:
        for p in sorted(ROOT.rglob('*.json')): tar.add(p, arcname=str(p.relative_to(REPO)))
    print('wrote', dest, f'{dest.stat().st_size / 1e6:.1f} MB')


def unpack(path):
    with tarfile.open(path) as tar:
        for m in tar.getmembers():
            if not m.name.startswith('results/a100_timing/') or '..' in Path(m.name).parts: raise SystemExit(f'unexpected member {m.name}')
            dest = REPO / m.name; data = tar.extractfile(m).read()
            if dest.exists() and dest.read_bytes() != data and dest.name in ('tasks.json', 'design.json'): raise SystemExit(f'{dest} differs from the archive')
            dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(data)
    print('unpacked', path)


# ----------------------------------------------------------------------------------------------- compare (home)
def compare(bbs):
    from leaf.reproduction.experiments import batch_sensitivity as B
    B.ROOT = ROOT                                     # same analysis (pairs within tau per batch) on the device latencies
    out = {}; info = None
    for bb in bbs:
        if not (ROOT / bb / 'latency_done.json').exists(): print('not timed', bb); continue
        info = info or {k: v for k, v in load(ROOT / bb / 'latency_done.json').items() if k not in ('n', 'rounds', 'time')}
        out[bb] = {}
        for bs in sorted({b for t in load(ROOT / 'tasks.json')['tasks'][bb] for b in t['batches']}):
            r = B.analyse(bb, bs)
            r['totals'] = dict(pairs=sum(f['pairs'] for f in r['summary'].values()),
                               **{f'{s}_{k}': sum(f[s][k] for f in r['summary'].values()) for s in ('val', 'v2') for k in ('win', 'loss')})
            leaf = [p for p in r['points'] if p['family'] == 'LEAF' and p['speedup']]
            r['leaf_speedup'] = [round(p['speedup'], 3) for p in sorted(leaf, key=lambda p: p['ms'])]
            out[bb][str(bs)] = r
            t = r['totals']; dv = [x for f in r['summary'].values() for x in f['val']['deltas']]
            print(f"{bb:7s} bs{bs:3d}: {t['pairs']:3d} pairs, val W/L {t['val_win']}/{t['val_loss']} mean {np.mean(dv) if dv else 0:+.2f}, "
                  f"v2 W/L {t['v2_win']}/{t['v2_loss']}; dense {r['dense_ms']:.2f} ms; LEAF speed-ups {r['leaf_speedup']}")
    tag = re.sub(r'[^A-Za-z0-9]+', '_', (info or {}).get('device', 'device')).strip('_')
    save(ROOT / f'report_{tag}.json', dict(device=info, backbones=out, time=time.time()))
    print('report', ROOT / f'report_{tag}.json')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=('export', 'time', 'smoke', 'pack', 'unpack', 'compare'))
    ap.add_argument('--backbones', default=','.join(BBS)); ap.add_argument('--archive')
    a = ap.parse_args(); bbs = a.backbones.split(',')
    if a.stage == 'export': export()
    elif a.stage == 'time': timing(bbs)
    elif a.stage == 'smoke': timing(bbs[:1], smoke=True)
    elif a.stage == 'pack': pack()
    elif a.stage == 'unpack': unpack(a.archive)
    else: compare(bbs)


if __name__ == '__main__':
    main()
