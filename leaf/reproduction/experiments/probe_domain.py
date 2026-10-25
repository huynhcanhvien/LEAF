"""Out-of-domain probes: does the planner need calibration images of the target distribution?

The released DeiT-B schedules are planned from single-station probes on 5,000 ImageNet training images. Here the
same 1,917 probe configurations are measured on 1,000 unlabelled MS-COCO images (data/cache_coco: the first 2,000
images of the COCO Karpathy test cache, whose even half is the probe split), the schedules are re-planned with the
released head groups, SALT model and budgets (schedules/leaf/planning/deit_b), evaluated on ImageNet val and V2 and
timed in one session with the released schedules (the calib_size protocol). The KL to the dense model's prediction
needs no labels and no class structure, so any natural images can serve; the test is whether ImageNet accuracy holds.

Stages (python leaf/reproduction/experiments/probe_domain.py <stage> [--backbones deit_b]):
  search  (GPUs) `leaf search --steps probe,plan` on the COCO cache -> <bb>/coco1000/{probes,schedules.json}
  plan    (CPU)  <bb>/plans.json in the calib_size format (subset coco1000 + released), sealed
  val     (GPU)  val / v2 of the re-planned schedules (released ones reused)
  time    (GPU)  one session, GPU 0, both cards idle
  report  (CPU)  report.json
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from leaf.reproduction.experiments import ablation_and_alpha_grid as A
from leaf.reproduction.experiments import calib_size as CS
from leaf.reproduction.experiments import dp_verify as DV

ROOT = REPO / 'results/probe_domain'
BBS = ('deit_b',)
PLANNING = REPO / 'schedules/leaf/planning'
DATA_CACHE = REPO / 'data/cache_coco'
SUBSET = ('coco1000', 1000, 'coco')
CS.ROOT, CS.SUBSETS = ROOT, (SUBSET,)        # reuse the calibration-size val / timing / report stages on this root


def workspace(bb): return ROOT / bb / 'coco1000'


def search(bb, gpus):
    ws = workspace(bb); ws.mkdir(parents=True, exist_ok=True); src = PLANNING / bb
    cmd = [sys.executable, '-m', 'leaf', 'search', '--backbone', bb, '--workspace', str(ws), '--gpus', gpus, '--steps', 'probe,plan',
           '--groups', str(src / 'groups.json'), '--salt', str(src / 'salt.json'), '--budgets', str(src / 'budgets.json'),
           '--data-cache', str(DATA_CACHE)]
    print(' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=REPO)


def plan(bb):
    dest = ROOT / bb / 'plans.json'
    if dest.exists(): print('exists', dest); return
    sched = CS.load(workspace(bb) / 'schedules.json')['schedules']; ref = DV.released(bb); c = A.Ctx(bb); jobs = {}
    for key, row in sched.items():
        bid = key.split('/')[-1]
        if bid not in ref or 'config' not in row: continue
        b = next(x for x in c.budgets if x['id'] == bid)
        state = dict(sites=[s['block'] for s in row['stations']], counts=[s['tokens_after'] for s in row['stations']])
        same = A.sha(row['config']) == A.sha(ref[bid]['config'])
        jobs[f'{SUBSET[0]}/{bid}'] = dict(config=row['config'], state=state, fhat=row['planner']['fhat'], predicted_ms=row['planner']['predicted_ms'],
                                        budget=b, subset=SUBSET[0], n_images=SUBSET[1], seed=SUBSET[2], same_as_released=same)
    for bid, r in ref.items():
        jobs[f'released/{bid}'] = dict(config=r['config'], state=r['state'], fhat=r.get('fhat'), predicted_ms=r.get('predicted_ms'),
                                       budget=next(x for x in c.budgets if x['id'] == bid), subset='released', n_images=A.CALIB, seed=None)
    jobs = A._shas(jobs)
    CS.save(dest, dict(backbone=bb, jobs=jobs, sha=A.sha(jobs), subsets=[SUBSET[0]], probe_images='1,000 MS-COCO images (data/cache_coco, even half)', time=time.time()))
    print(bb, len(jobs), 'jobs; changed vs released:', sum(1 for k, j in jobs.items() if k.startswith(SUBSET[0]) and not j['same_as_released']), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=('search', 'plan', 'val', 'time', 'report'))
    ap.add_argument('--backbones', default=','.join(BBS))
    ap.add_argument('--gpus', default='0,1')
    a = ap.parse_args(); bbs = a.backbones.split(',')
    if a.stage == 'search':
        for bb in bbs: search(bb, a.gpus)
    elif a.stage == 'plan':
        for bb in bbs: plan(bb)
    elif a.stage == 'val': CS.val(bbs)
    elif a.stage == 'time': CS.timing(bbs)
    else: CS.report(bbs)


if __name__ == '__main__':
    main()
