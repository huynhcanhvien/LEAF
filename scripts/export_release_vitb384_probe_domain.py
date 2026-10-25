"""Release files for the two experiments added on 2026-10-25, in the format of schedules/leaf/ (leaf/reproduction/report/release.py):

  paper/archived/repo/schedules/leaf/augreg_b384.json                 the six latency-planned LEAF schedules on ViT-B/16 at 384 px
  paper/archived/repo/schedules/leaf/planning/augreg_b384/            probes, groups, SALT model + workloads, budgets (what `leaf plan` needs)
  paper/archived/repo/schedules/leaf/deit_b_coco_probes.json          the seven DeiT-B schedules re-planned from probes on 1,000 COCO images
  paper/archived/repo/schedules/leaf/planning/deit_b_coco1000/        the COCO probe table (groups, SALT and budgets are those of deit_b)

Copy the `schedules/` tree into the repository. CPU only; reads the sealed result folders.
Usage: python scripts/export_release_vitb384_probe_domain.py
"""
import json
import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
os.environ['LEAF_BACKBONES'] = str(REPO / 'leaf/reproduction/experiments/backbones_vitb384.json')
sys.path.insert(0, str(REPO))
from leaf.backbones import SPECS                      # noqa: E402
from leaf.api import stations                         # noqa: E402
from leaf.planner.alpha import config_gmac            # noqa: E402

OUT = REPO / 'paper/archived/repo/schedules/leaf'
V = REPO / 'results/vitb384_comparison/augreg_b384'
P = REPO / 'results/probe_domain'
HW = 'NVIDIA RTX 5060 Ti, BF16, CUDA graph, batch 32; latency = mean of three interleaved rounds'


def load(p): return json.loads(Path(p).read_text())


def save(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True); Path(p).write_text(json.dumps(obj, indent=1, sort_keys=False, default=float))


def rel(p): return str(Path(p).relative_to(REPO))


def vitb384():
    bb = 'augreg_b384'; spec = SPECS[bb]
    plans = load(V / 'plans.json'); rep = load(V / 'report.json'); pts = {p['name']: p for p in rep['points']}
    budgets = load(V / 'search/budgets.json'); dense = pts['ref/dense']
    doc = dict(
        backbone=bb, timm_id='vit_base_patch16_384.augreg_in21k_ft_in1k', image_size=384, n_tokens=577,
        method='LEAF: stations scored by (1 - alpha) z(log P) + alpha z(D), prune -> MLERP (M3) merge, '
               'planned by the Lagrangian DP under the station-rich SALT latency estimate; token-count grid step 8',
        forecast=dict(G=16, rank=32), operator='M3', alpha_grid=[0., .25, 1 / 3, .5, 2 / 3, .75, 1.], theta_grid=[0., .5, 1.], hardware=HW,
        dense=dict(val=dense['val'], v2=dense['v2'], ms_bs32=round(dense['ms'], 3), sd=round(dense['sd'], 3),
                   ms_bs32_at_planning=round(budgets['measured']['dense']['ms'], 3)),
        evaluation='Supplementary Material S5.12 of the KBS manuscript: 153 arms timed in one session (results/vitb384_comparison/augreg_b384/report.json)',
        sources=dict(plans=rel(V / 'plans.json'), report=rel(V / 'report.json'), search=rel(V / 'search')),
        schedules={})
    for b in budgets['budgets']:
        bid = b['id']; job = plans['jobs'][f'leaf/latency/{bid}']; row = pts[f'leaf/latency/{bid}']; cfg = job['config']
        doc['schedules'][f'latency/{bid}'] = dict(
            budget=dict(kind='latency_ms_bs32', value=b['latency_ms'], id=bid, rule='fraction of the dense latency measured before planning'),
            stations=stations(cfg, spec), config=cfg, config_sha=job['config_sha'],
            planner=dict(fhat=job['planner']['fhat'], predicted_ms=job['planner']['predicted_ms'], gmac=config_gmac(cfg, spec)),
            measured=dict(val=row['val'], v2=row['v2'], ms_bs32=round(row['ms'], 3), sd=round(row['sd'], 3), stations=len(stations(cfg, spec))))
    save(OUT / f'{bb}.json', doc)
    d = OUT / 'planning' / bb; d.mkdir(parents=True, exist_ok=True)
    for src, dst in (('search/probes/probes.json', 'probes.json'), ('search/groups.json', 'groups.json'), ('search/salt.json', 'salt.json'),
                     ('search/salt_workloads.json', 'salt_workloads.json'), ('search/budgets.json', 'budgets.json')):
        (d / dst).write_text((V / src).read_text().replace(str(REPO) + '/', ''))   # file pointers relative to the repository root
    return len(doc['schedules'])


def probe_domain():
    bb = 'deit_b'; spec = SPECS[bb]
    plans = load(P / bb / 'plans.json'); rep = load(P / 'report.json'); rows = {r['budget']: r for r in rep['rows'] if r['subset'] == 'coco1000'}
    released = load(REPO / 'schedules/leaf/deit_b.json')
    doc = dict(
        backbone=bb, timm_id=released['timm_id'],
        method=released['method'] + '; probes measured on 1,000 unlabelled MS-COCO images instead of 5,000 ImageNet training images '
               '(same head groups, SALT model and budgets as schedules/leaf/deit_b.json)',
        probe_images='MS-COCO Karpathy test split, first 2,000 images, even indices (1,000 images, no labels used)',
        forecast=released['forecast'], operator=released['operator'], alpha_grid=released['alpha_grid'], theta_grid=released['theta_grid'],
        hardware=HW, dense=released['dense'],
        evaluation='Supplementary Material S5.5 of the KBS manuscript, paragraph "Out-of-domain probe images" (results/probe_domain/report.json); '
                   'measured values come from one timing session shared with the released schedules',
        sources=dict(plans=rel(P / bb / 'plans.json'), report=rel(P / 'report.json'), probes=rel(P / bb / 'coco1000/probes/probes.json')),
        schedules={})
    for name, job in plans['jobs'].items():
        if not name.startswith('coco1000/'): continue
        bid = name.split('/')[1]; cfg = job['config']; r = rows[bid]
        ref = released['schedules'][f'latency/{bid}']
        doc['schedules'][f'latency/{bid}'] = dict(
            budget=ref['budget'], stations=stations(cfg, spec), config=cfg, config_sha=job['config_sha'],
            same_as_released=bool(job.get('same_as_released')),
            planner=dict(fhat_coco=job['fhat'], predicted_ms=job['predicted_ms'], gmac=config_gmac(cfg, spec)),
            measured=dict(val=r['val']['top1'], v2=r['v2']['top1'], ms_bs32=round(r['ms'], 3),
                          delta_val_vs_released=r['val']['delta'], delta_v2_vs_released=r['v2']['delta'],
                          holm_val=r['val']['holm'], holm_v2=r['v2']['holm'], stations=r['K']))
    save(OUT / 'deit_b_coco_probes.json', doc)
    d = OUT / 'planning' / 'deit_b_coco1000'; d.mkdir(parents=True, exist_ok=True)
    pr = load(P / bb / 'coco1000/probes/probes.json')
    pr = dict(images=doc['probe_images'], images_field_as_written_by_leaf_search=pr.get('images'), **{k: v for k, v in pr.items() if k != 'images'})
    save(d / 'probes.json', pr)
    (d / 'README.md').write_text('Probe table of DeiT-B measured on 1,000 unlabelled MS-COCO images (probe_domain.py). Head groups, the SALT model, '
                                 'its workloads and the budgets are those of `planning/deit_b/`.\n')
    return len(doc['schedules'])


if __name__ == '__main__':
    print('augreg_b384 schedules:', vitb384()); print('deit_b COCO-probe schedules:', probe_domain())
    for p in sorted(OUT.rglob('*')):
        if p.is_file(): print(f'{p.stat().st_size / 1e3:9.1f} kB  {p.relative_to(REPO)}')
