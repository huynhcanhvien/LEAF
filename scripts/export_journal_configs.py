"""Copy the sealed configurations of the experiments added for the journal version into schedules/journal/.

For every experiment this keeps only what defines the evaluated arms and the planner inputs: the sealed plans
(every arm's runtime config and its SHA-256), the timing designs, budgets, SALT models and the workloads they were
fitted on, and DiffRate's searched rates. Per-image predictions and timings are not copied. Re-running an experiment
with these files (`python leaf/reproduction/experiments/<name>.py worker|time|report`) evaluates exactly these arms.
Usage: python scripts/export_journal_configs.py
"""
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RES, OUT = REPO / 'results', REPO / 'schedules/journal'

# experiment -> (paper section, glob patterns relative to results/<experiment>)
EXPERIMENTS = {
    'clip_comparison': ('Section 6.5, Supplementary S5.2: CLIP ViT-B/16, zero-shot ImageNet and retrieval',
                        ['clip_b16/plans*.json', 'clip_b16/search/budgets.json', 'clip_b16/search/salt.json', 'clip_b16/search/salt_workloads.json',
                         'clip_b16/search/schedules*.json', 'clip_b16/dr_search_schedules_readable.json']),
    'diffrate_clip_search': ('Section 6.5: DiffRate\'s own search on CLIP (searched rates per GMAC budget)', ['rates.json', '*/rates.json']),
    'search_baselines': ('Appendix B: random search and regularised evolution', ['sealed.json', 'schedules_readable.json', '*/plans*.json']),
    'second_device': ('Section 6.3: transfer to a Tesla T4 (SALT refitted on the device)', ['design.json', '*/plans.json']),
    'dp_verify': ('Section 7, Supplementary S5.7: planner candidates verified on the measured KL', ['*/plans.json']),
    'avgpool_final_station': ('Section 7, Supplementary S5.8: no final-block station on average-pooled heads', ['*/plans.json']),
    'calib_size': ('Section 7, Supplementary S5.5: 1,000 to 10,000 probe images', ['*/plans.json']),
    'probe_domain': ('Section 7, Supplementary S5.5: probes on 1,000 MS-COCO images', ['deit_b/plans.json', 'deit_b/coco1000/schedules.json']),
    'vitb384_comparison': ('Section 6.1, Supplementary S5.12: ViT-B/16 at 384 pixels, all families',
                           ['augreg_b384/plans.json', 'augreg_b384/plans_dense.json', 'augreg_b384/search/budgets.json', 'augreg_b384/search/salt.json',
                            'augreg_b384/search/salt_workloads.json', 'augreg_b384/search/schedules.json']),
    'station_discount': ('Section 7: per-station discount of the surrogate', ['*/plans.json']),
    'final_station_alpha7': ('Section 7: schedules without their final-block station', ['*/plans.json']),
    'salt_low_coverage': ('Section 6.3, Supplementary S5.6: SALT below the range of its timing data',
                          ['*/workloads.json', '*/salt_ext.json', '*/plans_ext.json']),
}


def main():
    rows = []
    for exp, (where, pats) in EXPERIMENTS.items():
        files = sorted({p for pat in pats for p in (RES / exp).glob(pat) if p.is_file() and 'unused' not in p.name})
        for p in files:
            dest = OUT / exp / p.relative_to(RES / exp); dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(p.read_text().replace(str(REPO) + '/', ''))      # file pointers relative to the repository root
        size = sum(p.stat().st_size for p in files)
        rows.append((exp, where, len(files), size)); print(f'{exp:24s} {len(files):3d} files {size / 1e6:6.2f} MB')
    lines = ['# Sealed configurations of the journal-version experiments', '',
             'Written by `scripts/export_journal_configs.py` from the sealed campaign folders. Each folder holds the files that define the',
             'evaluated arms (every arm\'s runtime config with its SHA-256, sealed before any evaluation), the timing design, the budgets and',
             'the SALT models with their workloads. Place a folder under `results/<experiment>/` (same relative paths) and run the',
             '`worker`, `time` and `report` stages of `leaf/reproduction/experiments/<experiment>.py` to evaluate exactly these arms;',
             'the LEAF schedules themselves are also in `schedules/leaf/` in readable form.', '',
             '| Folder | Paper | Files |', '|---|---|---:|']
    lines += [f'| `{e}/` | {w} | {n} |' for e, w, n, _ in rows]
    (OUT / 'README.md').write_text('\n'.join(lines) + '\n')
    print('total', sum(r[3] for r in rows) / 1e6, 'MB ->', OUT.relative_to(REPO))


if __name__ == '__main__':
    main()
