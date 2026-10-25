"""Reproduction of the paper: `leaf experiments` re-runs its experiments, `leaf report` rebuilds its outputs.

  leaf experiments list                                 experiments, their paper sections, stages and result folders
  leaf experiments run <experiment> [stage] [options]   run one experiment (its own command line)

  leaf report analyze                                   CPU: recompute every analysis from the raw results (results/)
  leaf report tables | figures | all                    CPU: write paper/tables/*.tex and paper/figures/*.pdf
  leaf report assets [--backbones ...] [--no-figures]   CPU: schedule books assets/schedules/<bb>.json (every schedule the
                                                        paper evaluated, with its numbers) and PNG figures in assets/figures

GPU stages refuse a busy card; the paper's protocol evaluates in BF16 at batch 32 and times on one idle GPU.
"""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EXP = REPO / 'leaf/reproduction/experiments'

# name -> (paper section, results folder, stages)
EXPERIMENTS = {
    'main_comparison': ('Sec. 5.2 comparison with nine re-run families; Tab. S1', 'main_comparison',
                        'plan (CPU) | worker (GPU, confirm/val/V2) | time (GPU, alone)'),
    'ablation_and_alpha_grid': ('the paper\'s schedules (group leaf, 7-value alpha grid); Sec. 5.5 ablation, grids, surrogate check',
                                'ablation_and_alpha_grid', 'estimate | probe | plan [--dry-run] | worker | time | surrogate | report'),
    'main_comparison_analysis': ('Sec. 5.2 latency pairing, McNemar/Holm, latency transfer (Tab. latency-pairs, Tab. S1, Fig. 2)',
                                 'main_comparison_analysis', '[check] [--tables]'),
    'campaign_reports': ('reports of the main campaign and the latency-pricing campaign; main-campaign ablation report',
                         'main_comparison', 'main | pricing | ablation | export'),
    'equal_gmac': ('Sec. 5.3 equal GMAC (Tabs. equal-GMAC, Fig. residual vs stations)', 'equal_gmac', 'gmac (GPU) | analyze | figure_paper'),
    'latency_pricing': ('Sec. 5.4 where latency departs from FLOPs (Tab. latency pricing, Fig. SALT vs GMAC)', 'latency_pricing',
                        'inventory | gpu_time | cpu_time | analyze | plan | worker | final_time | compare'),
    'imagenet_real': ('ImageNet-ReaL re-scoring of the main comparison', 'imagenet_real', '(no stage)'),
    'clip_comparison': ('CLIP ViT-B/16: zero-shot ImageNet and Flickr30k/COCO retrieval with the re-run families (revision)',
                        'clip_comparison', 'search (GPUs) | plan (CPU) | worker (GPU) | time (GPU, alone) | report (CPU)'),
    'seg_comparison': ('segmentation pilot: Segmenter ViT-B/16 on ADE20K (mIoU), LEAF vs ToMe with unmerging',
                       'seg_comparison', 'search (GPU) | plan (CPU) | worker (GPU) | time (GPU, alone) | report (CPU)'),
    'final_station_and_ci': ('final-block station, paired confidence intervals, station fit (Sec. 5.5, 5.2)', 'final_station_and_ci',
                             'lastblock | ci | stationfit | all'),
    'alpha_kappa_equivalence': ('alpha = kappa/(1+kappa) gives the same token order (per-image check on val)', 'alpha_kappa_equivalence',
                                'tasks | worker | report | status'),
    'alpha_kappa_conversion': ('writes the alpha form of the kappa schedule sets (schedules/kappa_form -> schedules/<set>)', '-', '(no stage)'),
    'forecast_rank': ('Tab. low-rank approximation of the forecast (energy retained by rank r)', 'forecast_rank', '--backbones ...'),
    'forecast_quality': ('Tab. forecast quality: Spearman of P, D and the combined score with the future attention', 'leaf_m0_forecast_quality',
                         'run (GPU) | check'),
    'forecast_approximation': ('token-ranking agreement of the G16/r32 forecast with full rank and exact D', 'forecast_approximation', '--gpu'),
    'token_schedule': ('Fig. token schedule of a GMAC-planned DeiT-B schedule', 'token_schedule', '--backbone --schedule [--render-only]'),
    'search_cost': ('Tab. search cost (throughput of the dense models)', 'search_cost', 'measure (GPU)'),
    'gmac_planning': ('probes on 5,000 calibration images (M0 / M3), teachers, GMAC-planned schedules', 'gmac_planning',
                      'setup | profile | probes | prepare | macs | accuracy | latency | compare | audit'),
    'low_budget_and_factorial': ('very low budgets and the D x M3 factorial (Tab. S2); probe curves for the planner',
                                 'low_budget_and_factorial', 'launch | coordinate | setup | plan | worker | timing | analyze | verify | status'),
    'horizon_norm_seed_ablation': ('forecast horizon, M3 norm variants, calibration seeds', 'horizon_norm_seed_ablation',
                                   'launch | coordinate | probes | planning | worker | timing | verify | status'),
    'salt_fit': ('SALT: fit of the station-rich latency model and eight alternative formulas (Tab. S2)', 'salt_fit', '(no stage)'),
    'salt_formulas': ('SALT formulas planned through the DP and evaluated', 'salt_fit', 'plan | launch | worker | timing | report'),
    'salt_validation': ('SALT validation on 84 schedules', 'salt_validation', 'plan | launch | worker | timing | report'),
    'salt_m3_pilot': ('SALT with the M3 operator, pilot', 'salt_m3_pilot', 'plan | worker | timing | report | launch'),
    'merge_variant_screening': ('merge variants M1-N6 (Tab. S2)', 'merge_variants', 'design | seal | launch | scheduler | worker | benchmark | report'),
    'merge_norm_gate': ('norm-preserving merge: correctness and cost gate', 'merge_variants', 'launch | run'),
    'merge_norm_evaluation': ('norm-preserving merge (M3, N4) planned and evaluated', 'merge_variants',
                              'setup | launch | scheduler | prepare | latency | heldout | report'),
}
# the CPU analysis chain behind the tables, in dependency order
ANALYZE = [
    ('campaign_reports', ['main']), ('campaign_reports', ['ablation']), ('imagenet_real', []), ('equal_gmac', ['analyze']),
    ('final_station_and_ci', ['all']), ('ablation_and_alpha_grid', ['report']), ('main_comparison_analysis', ['check']),
    ('main_comparison_analysis', []), ('alpha_kappa_equivalence', ['report']), ('latency_pricing', ['analyze']),
    ('latency_pricing', ['compare']),
]


def run(name, args):
    path = EXP / f'{name}.py'
    if not path.exists(): raise SystemExit(f'unknown experiment {name!r}; see: leaf experiments list')
    return subprocess.run([sys.executable, str(path)] + list(args), cwd=REPO).returncode


def usage(argv):
    if not argv or argv[0] in ('-h', '--help', 'help'):
        print(__doc__); return True
    return False


def experiments(argv):
    """leaf experiments list | run <experiment> [stage] [options]"""
    if usage(argv): return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == 'list':
        for name, (section, folder, stages) in EXPERIMENTS.items():
            print(f'{name}\n    {section}\n    results/{folder}    stages: {stages}')
        return 0
    if cmd == 'run':
        if not rest: raise SystemExit('usage: leaf experiments run <experiment> [stage] [options]')
        return run(rest[0], rest[1:])
    raise SystemExit(f'unknown command leaf experiments {cmd!r}\n{__doc__}')


def report(argv):
    """leaf report analyze | tables | figures | assets | all"""
    if usage(argv): return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == 'analyze':
        for name, args in ANALYZE:
            print(f'== {name} {" ".join(args)}', flush=True)
            if run(name, args): raise SystemExit(f'{name} {args} failed')
        return 0
    if cmd == 'assets':
        from leaf.reproduction.report.schedule_books import main as assets_main
        assets_main(rest); return 0
    if cmd in ('tables', 'figures', 'all'):
        from leaf.reproduction.report import build
        build.main([cmd]); return 0
    raise SystemExit(f'unknown command leaf report {cmd!r}\n{__doc__}')
