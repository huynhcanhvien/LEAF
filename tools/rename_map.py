"""Old -> new names used when this repository was created from the research repository (efficient-vlm).

One table for everything that was renamed: result folders and stored paths (PATHS), experiment modules (MODULES),
and the arm / group / budget keys stored in results and used in code (KEYS). tools/migrate_results.py applies
PATHS and KEYS to the copied raw results; the experiment code under leaf/reproduction/ was ported with the same tables.
PROVENANCE.md prints them.
"""
OLD_REPO = '/home/huynhvien/projects/efficient-vlm'
M3_WORKTREE = '/home/huynhvien/.codex-vien/worktrees/m3-baselines-20261016/efficient-vlm'
M0_WORKTREE = '/home/huynhvien/.codex-vien/worktrees/5d6e/efficient-vlm'

# result folders: old campaign/date -> results/<experiment>
RESULT_DIRS = {
    'q1_main_v2/2026-10-18': 'main_comparison',
    'q1_main_r1/2026-10-20': 'main_comparison_analysis',
    'ablation_v3/2026-10-19': 'ablation_and_alpha_grid',
    'alpha_equivalence/2026-10-19': 'alpha_kappa_equivalence',
    'q1_latency_pricing/2026-10-18': 'latency_pricing',
    'q1_equal_gmac/2026-10-18': 'equal_gmac',
    'q1_real/2026-10-18': 'imagenet_real',
    'q1_review_cpu/2026-10-19': 'final_station_and_ci',
    'q1_forecast_approx/2026-10-19': 'forecast_approximation',
    'q1_tokens/2026-10-18': 'token_schedule',
    'dp_m3_baselines/2026-10-16': 'gmac_planning',
    'latency_d_m3_low_budget/2026-10-18': 'low_budget_and_factorial',
    'latency_d_m3_extra_ablation/2026-10-18': 'horizon_norm_seed_ablation',
    't_proxy_comparison/2026-10-17': 'salt_fit',
    't_proxy_validation/2026-10-17': 'salt_validation',
    't_m3_pilot/2026-10-17': 'salt_m3_pilot',
    'merge_variants/2026-10-16': 'merge_variants',
    'paper_final/2026-10-14': 'leaf_m0',
}


def path_replacements():
    """(old, new) string pairs for paths stored in result files, longest first: absolute paths of the old machine
    and repo-relative old paths both become repo-relative new paths."""
    pairs = []
    for old, new in RESULT_DIRS.items():
        for root in (OLD_REPO, M3_WORKTREE, M0_WORKTREE):
            pairs.append((f'{root}/artifacts/results/{old}', f'results/{new}'))
        pairs.append((f'artifacts/results/{old}', f'results/{new}'))
    pairs.append((f'{OLD_REPO}/artifacts/results/q1_logs/search_cost_throughput.json', 'results/search_cost/throughput.json'))
    pairs.append(('artifacts/results/q1_logs/search_cost_throughput.json', 'results/search_cost/throughput.json'))
    pairs.append((f'{OLD_REPO}/data/cache', 'data/cache'))
    return sorted(pairs, key=lambda p: -len(p[0]))


# experiment modules: old script / helper -> leaf.reproduction module
MODULES = {
    'scripts.q1_main_v2': 'leaf.reproduction.experiments.main_comparison',
    'scripts.q1_r1_main': 'leaf.reproduction.experiments.main_comparison_analysis',
    'scripts.q1_report': 'leaf.reproduction.experiments.campaign_reports',
    'scripts.q1_report_fig_large': 'leaf.reproduction.experiments.accuracy_latency_figures',
    'scripts.paper_v3_figs_large': 'leaf.reproduction.experiments.leaf_m0_figures',
    'scripts.q1_common': 'leaf.reproduction.runners.exclusive_timing',
    'scripts.q1_equal_gmac': 'leaf.reproduction.experiments.equal_gmac',
    'scripts.q1_forecast_approx': 'leaf.reproduction.experiments.forecast_approximation',
    'scripts.q1_latency_pricing': 'leaf.reproduction.experiments.latency_pricing',
    'scripts.q1_real': 'leaf.reproduction.experiments.imagenet_real',
    'scripts.q1_review_cpu': 'leaf.reproduction.experiments.final_station_and_ci',
    'scripts.q1_visualize_tokens': 'leaf.reproduction.experiments.token_schedule',
    'scripts.ablation_v3': 'leaf.reproduction.experiments.ablation_and_alpha_grid',
    'scripts.ablation_v3_paper': 'leaf.reproduction.experiments.ablation_tables',
    'scripts.alpha_convert': 'leaf.reproduction.experiments.alpha_kappa_conversion',
    'scripts.alpha_equivalence': 'leaf.reproduction.experiments.alpha_kappa_equivalence',
    'scripts.dp_m3_baselines': 'leaf.reproduction.experiments.gmac_planning',
    'scripts.latency_d_m3_low_budget': 'leaf.reproduction.experiments.low_budget_and_factorial',
    'scripts.latency_d_m3_extra_ablation': 'leaf.reproduction.experiments.horizon_norm_seed_ablation',
    'scripts.merge_norm_stage15': 'leaf.reproduction.experiments.merge_norm_gate',
    'scripts.merge_norm_stage2': 'leaf.reproduction.experiments.merge_norm_evaluation',
    'scripts.merge_norm_finish': 'leaf.reproduction.experiments.merge_norm_finalize',
    'scripts.merge_norm_watch': 'leaf.reproduction.experiments.merge_norm_monitor',
    'scripts.merge_variants': 'leaf.reproduction.experiments.merge_variant_screening',
    'scripts.t_proxy_compare': 'leaf.reproduction.experiments.salt_fit',
    'scripts.t_proxy_experiments': 'leaf.reproduction.experiments.salt_formulas',
    'scripts.t_proxy_validation': 'leaf.reproduction.experiments.salt_validation',
    'scripts.t_proxy_validation_extra': 'leaf.reproduction.experiments.salt_validation_extra_timing',
    'scripts.t_proxy_validation_finish': 'leaf.reproduction.experiments.salt_validation_audit',
    'scripts.t_proxy_validation_resume': 'leaf.reproduction.experiments.salt_validation_resume',
    'scripts.t_proxy_summary': 'leaf.reproduction.experiments.salt_formula_summary',
    'scripts.t_proxy_finalize': 'leaf.reproduction.experiments.salt_formula_finalize',
    'scripts.t_m3_pilot': 'leaf.reproduction.experiments.salt_m3_pilot',
    'scripts.t_m3_pilot_parallel': 'leaf.reproduction.experiments.salt_m3_pilot_parallel',
    'scripts.t_m3_pilot_summary': 'leaf.reproduction.experiments.salt_m3_pilot_summary',
    'scripts.forecast_rank_audit': 'leaf.reproduction.experiments.forecast_rank',
    'scripts.export_leaf_schedules': 'leaf.reproduction.report.release',
    # experiment-only helpers (efficient-vlm experiments/lib)
    'evaluation.dp_m3_baselines': 'leaf.reproduction.runners.campaign_runner',
    'evaluation.merge_variants': 'leaf.reproduction.runners.merge_variant_runner',
    'evaluation.alpha_runner': 'leaf.reproduction.runners.alpha_runner',
    'evaluation.protocol': 'leaf.reproduction.runners.sealed_stage_protocol',
    'evaluation.extra_ablation_analysis': 'leaf.reproduction.experiments.horizon_norm_seed_analysis',
    'evaluation.low_budget_analysis': 'leaf.reproduction.experiments.low_budget_analysis',
    'evaluation.merge_norm_campaign': 'leaf.reproduction.experiments.merge_norm_runner',
    'models.extra_ablation_vit': 'leaf.reproduction.ablation_models.ablation_vit',
    'models.extra_ablation_kernel': 'leaf.reproduction.ablation_models.norm_target_merge_kernel',
    'planner.plan_proxy_variants': 'leaf.reproduction.ablation_models.planner_ablation_switches',
    # paper table tools (efficient-vlm artifacts/paper/v2/tools)
    'artifacts.paper.v2.tools.q1_v2_tables': 'leaf.reproduction.report.tables',
    'artifacts.paper.v2.tools.typeset': 'leaf.reproduction.report.typesetting',
}

# old module file (in efficient-vlm) -> new module, for the port
SOURCE_FILES = {}
for _old, _new in MODULES.items():
    _head, _name = _old.split('.', 1)
    if _head == 'scripts': _src = f'experiments/{_name}.py'
    elif _head in ('evaluation', 'models', 'planner'): _src = f'experiments/lib/{_head}/{_name}.py'
    else: _src = _old.replace('.', '/') + '.py'
    SOURCE_FILES[_src] = _new


# ---------------------------------------------------------------------------------------------- keys
# Ablation groups of the alpha-grid / ablation campaign (results/ablation_and_alpha_grid): old code -> name.
# An arm key is '<group>/<variant>/<budget>'; the group is also stored as row['group'] and names plans/<group>.json.
ABLATION_GROUPS = {
    'R1': 'leaf',                    # the paper's method: all budgets re-planned on the 7-value alpha grid
    'R2': 'grid_theta_alpha',        # 5-value theta grid and 7-value alpha grid
    'R3': 'grid_theta',              # 5-value theta grid
    'A1': 'schedule_structure',      # uniform / single station / DiffRate stations (+ score x operator)
    'A2': 'shared_settings',         # one (theta, alpha), one theta or one alpha for all stations
    'A3': 'forecast_only',           # alpha = 1 (D only)
    'A4': 'planner_switches',        # no rescaling / single price
    'A5': 'random_head_groups',      # random head groups (placebo)
    'A7all': 'final_station_all',    # final-block station removed, all backbones
    'A7': 'final_station',           # final-block station removed / forbidden (DeiT-B, MAE-B)
    'K': 'fixed_station_count',      # exactly K stations
    'H': 'surrogate_check',          # random schedules: surrogate F-hat vs measured F
    'S1': 'alpha_grid',              # coarser / extended alpha grids
    'S2': 'forecast_compression',    # head groups G and rank r
    'S3': 'probe_sample_size',       # 1,250 / 2,500 probe images
    'S4': 'theta_grid',              # theta grids
    'S5': 'probe_removal_grid',      # probe removal grid (5 / 17 points)
    'S6': 'salt_workload_count',     # SALT refit on fewer timed workloads
}
MAIN_ARMS = {   # main comparison (results/main_comparison) ablation arms 'abl/<arm>/<budget>'
    'P_only': 'no_forecast', 'prune_only': 'prune_only', 'merge_only': 'merge_only', 'M0_replan': 'm0_operator',
    'M0_fixed': 'm0_operator_same_path', 'noD_fixed': 'no_forecast_same_path',
    'future1_fixed': 'horizon1_same_path', 'future3_fixed': 'horizon3_same_path',
}
FACTORIAL = {'A00': 'm0_p', 'A10': 'm0_pd', 'A01': 'm3_p', 'A11': 'm3_pd'}   # operator x score (low-budget factorial)
LABELS = {'laps2': 'leaf_alpha4', 'LaPS-v2': 'LEAF', 'LaPS (M0, paper)': 'LEAF-M0'}
FILE_RENAMES = {'numbers_r1.json': 'numbers_leaf.json', 'report_r1.json': 'report_leaf.json',
                'h_schedules.json': 'surrogate_check_schedules.json', 'h_done': 'surrogate_check_done', 'h_val': 'surrogate_check_val'}

import re as _re  # noqa: E402

_DATA_RULES = None


def _data_rules():
    rules = [(_re.compile(r'laps2'), 'leaf_alpha4'),
             (_re.compile(_re.escape('LaPS (M0, paper)')), 'LEAF-M0'), (_re.compile(r'LaPS-v2'), 'LEAF'),
             (_re.compile(r'"LaPS"'), '"LEAF-M0"'),
             (_re.compile(r'"_H": '), '"_surrogate_check": '), (_re.compile(r'LaPS-GMAC'), 'LEAF-GMAC')]   # reference label of the merge-variant campaign (LEAF-M0 schedules)
    for old, new in MAIN_ARMS.items():
        rules.append((_re.compile(r'"abl/' + old + r'/'), f'"ablation/{new}/'))
    rules.append((_re.compile(r'"abl/'), '"ablation/'))
    for old, new in MAIN_ARMS.items():   # bare arm names (results/main_comparison/ablation_report.json)
        rules.append((_re.compile(r'"arm": "' + old + r'"'), f'"arm": "{new}"'))
    for old, new in sorted(ABLATION_GROUPS.items(), key=lambda kv: -len(kv[0])):
        rules.append((_re.compile(r'"' + old + r'/'), f'"{new}/'))
        rules.append((_re.compile(r'"group": "' + old + r'"'), f'"group": "{new}"'))
        rules.append((_re.compile(r'/plans/' + old + r'\.json'), f'/plans/{new}.json'))
    for old, new in FACTORIAL.items():
        rules.append((_re.compile(r'\b' + old + r'\b'), new))
    for old, new in FILE_RENAMES.items():
        rules.append((_re.compile(r'(?<=[/"])' + _re.escape(old) + r'(?=[/"])'), new))
    return rules


def apply_keys(text):
    """Key renames on the text of a stored JSON result (exact, anchored on quotes and slashes)."""
    global _DATA_RULES
    if _DATA_RULES is None: _DATA_RULES = _data_rules()
    for rx, rep in _DATA_RULES: text = rx.sub(rep, text)
    return text


def renamed_file(name):
    """New file / folder name for a result file or folder name."""
    if name in FILE_RENAMES: return FILE_RENAMES[name]
    stem, dot, ext = name.partition('.')
    if ext == 'json' and stem in ABLATION_GROUPS: return ABLATION_GROUPS[stem] + '.json'
    return name.replace('laps2', 'leaf_alpha4')


def original_name(name):
    """Inverse of renamed_file (file / folder name in the research repository)."""
    for old, new in FILE_RENAMES.items():
        if name == new: return old
    stem, dot, ext = name.partition('.')
    for old, new in ABLATION_GROUPS.items():
        if ext == 'json' and stem == new: return old + '.json'
    return name.replace('leaf_alpha4', 'laps2')
