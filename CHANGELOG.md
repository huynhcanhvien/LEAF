# Changelog

## Unreleased

- **Names.**
  - Commands: `leaf evaluate`, `benchmark`, `group-heads`, `fit-latency` and `data prepare`; `leaf experiments list | run` and `leaf report analyze | tables | figures | assets | all` replace `leaf paper`.
  - Package: `leaf/reproduction/` (`experiments/`, `runners/`, `ablation_models/`, `report/`) replaces `leaf/paper/`.
  - Modules: `model/reduced_vit`, `mlerp_vit`, `merge_variant_vit`, `alpha_score`, `token_ops`, `kernels/`; `forecast/attention_forecast`, `head_groups`, `cpu_forecast`, `kernels/`; `planner/schedule_caps` (it replaces the two copies `schedule` and `merge_schedule`); `latency/timing`, `cpu_timing`, `profile_model`.
  - Public functions and classes: `ReducedViT`, `MlerpViT`, `AttentionForecast`, `ForecastConfig`, `config_from_state`, `plan_under_latency`, `to_alpha_config`, `config_hash`, `measure_latency`, `fit_salt`, `group_heads`, ...
  - Config kinds (`pd`, `pdm`, `pda`) are unchanged: they are part of the hashes that name the result files.
- **Backbone registry.** `leaf/backbones.json` maps CLI names, aliases (`deit-b`, `DeiT-B`, ...) and timm ids to backbones. `leaf backbones` lists it, and `LEAF_BACKBONES=my_backbones.json` adds your own ViTs.
- **Progress logs.** Commands log progress with time, elapsed time and ETA on stderr; results stay on stdout.
- **LEAF-M0 campaign code removed** (`legacy_campaigns/`). It stays in the research repository. Its forecast-quality experiment is ported (`leaf experiments run forecast_quality`) and reproduces the stored results exactly.

- **`leaf search`.** The whole schedule search in one workspace folder: group → probe → budgets → workloads → time → fit → plan.
  - It resumes after an interruption: steps whose output exists are skipped.
  - `--gpus 0,1` shards the probes over the GPUs; timing runs on one GPU.
- **`leaf budgets`.** Times anchor schedules on this GPU and derives latency budgets with the paper's rule. Anchors: DiffRate (0.97 × latency, plus two extras), LEAF GMAC points, dense fractions or any configs.
- **`leaf workloads`.** Builds SALT timing workloads for a new backbone from its probes.
- **Probes.**
  - `leaf probe` keeps earlier grids in `probes.json` and probes only new grid points.
  - `--reuse` links the runs of another probe folder.
  - `leaf plan --operator M0` plans with the M0 merge.
- **α form everywhere.**
  - The schedule sets in `schedules/` are in the α form; the campaigns' κ exports moved to `schedules/kappa_form/`.
  - The M0 and calibration-seed probe tables are in `schedules/leaf/planning/<bb>/`.
  - `tools/alpha_form.py` builds an α-keyed cache of the paper's probe runs.
- **Fix.** `merge_norm_evaluation` read LEAF-M0 schedules from a path that no longer existed.

## 1.0.0 (2026-10-21)

First release of the standalone repository.

- **Method package `leaf`.** Stations with the alpha score `(1 - alpha) z(log P) + alpha z(D)`, the G16/r32 forecast D (Triton), the M0 and M3 merge operators, the station-aware latency estimate SALT, and the Lagrangian DP planner.
- **Command line.**
  - Released schedules: `leaf schedules | eval | bench`.
  - Planning: `leaf data | group | probe | time-workloads | fit-latency | plan`.
  - Paper reproduction: `leaf experiments list | run | analyze | tables | figures | all`.
- **Released schedules.** The paper's schedules for seven backbones (48 latency-planned, 48 GMAC-planned), with measured accuracy and latency and the planning inputs (`schedules/leaf/`).
- **Experiments.** Every experiment of the paper (`leaf/reproduction/`), ported from the research repository with descriptive names. Every analysis, table and figure is regenerated identically from the stored results (see PROVENANCE.md).
