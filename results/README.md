# Raw results of the paper's campaigns

This folder is not part of the git repository (about 7 GB in 170k files). It holds what the campaigns wrote:
- sealed plans;
- per-image predictions, correctness and KL divergence (`<split>/runs/<config_sha>.{json,npz}`);
- timings (`latency/*.json`);
- the analysis outputs.

`leaf report analyze | tables | figures` read it.

| Folder | Experiment (`leaf/reproduction/experiments/<name>.py`) | Used for |
|---|---|---|
| `main_comparison` | `main_comparison` | baselines, DiffRate, LEAF on the 4-value alpha grid, ablation and seed arms, main timing session |
| `ablation_and_alpha_grid` | `ablation_and_alpha_grid` | **the paper's LEAF schedules** (group `leaf`), component ablations, setting grids, surrogate check |
| `main_comparison_analysis` | `main_comparison_analysis` | latency pairing, McNemar / Holm, latency transfer, numbers of the text |
| `equal_gmac`, `imagenet_real`, `final_station_and_ci` | same names | equal-GMAC comparison, ReaL re-scoring, final-block station, confidence intervals |
| `latency_pricing` | `latency_pricing` | SALT vs GMAC planning on GPU bs4/8 and CPU bs1/8 |
| `alpha_kappa_equivalence` | `alpha_kappa_equivalence` | alpha and kappa scores give identical predictions |
| `forecast_approximation`, `forecast_rank` | same names | forecast quality of the G16/r32 approximation |
| `token_schedule`, `search_cost` | same names | token figure data, search-cost throughput |
| `gmac_planning` | `gmac_planning` | probes (5,000 calibration images), dense teachers, GMAC-planned schedules |
| `low_budget_and_factorial`, `horizon_norm_seed_ablation` | same names | probe curves used by the planner, very low budgets, horizon / norm / seed ablations |
| `salt_fit`, `salt_validation`, `salt_m3_pilot` | `salt_*` | fit and validation of the latency model SALT |
| `merge_variants` | `merge_variant_screening`, `merge_norm_*` | merge variants, the M3 merge |
| `leaf_m0`, `leaf_m0_forecast_quality`, `leaf_m0_summaries` | LEAF-M0 campaign; `forecast_quality` | the earlier variant LEAF-M0 |
| `figures`, `paper_build`, `reports`, `logs` | (outputs) | working outputs of the generators and campaigns |

## Getting the results

The per-image correctness of every evaluated configuration, the raw timings and the analysis reports are released in `release/LEAF_raw_results_v1.zip` (unzip it and run `python verify.py`). The complete raw results are not distributed with the repository; they are available from the authors on request.

`figures/` and `paper_build/` are working outputs; `leaf report tables` and `leaf report figures` recreate them.

**From scratch.** Re-running the campaigns from the sealed configurations in `schedules/` regenerates everything on two RTX-class GPUs: the probes take about 35 GPU-hours, and each timing session needs the machine to itself.

**Note on `leaf report analyze`.** `analyze` rewrites the analysis outputs in place. Floating-point sums can then differ from the stored files in the last digits (at most about 2e-6 relative); the tables and figures, which are what the paper reports, are unchanged.
