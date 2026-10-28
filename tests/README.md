# Tests

Tests are plain scripts: `python tests/<file>.py`, run from the repository root. Each prints `PASS` / `FAIL` lines and exits with 1 on any failure (`tests/common.py`). Select the GPU with `CUDA_VISIBLE_DEVICES`.

## Method

| Test | Checks | Needs |
|---|---|---|
| `test_leaf_cli.py` | `leaf plan` reproduces the released latency- and GMAC-planned schedules exactly; `leaf fit-latency` reproduces the released SALT models | CPU, `schedules/leaf/` |
| `test_search_cli.py` | `leaf budgets` rule = the paper's budgets; `leaf workloads`; probe grids merged and cached; `leaf plan --operator M0` and the seed probe tables give the paper's ablation schedules; `leaf search --dry-run` on one and two GPUs | CPU, `schedules/leaf/` |
| `test_schedule_books.py` | the schedule books in `assets/schedules/`: complete entries, `leaf/` = release, DiffRate rows = paper | CPU, `assets/` |
| `test_planner.py` | DP exactness, budget search, surrogate tables | CPU |
| `test_operator_aware_planning.py` | SALT features, operator configs, `plan_latency` | CPU |
| `test_gmac_planner.py` | GMAC-only DP | CPU |
| `test_merge_norm.py` | M3 (MLERP) merge against the reference | CPU |
| `test_gpu_coordination.py` | GPU lock and exclusive timing | CPU |
| `test_reduced_vit.py` | keeping all tokens reproduces the dense model; fused merge kernel = scatter path | GPU |
| `test_prefix_cache.py` | `leaf probe --prefix-cache` gives bit-identical probes to the full forward | GPU |
| `test_attention_forecast.py` | forecast D: folding, grouping and low rank against the reference | GPU |
| `test_determinism.py` | bit-identical predictions across runs and processes | GPU |
| `test_baselines.py` | baseline adapters against the official code (`official_compare.py`) | `OFFICIAL_REPOS=<dir>`, otherwise skipped |

## Model variants

| Test | Checks | Needs |
|---|---|---|
| `test_horizon_norm_ablation.py`, `test_merge_variants.py` | the forecast-horizon and norm-target ablation models and the merge variants against their references | CPU (+ GPU if idle) |

```bash
python tests/test_leaf_cli.py                                   # DeiT-T and DeiT-B (about 1 min on CPU)
python tests/test_leaf_cli.py deit_t deit_s deit_b augreg_s augreg_b mae_b mae_l
```
