# Assets

- `figures/`: PNG renderings of figures of the paper, for the README.
- `schedules/`: one **schedule book** per backbone, with every schedule the paper evaluated on ImageNet and the paper's numbers.

## Figures

| | |
|---|---|
| ![LEAF at a glance](figures/method_overview.png) | **Method overview** (Fig. 1) |
| ![Accuracy-latency on ImageNet val](figures/accuracy_latency_imagenet_val.png) | **Accuracy vs latency, ImageNet val**, seven backbones, nine re-run families |
| ![Token schedule](figures/token_schedule_deit_b_10gmac.png) | **Token schedule** of a GMAC-planned DeiT-B schedule (10 GMAC) |

## Schedule books

`schedules/<backbone>.json` has the format of a released schedule file. Every key therefore runs with the usual commands, and they print the paper's number next to the new one (`paper_top1`, `paper_ms_bs32`):

```bash
leaf schedules assets/schedules/deit_b.json                                    # list: key, ms, val, V2, stations / config
CUDA_VISIBLE_DEVICES=0 leaf evaluate  --schedule assets/schedules/deit_b.json --key main/diffrate/DR8.7 --split val
CUDA_VISIBLE_DEVICES=0 leaf evaluate  --schedule assets/schedules/deit_b.json --key ablation/forecast_only/d_only/extra2 --split v2
CUDA_VISIBLE_DEVICES=0 leaf benchmark --schedule assets/schedules/deit_b.json --key main/tome/r16      # nothing else running
leaf schedules assets/schedules/deit_b.json --key main_ablation/no_forecast/DR8.7 --json          # config, numbers, tables
```

**Accuracy** is deterministic: BF16 at batch 32, with the same weights and data cache. Re-evaluating an entry gives the paper's top-1 exactly.

Checked on 43 entries, one of each config kind and section on DeiT-B: 42 give the paper's val top-1 exactly. The exception is MCTF (`main/mctf/r16`: 80.094 vs 80.088, 3 of 50,000 images). Its stored run predates the deterministic mode, and MCTF's official merge uses `scatter_reduce` with atomics. The original code at the tag `pre-reorg-2026-10-20` gives the same 80.094 today.

**Latency** depends on the machine. The paper timed on one RTX 5060 Ti at batch 32, as one CUDA graph, with the whole machine idle. On the same card, expect the paper's milliseconds within a few percent.

**Fields of an entry:**
- `config`: alpha form (`pda`), or the baseline's own config (`tome`, `evit`, `tofu`, ...);
- `stations`;
- `measured`:
  - `val`, `v2`: top-1 in %;
  - `ms_bs32` and `sd`: latency in the entry's campaign session; `latency_pricing` entries have `latency_gpu` and `latency_cpu` per batch size instead;
  - `gmac`;
- `paper`: the tables and figures (`fig:`) of the paper that use the entry;
- `campaign`, `source_key` and `source_config_sha`: the campaign, the arm name and the configuration hash it was evaluated under (the hashes of `release/`).

**Sections** (key prefixes):

| Prefix | What | Paper |
|---|---|---|
| `leaf/latency/<budget>` | LEAF, the paper's method (= `schedules/leaf/`) | main results, latency pairs, Fig. accuracy–latency |
| `leaf/gmac/<budget>` | LEAF planned under a GMAC budget | equal-GMAC tables and figure |
| `main/<family>/<point>` | DiffRate, ToMe, ToMe+M3, PiToMe, EViT, PPT, MCTF, ToFu, RAPID, Pruning One More Token, LEAF-M0 | main results, latency pairs, equal GMAC, Fig. accuracy–latency |
| `main_ablation/<arm>/<budget>`, `calibration_seed/<seed>/<budget>`, `leaf_alpha4/<budget>` | component ablations, other calibration draws, LEAF on the four-value alpha grid | Tab. ablation |
| `ablation/<group>/...` | schedule structure, shared settings, forecast only, setting grids, forecast compression, probe sample size, final station, fixed station count | Tabs. ablation structure, grid sensitivity |
| `low_budget/...` | very low budgets and the D × M3 factorial | Tabs. LEAF vs LEAF-M0, negative variants |
| `horizon/...` | forecast horizon, norm variants, D placebos | ablation, negative variants |
| `latency_pricing/...` | SALT- and GMAC-planned per device regime (GPU bs4/8/32, CPU bs1/8) | Tab. latency pricing, Fig. SALT vs GMAC |
| `merge_variants/...` | the M3 and N4 merges planned and evaluated | negative variants |
| `gmac_planned/...` | GMAC-planned DP with M0 / M3 | equal GMAC, Fig. token schedule |
| `leaf_m0/...` | the earlier variant LEAF-M0 | LEAF vs LEAF-M0, M0 vs DiffRate |

**Number of schedules per backbone:**

| Backbone | total | `leaf` | `main` | `leaf_alpha4` | `main_ablation` | `calibration_seed` | `ablation` | `low_budget` | `horizon` | `latency_pricing` | `merge_variants` | `gmac_planned` | `leaf_m0` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| deit_t | 249 | 14 | 82 | 7 |  |  | 35 | 78 |  |  |  | 21 | 12 |
| deit_s | 299 | 14 | 78 | 7 |  |  | 35 | 78 |  | 54 |  | 21 | 12 |
| deit_b | 532 | 14 | 80 | 7 | 56 | 14 | 130 | 99 | 24 | 55 | 20 | 21 | 12 |
| augreg_s | 290 | 14 | 76 | 7 |  | 14 | 33 | 94 | 24 |  |  | 21 | 7 |
| augreg_b | 230 | 14 | 73 | 7 |  |  | 35 | 73 |  |  |  | 21 | 7 |
| mae_b | 470 | 12 | 75 | 6 | 48 | 12 | 107 | 98 | 24 | 44 | 16 | 18 | 10 |
| mae_l | 222 | 14 | 71 | 7 |  |  | 21 | 76 |  |  |  | 21 | 12 |

A schedule evaluated in two campaigns appears once per campaign, each with its own timing session. Entries of the `ablation` group that equal the method's schedule are marked `same_as_leaf`.
