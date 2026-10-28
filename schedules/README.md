# Schedules

## `leaf/`: the paper's method

One file per backbone, `leaf/<backbone>.json`, with the paper's numbers.

| File | Backbone | timm checkpoint | Schedules (latency + GMAC) |
|---|---|---|---:|
| `deit_t.json` | DeiT-T | `deit_tiny_patch16_224.fb_in1k` | 7 + 7 |
| `deit_s.json` | DeiT-S | `deit_small_patch16_224.fb_in1k` | 7 + 7 |
| `deit_b.json` | DeiT-B | `deit_base_patch16_224.fb_in1k` | 7 + 7 |
| `augreg_s.json` | ViT-S (AugReg) | `vit_small_patch16_224.augreg_in21k_ft_in1k` | 7 + 7 |
| `augreg_b.json` | ViT-B (AugReg) | `vit_base_patch16_224.augreg2_in21k_ft_in1k` | 7 + 7 |
| `mae_b.json` | ViT-B (MAE) | official MAE fine-tuned weights | 6 + 6 |
| `mae_l.json` | ViT-L (MAE) | official MAE fine-tuned weights | 7 + 7 |
| `augreg_b384.json` | ViT-B/16 at 384 px (AugReg, 577 tokens) | `vit_base_patch16_384.augreg_in21k_ft_in1k` | 6 + 0 |

Added for the journal revision:
- `augreg_b384.json`: latency budgets 0.40-0.80 x the dense batch-32 latency, token-count grid step 8; register the backbone with `LEAF_BACKBONES=leaf/backbones_vitb384.json` and build its 384-px cache with `leaf data prepare --split val --side 384 --crop-pct 1.0 --out-dir data/cache384` (pass `--data-cache data/cache384` to `leaf evaluate`). Planning inputs in `leaf/planning/augreg_b384/`.
- `deit_b_coco_probes.json`: the seven DeiT-B latency schedules re-planned from probes on 1,000 unlabelled MS-COCO images (`leaf/planning/deit_b_coco1000/probes.json`; groups, SALT model and budgets of `deit_b`), with `same_as_released` and the paired differences to the released schedules.

Use the checkpoint recorded in the file (`timm_id`); `leaf evaluate/bench --schedule ...` does this automatically. For `augreg_b` it differs from `leaf.backbones.SPECS['augreg_b']`.

**Top level:**
- `backbone`, `timm_id`;
- `forecast` (G = 16 head groups, rank 32), `operator` (M3), `alpha_grid`, `theta_grid`, `hardware`;
- `dense`: top-1 on ImageNet val and V2, and bs32 latency;
- `sources`: the result files the entries come from.

**Entries** of `schedules`, keyed `<budget kind>/<budget id>`:
- **`latency/<id>`** is the paper's method: planned under the latency budget (ms, bs32) with SALT and alpha in {0, 1/4, 1/3, 1/2, 2/3, 3/4, 1}.
- **`gmac/<id>`** is planned under a GMAC budget with alpha in {0, 1/2, 2/3, 3/4}. The paper uses these in the equal-GMAC comparison and in the comparison with published results.
- **`DRx` budgets** are anchored at the released DiffRate schedule of x GMAC (0.97 × its measured latency, or its actual MACs).
- **`extra1` and `extra2`** are faster than the fastest anchor; `extra2` is the faster of the two.
- **`dense0.xx`** budgets are AugReg budgets, a fraction of the dense latency.

**Fields of an entry:**
- `stations`: `block` (0-based), `tokens_after` (including the class token), `theta` (share of the removed tokens that is merged) and `alpha` (forecast weight of `(1 - alpha) z(log P) + alpha z(D)`).
- `config`: the runtime configuration (kind `pda`), for `leaf.api.build_model` / `LeafRunner.run`. The head-group labels are embedded, and the forecast matrices are recomputed from the backbone weights.
- `planner`: surrogate distortion F̂, predicted bs32 latency (SALT), and GMAC including station and forecast operations.
- `measured`: top-1 on ImageNet val (50k) and V2 (10k), and bs32 latency (mean of three interleaved rounds, with SD). For `latency/` entries, `latency_transfer` is the factor that carried the ablation-session timing to the main session (null: timed in the main session).

**Planning inputs** are in `leaf/planning/<backbone>/`:
- `probes.json`: probe means, in planning order;
- `groups.json`: head groups;
- `salt.json`: the SALT model;
- `salt_workloads.json`: the timed workloads SALT was fitted on;
- `budgets.json`: the paper's latency budgets;
- `probes_m0.json`: probe means with the M0 merge (`leaf plan --operator M0`; the M0-operator ablation);
- `probes_seed251.json`, `probes_seed351.json` (DeiT-B, MAE-B, AugReg-S): probe means on other calibration draws (the seed ablation).

`probes_m0.json` and `probes_seed*.json` hold α ∈ {0, 1/2, 2/3, 3/4}. With `--alphas 0 1/2 2/3 3/4`, `leaf plan` gives the paper's `ablation/m0_operator/*` and `seed251|seed351/*` schedules from them exactly.

`leaf plan` reproduces every released schedule from these inputs exactly (`tests/test_leaf_cli.py`).

## Schedule exports of the other campaigns

Every set below holds `pda` configs and `alpha` per station. `config_sha` is the hash of the config as released; `source_config_sha`, where present, is the hash the campaign recorded for the same schedule. Any arm of these files, and of the sealed plans in `journal/`, runs with `leaf evaluate | benchmark --schedule <file> --key <arm>` (`leaf schedules <file>` lists the arms).

| Folder | Campaign | Content |
|---|---|---|
| `main_comparison/` | `main_comparison` | LEAF on the four-value alpha grid (`leaf_alpha4/`), ablation and seed arms |
| `main_comparison_plans/` | `main_comparison` | every planned job of the main campaign (sealed plans), baselines and references included |
| `gmac_planned/` | `gmac_planning` | GMAC-planned DP with M0 / M3 |
| `latency_pricing/` | `latency_pricing` | SALT- and GMAC-planned schedules per device regime |
| `low_budget_and_factorial/` | `low_budget_and_factorial` | very low budgets, D × M3 factorial |
| `horizon_norm_seed_ablation/` | `horizon_norm_seed_ablation` | horizon, norm and seed variants |
| `leaf_m0/` | LEAF-M0 campaign | the earlier variant LEAF-M0 (its own keys) |

## `journal/`: sealed configurations of the journal-version experiments

CLIP, the T4 transfer, black-box search baselines, ViT-B/16 at 384 px, calibration-set size, out-of-domain probes, the planner-candidate check, the average-pooled-head rule, the station discount and SALT below its timing range: every evaluated arm's config with its SHA-256, the timing designs, budgets and SALT models. See `journal/README.md`.
