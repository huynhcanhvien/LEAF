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
| `test_gmac_planner.py` | GMAC-only DP and cost matching | CPU |
| `test_alpha.py` | the alpha score orders tokens like kappa = alpha / (1 - alpha); alpha kernel against torch | CPU (+ GPU if idle) |
| `test_merge_norm.py` | M3 (MLERP) merge against the reference | CPU |
| `test_gpu_coordination.py` | GPU lock and exclusive timing | CPU |
| `test_reduced_vit.py` | keeping all tokens reproduces the dense model; fused merge kernel = scatter path | GPU |
| `test_attention_forecast.py` | forecast D: folding, grouping and low rank against the reference | GPU |
| `test_determinism.py` | bit-identical predictions across runs and processes | GPU |
| `test_clip.py` | CLIP ViT-B/16: zero-shot head, timm tower = open_clip, keep-all = dense, DiffRate (transferred) and baselines run, retrieval ranks/recall, `clip_comparison` plan/pairing/report on synthetic data; with the caches, dense zero-shot top-1 and Flickr R@1 | CPU, timm/open_clip weights |
| `test_clip_l14.py` | CLIP ViT-L/14 (patch 14, 257 tokens): GMAC, SALT constants and probe grid follow the patch and token count while every 197-token backbone keeps the paper's values; MAE-L DiffRate rates scaled to 257 tokens; campaign helpers on 24 blocks; timm tower = open_clip, keep-all = dense, every baseline runs | CPU, timm/open_clip weights |
| `test_segmentation.py` | Segmenter ViT-B/16 (ADE20K): the converted MMSegmentation checkpoint = an independent MMSegmentation forward, keep-all = dense, source map = reference trace (prune, merge, ToMe unmerge), window grid / stitching / area statistics / mIoU = MMSegmentation, GMAC accounting, the DP at 1,025 tokens; with a GPU, the segmented merge kernel used beyond 257 tokens | CPU (+ GPU if idle), the Segmenter checkpoint |
| `test_baselines.py` | baseline adapters against the official code (`official_compare.py`) | `OFFICIAL_REPOS=<dir>`, otherwise skipped |

## Experiments (`leaf/reproduction/`)

| Test | Checks | Needs |
|---|---|---|
| `test_ablation_and_alpha_grid.py` | the planner copy with ablation switches equals the planner; arm construction | `results/` |
| `test_horizon_norm_ablation.py`, `test_merge_norm_campaign.py`, `test_merge_variants.py`, `test_merge_variants_protocol.py`, `test_latency_pricing.py` | campaign logic: pairing, sealing, statistics, norm variants | CPU; some read `results/` |

`test_latency_pricing.py` skips one check, which verified the campaign's source files at their old paths in the research repository (see PROVENANCE.md).

```bash
python tests/test_leaf_cli.py                                   # DeiT-T and DeiT-B (about 1 min on CPU)
python tests/test_leaf_cli.py deit_t deit_s deit_b augreg_s augreg_b mae_b mae_l
```
