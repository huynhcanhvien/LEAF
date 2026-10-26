# LEAF: Latency-Budgeted Efficient Planning with Attention Forecasts for Training-Free Token Reduction

LEAF speeds up a **frozen** Vision Transformer by removing tokens between blocks. Nothing is trained or fine-tuned.
For a given backbone, device and latency budget, it plans the whole token schedule:
- **where** to reduce (which blocks);
- **how many** tokens to keep;
- **how** to remove them (the share pruned rather than merged);
- **which** tokens to remove (how much the score trusts a forecast of future attention).

Planning is an exact dynamic program over measured single-station probes and a station-aware latency estimate. Once the probes exist, it takes seconds.

This repository holds the method (`leaf/`), every experiment of the paper (`leaf/reproduction/`), the released schedules (`schedules/`) and the generators of the paper's tables and figures (`leaf report`, `tools/`). The manuscript is under review at *Knowledge-Based Systems*; see [Citation](#citation).

![From an image to a planned token schedule](assets/figures/token_schedule_deit_b_10gmac.png)

*From an image to a planned token schedule.* LEAF on DeiT-B at a 10-GMAC budget (actual cost: 9.93 GMAC) chooses an initial pruning station followed by three merging stations. Columns show the input and all stations before the final block; counts exclude the class token. Each surviving token is drawn as one square at its representative position: an unmerged token shows its original patch, and a merged token shows the mean image colour of all its source patches inside a coloured border. Every other position, whether pruned or merged into another token, is black, so the number of squares equals the number of tokens and falls from 163 to 52. The final post-attention station is omitted here but executed and included in the cost. Source membership is traced as in DiffRate and ToMe; the squares show membership, not decoded features.

![LEAF at a glance](assets/figures/method_overview.png)

*LEAF at a glance.* **(a)** The planner probes one station at a time on unlabelled images and fits a latency model on the target GPU. Both estimates add up over blocks, so a dynamic program finds the schedule that best trades damage for latency, and bisection on the latency price meets the budget. **(b)** A schedule fixes the blocks that host stations and the number of tokens kept after each. **(c)** Each station scores tokens by present attention plus a forecast of future attention, and removes Δ = n<sub>t−1</sub> − n<sub>t</sub> tokens before the block's MLP: it prunes the lowest-scoring (1 − θ<sub>t</sub>)Δ and merges the next θ<sub>t</sub>Δ (rounded) into the most similar kept tokens.

## Contents
- [Method in brief](#method-in-brief)
- [Results](#results)
- [Installation](#installation)
- [Data preparation](#data-preparation)
- [Use a released schedule](#use-a-released-schedule)
- [Plan schedules for a new budget, device or backbone](#plan-schedules-for-a-new-budget-device-or-backbone)
- [Reproduce one number of the paper](#reproduce-one-number-of-the-paper)
- [Reproduce the paper](#reproduce-the-paper)
- [Repository layout](#repository-layout)
- [Names used in files and results](#names-used-in-files-and-results)
- [Tests](#tests)
- [Limitations](#limitations)
- [Provenance and acknowledgements](#provenance-and-acknowledgements)
- [Citation](#citation)
- [License](#license)

## Method in brief

**Station.** A station sits between the attention and the MLP of a block. It:
1. scores the patch tokens;
2. prunes the lowest-scoring ones;
3. merges a further share θ of the removed tokens into their most similar kept tokens.

The merge is the norm-preserving MLERP rule (M3), and attention is proportional to token size. The class token is never removed.

**Score.** A station ranks tokens by

```
S_j = (1 - α) · z(log P_j) + α · z(D_j),        α ∈ {0, 1/4, 1/3, 1/2, 2/3, 3/4, 1}
```

- `P` is the present class-token attention (the EViT criterion).
- `D` forecasts the class-token attention that later blocks will pay, by applying the query-key maps of all later blocks to the current tokens.
- `D` is made cheap in three steps:
  1. folding the weights (exact);
  2. grouping similar heads, G = 16 (an approximation);
  3. truncating to rank r = 32 (an approximation).

  Two Triton kernels then compute it in at most 51 µs per station at batch 32.

**Planner.**
1. **Probes.** On 5,000 unlabelled training images, LEAF measures the KL divergence to the dense model of every single station: block, number of removed tokens, θ ∈ {0, ½, 1} and α. That is 1,917 probes for a 12-block model. Incremental differences of these curves give a distortion surrogate that is separable along depth.
2. **SALT.** The station-aware latency estimate is a non-negative linear model over per-block GMAC and per-station launch and traffic features. It is fitted to about 30–50 timed workloads on the target device and batch size.
3. **Lagrangian DP.** For each latency price, an exact dynamic program over (block, token count) states minimises distortion plus priced time; the number of stations comes out of the solution. A bisection on the price meets the budget, and a final rescaling of the keep ratios uses the remaining slack.

All shapes are fixed per schedule, so a planned model runs as one CUDA graph.

## Results

Every method was re-run on the same weights, data pipeline and BF16 precision. Latency is measured on one NVIDIA RTX 5060 Ti at batch 32 as one CUDA graph, as the mean of three interleaved rounds in one session per backbone.

**DeiT-B** (dense: 81.83 val / 70.95 V2, 32.2 ms). LEAF is planned under 0.97 × the measured latency of each released DiffRate schedule.

| Budget | Method | GMAC | ms ↓ | ImageNet val ↑ | ImageNet-V2 ↑ |
|---|---|---:|---:|---:|---:|
| DR8.7 | DiffRate | 8.74 | 18.81 | 79.11 | 67.44 |
| | **LEAF** | 9.00 | **17.97** | **80.14** | **68.56** |
| DR10.0 | DiffRate | 9.99 | 21.49 | 80.66 | 69.40 |
| | **LEAF** | 10.30 | **20.67** | **80.78** | **69.45** |
| DR11.5 | DiffRate | 11.53 | 24.26 | **81.48** | **70.76** |
| | **LEAF** | 11.64 | **23.34** | 81.44 | 70.15 |
| DR12.5 | DiffRate | 12.53 | 26.91 | 81.71 | 71.01 |
| | **LEAF** | 13.11 | **26.77** | 81.71 | 71.00 |

The paper covers seven backbones (DeiT-T/S/B, ViT-S/B AugReg, MAE ViT-B/L) and 48 budgets, against ten re-run families: DiffRate, ToMe, ToMe+MLERP (`ToMe+M3` in file names), PiToMe, EViT, PPT, MCTF, ToFu, Pruning One More Token and RAPID (MCTF is slower than the dense model, so nine families have latency-matched points).
- At equal measured latency, LEAF is significantly more accurate than the re-run method in 64 of 68 pairs on ImageNet val (McNemar, Holm-corrected) and never significantly less accurate; it dominates 7 of the 24 released DiffRate schedules.
- At equal GMAC it uses 4–8 stations where DiffRate uses 9–22 and is faster in 22 of 24 pairs.
- Where latency departs from FLOPs, planning by measured latency beats planning by GMAC in all 12 batch-32 pairs on a Tesla T4 (only SALT refitted) and by up to 2.3 points on a CPU.
- With the same code and settings on CLIP ViT-B/16, LEAF wins all 43 latency pairs on zero-shot accuracy, improves retrieval that the planner never sees, and is more accurate than DiffRate's own search on CLIP at equal latency.

![Accuracy vs latency on ImageNet val](assets/figures/accuracy_latency_imagenet_val.png)

The paper's schedules and their measured accuracy and latency are in two places:
- [`schedules/leaf/`](schedules/README.md): the method's released schedules;
- [`assets/schedules/`](assets/README.md): one book per backbone with **every schedule the paper evaluated**: LEAF, DiffRate and every re-run baseline point, ablation arms, other campaigns.

Every entry runs with `leaf evaluate` / `leaf benchmark`, and prints the paper's number next to yours (see [Reproduce one number of the paper](#reproduce-one-number-of-the-paper)). More figures are in [`assets/`](assets/README.md).

## Installation

Tested with Python 3.12, PyTorch 2.14 (CUDA 13.0), timm 1.0.30 and Triton (bundled with PyTorch), on Linux with an NVIDIA RTX 5060 Ti.

```bash
git clone https://github.com/huynhcanhvien/LEAF.git leaf && cd leaf
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -e ".[paper]"            # the `leaf` package and command; [paper] adds matplotlib for the figures
cp .env.example .env                 # HF_TOKEN, only for downloading ImageNet
```

Every command can be run as `leaf <command>` or as `python -m leaf <command>` from the repository root. `leaf backbones` lists the backbones and the names every `--backbone` accepts (`deit_b`, `deit-b`, `DeiT-B` or the timm id); the list is `leaf/backbones.json`, and `LEAF_BACKBONES=my_backbones.json` adds your own. Commands that read or write `data/`, `results/`, `schedules/` or `paper/` use paths relative to the repository root, so run them from there. Symbolic links are fine, for example `ln -s /big/disk/cache data/cache`.

## Data preparation

All splits are decoded once into uint8 memory maps (Resize 256 bicubic → CenterCrop 224) under `data/cache/`. Normalisation happens on the GPU for each backbone.

```bash
leaf data download                               # ImageNet-1k WebDataset shards (timm/imagenet-1k-wds) -> data/imagenet_tar
leaf data prepare --split val                      # 50,000 validation images
leaf data prepare --split calib_bal                # 10 images/class from train shards 900-999; even indices = the 5,000 probe images
leaf data prepare --split calib                    # first 10k images of train shards 1000+; the first 1,024 are used for head grouping
leaf data prepare --split confirm                  # train shards 1008-1023 (confirmation set of the paper's protocol)
leaf data prepare --split v2 --v2-tar imagenetv2-matched-frequency.tar.gz
```

The planner uses training images only. To download ImageNet, accept the ImageNet licence on Hugging Face first. ImageNet-ReaL labels are included in `data/real/`.

## Use a released schedule

```bash
leaf schedules schedules/leaf/deit_b.json                                  # stations, budget, measured latency and accuracy
CUDA_VISIBLE_DEVICES=0 leaf evaluate  --schedule schedules/leaf/deit_b.json --key latency/DR10.0 --split val
CUDA_VISIBLE_DEVICES=0 leaf benchmark --schedule schedules/leaf/deit_b.json --key latency/DR10.0 --batches 1 32 128
CUDA_VISIBLE_DEVICES=0 leaf benchmark --backbone deit_b --dense                # reference in the same session
```

`eval` reproduces the released accuracy exactly. On DeiT-T `latency/DR0.8` it gives the same per-image predictions on all 50,000 validation images as the paper. Latency depends on the device: the released budgets hold on an RTX 5060 Ti at batch 32.

```python
from leaf.api import load_schedule, LeafRunner
from leaf.latency.timing import measure_latency
doc, entry = load_schedule('schedules/leaf/deit_b.json', 'latency/DR10.0')
runner = LeafRunner('deit_b', root='runs/leaf/deit_b', timm_id=doc['timm_id'])   # batch 32, data/cache
arr = runner.run(entry['config'], 'val')                                       # per-image correct / pred / KL
print(100 * arr['correct'].mean(), measure_latency(runner.inference_model(entry['config']), 32)['ms_mean'])
```

## Plan schedules for a new budget, device or backbone

**A new budget on the paper's setup** (CPU, seconds). `--budgets` with the released `budgets.json` reproduces every released schedule exactly.

```bash
P=schedules/leaf/planning/deit_b
leaf plan --backbone deit_b --probes $P/probes.json --groups $P/groups.json --salt $P/salt.json --budget-ms 19.5 --out runs/plans/deit_b.json
leaf plan --backbone deit_b --probes $P/probes.json --groups $P/groups.json --gmac 9.0 --out runs/plans/deit_b_gmac.json
```

**A new device or batch size.** The probes do not depend on the device, so only the timed workloads and the SALT fit are redone:

```bash
CUDA_VISIBLE_DEVICES=0 leaf time-workloads --backbone deit_b --batch 8 --workloads $P/salt_workloads.json --out runs/salt/deit_b_bs8.json
leaf fit-latency --backbone deit_b --timings runs/salt/deit_b_bs8.json --out runs/salt/deit_b_bs8_model.json
leaf plan --backbone deit_b --probes $P/probes.json --groups $P/groups.json --salt runs/salt/deit_b_bs8_model.json --budget-ms 6 --out runs/plans/deit_b_bs8.json
```

**A new backbone, GPU or grid: `leaf search`.** One command runs the whole search in a workspace folder:
- `group`: head groups of the forecast D;
- `probe`: single-station probes on 5,000 training images, sharded over the GPUs;
- `budgets`: anchor schedules timed on this GPU;
- `workloads`, `time`, `fit`: SALT fitted to timed workloads;
- `plan`: the Lagrangian DP at every budget.

```bash
leaf search --backbone deit_s --workspace runs/search/deit_s --gpus 0,1      # two GPUs: one probe shard per GPU
leaf search --backbone deit_s --workspace runs/search/deit_s --gpus 0        # one GPU
leaf search ... --dry-run                                                    # print the commands only
```

The workspace keeps every output:
- `groups.json`;
- `probes/`: a cache of every probe run, and `probes.json`;
- `budgets.json`;
- `salt_workloads.json`, `salt_timings.json` and `salt.json`;
- `schedules.json`, which works with `leaf schedules`, and with `leaf evaluate|benchmark --schedule`;
- `search.json`: the commands that were run.

A step whose output exists is skipped, so an interrupted search resumes where it stopped; `--redo STEP` runs a step again.

**Custom grids.**
- `--thetas` and `--alphas` set the planning grid (fractions such as `1/3` are allowed), and `--deltas` sets the probed token cuts.
- Probes are cached one by one in the workspace: rerunning with a larger grid probes only the new points.
- `probes.json` keeps every grid probed in that workspace; the planner uses the grid of the current call.

**Budgets.** `leaf budgets` measures anchor schedules on this GPU (bs32, three interleaved rounds, each run warmed up) and derives the budgets as in the paper. `--anchors` chooses the anchors:
- `diffrate`: DiffRate's released schedule at each of its GMAC points, budget = 0.97 × its latency, plus `extra1` and `extra2` below the fastest. This is the default when DiffRate has schedules for the backbone.
- `dense`: fractions of the dense latency (the AugReg budgets). This is the default otherwise.
- `gmac --gmac-points 8 10 12`: LEAF schedules GMAC-planned at your points.
- `configs --anchor-configs FILE`: any configs.

`leaf search` also takes `--budgets FILE`, `--budget-ms ...` and `--gmac ...`.

**Latency model.**
- `leaf workloads` plans about 45 timing workloads from the probes: GMAC-planned schedules at eight GMAC points in five variants (M3, M0, no forecast, pruning only, merging only), plus DiffRate's schedules.
- `leaf time-workloads` times them, and `leaf fit-latency` fits SALT.
- `--salt FILE` reuses a model fitted on the same GPU; `--workloads FILE` times another set, for example the paper's `salt_workloads.json`.

**Timing protocol.** Probes may use every GPU. Timing steps (`budgets`, `time`) run on the first GPU of `--gpus` after the probes have finished, and every GPU command refuses a card that another process uses. Keep the machine free of other jobs while timing.

**Reusing the paper's probes.**
- On a paper backbone with the released `groups.json`, `--reuse-probes results/probe_runs_alpha/<bb>` links the paper's probe runs instead of running them again. This needs the per-probe runs in `results/` (raw results, available on request) and `python tools/alpha_form.py probe-cache`; without them, `leaf probe` measures the probes.
- `leaf search --backbone deit_s --groups schedules/leaf/planning/deit_s/groups.json --reuse-probes results/probe_runs_alpha/deit_s --salt schedules/leaf/planning/deit_s/salt.json --budgets schedules/leaf/planning/deit_s/budgets.json` reproduces the seven released DeiT-S schedules without a GPU.

The single steps are also commands (`leaf group-heads | probe | budgets | workloads | time-workloads | fit-latency | plan`); every one has `--help`.

**Other planning inputs of the paper.** `schedules/leaf/planning/<bb>/` also holds:
- `probes_m0.json`: probes with the M0 merge, for `leaf plan --operator M0`;
- `probes_seed251.json` and `probes_seed351.json`: other calibration draws, for DeiT-B, MAE-B and AugReg-S.

Planned with `--alphas 0 1/2 2/3 3/4`, they give the paper's M0-operator and calibration-seed ablation schedules exactly.

Backbones are defined in `leaf/backbones.py` (`SPECS`); a timm ViT with a class token needs one line. Every command has `--help`.

## Reproduce one number of the paper

No raw results are needed: the schedule books in `assets/schedules/` hold every schedule the paper evaluated, with the paper's numbers. Prepare `data/cache` (val, and v2 for ImageNet-V2), then:

```bash
leaf schedules assets/schedules/deit_b.json | grep main/diffrate            # pick a key
CUDA_VISIBLE_DEVICES=0 leaf evaluate  --schedule assets/schedules/deit_b.json --key main/diffrate/DR8.7 --split val
# {"top1": 79.11, ..., "paper_top1": 79.11}
CUDA_VISIBLE_DEVICES=0 leaf benchmark --schedule assets/schedules/deit_b.json --key leaf/latency/DR8.7
# {"ms_mean": {"32": ...}, "paper_ms_bs32": 17.974}
```

- **Accuracy** reproduces exactly (BF16, batch 32, deterministic kernels).
- **Latency** is the paper's on an RTX 5060 Ti with the machine otherwise idle; on the same card, expect it within a few percent.
- `paper` in each entry names the tables and figures that use it. For the method's schedules, `schedules/leaf/<bb>.json` works the same way.

## Reproduce the paper

The repository holds everything needed to re-run the experiments as set up in the paper: the released schedules with their planning inputs (`schedules/leaf/`), and the sealed configuration of every evaluated arm, baselines included (`schedules/main_comparison_plans/`, `schedules/gmac_planned/`, `schedules/latency_pricing/`, ... for the original campaigns and `schedules/journal/` for the experiments added for the journal version). Re-running a campaign evaluates exactly these arms; accuracy reproduces exactly, latency within a few percent on the same card.

The raw results of the campaigns (about 9 GB: per-image predictions, timings and analysis outputs) are not part of the repository; they are available from the authors on request. The per-image correctness of every evaluated configuration, the raw timings and the analysis reports, with a script that re-checks every paired test of the paper in about a second, are deposited at Zenodo (DOI in the paper).

See [`results/README.md`](results/README.md) for the layout of `results/`. With `results/` in place (re-run or obtained from the authors):

```bash
leaf report assets         # schedule books (assets/schedules) and PNG figures (assets/figures)
leaf report analyze        # recompute every analysis (pairing, McNemar/Holm, equal GMAC, ReaL, ablations, pricing) from the raw results
leaf report tables         # write all 16 generated tables to paper/tables/
leaf report figures        # write the 6 figures to paper/figures/ (the token figure reads data/cache val)
cd paper && tectonic supp.tex && tectonic main.tex
```

`analyze` reproduces every stored analysis output, and `tables` and `figures` regenerate the tables and figures that the paper derives from the campaigns. Three tables of the submitted manuscript are written by separate scripts from later sessions: the CLIP tables (`tools/clip_paper_tables.py`), the black-box search table (`tools/search_paper_table.py`) and the T4 rows of the latency-pricing table (`second_device.py`). The LaTeX sources of the manuscript are not part of this repository.

To re-run an experiment on GPU, use `leaf experiments list` (experiments, their paper sections, stages and result folders) and `leaf experiments run <experiment> <stage>`, for example:

```bash
leaf experiments run main_comparison plan --backbones deit_b                 # CPU: re-plan (sealed plans are not overwritten)
CUDA_VISIBLE_DEVICES=0 leaf experiments run main_comparison worker           # GPU: confirm / val / V2 of every planned arm
CUDA_VISIBLE_DEVICES=0 leaf experiments run main_comparison time             # GPU, nothing else on the machine: bs32 timing
leaf experiments run ablation_and_alpha_grid plan --dry-run                  # the paper's LEAF schedules and every ablation arm
```

The paper's protocol has four parts:
- **Sealing.** All configurations, baseline grid points included, are sealed by hash before any validation or V2 evaluation.
- **Accuracy.** Accuracy is evaluated in BF16 at batch 32 with deterministic kernels.
- **Timing.** All configurations of a backbone are timed in one session, interleaved over three rounds with the middle round reversed, with no other job on the machine.
- **Statistics.** Each schedule is paired with the latency-nearest point of every family, and pairs are compared with McNemar's test and Holm correction.

### Further experiments of the paper

The experiments added for the journal version run as scripts with stages, `python leaf/reproduction/experiments/<name>.py --help`:
- `search_baselines.py`: random search and regularised evolution against the dynamic program (Appendix C of the paper; `results/search_baselines/`).
- `second_device.py`: device transfer to a Tesla T4 (`scripts/run_second_device.sh`, `scripts/kaggle/`).
- `seg_comparison` (also `leaf experiments run seg_comparison`): the segmentation pilot on Segmenter ViT-B/16 and ADE20K (`results/seg_comparison/seg_b_ade/`, tracked in git).
- `station_discount.py`, `separability_check.py`, `final_station_alpha7.py`, `mctf_cost.py`, `efficiency_and_cost.py`: the supporting analyses of Appendix A, Section 6.4 and Section 7.
- `vlm_llava.py`: an exploratory run on LLaVA-Phi-3-mini (not reported in the paper).

Added for the revision (October 2026; launchers in `scripts/run_*.sh`; `scripts/gen_*.py` write the supplementary tables from the `report.json` files):
- `dp_verify.py`: the planner's candidates (seven dynamic-program solutions and local mutants) verified on the measured calibration KL (Supplementary S5.7; `results/dp_verify/`).
- `salt_low_coverage.py`: SALT re-fitted with a few workloads below its timing range (S5.6; `results/salt_low_coverage/`).
- `avgpool_final_station.py`: the final-block station forbidden at planning time on MAE-B/L (S5.8; `results/avgpool_final_station/`).
- `calib_size.py`: schedules re-planned from 1,000 / 2,500 / 5,000 / 10,000 probe images (S5.5; `results/calib_size/`).
- `probe_domain.py`: DeiT-B re-planned from probes on 1,000 unlabelled MS-COCO images (S5.5; `results/probe_domain/`).
- `vitb384_comparison.py` with `backbones_vitb384.json` (`LEAF_BACKBONES`): the full comparison on ViT-B/16 at 384 px, 577 tokens, with cost-only densified baseline grids (Section 6.1 and S5.12; `results/vitb384_comparison/augreg_b384/`). `diffrate_vitb384_search.py` is DiffRate's own search at 384 px; it was not run to completion (about 36 GPU-hours per budget on the RTX 5060 Ti) and the paper reports no DiffRate point at 384 px.
- `batch_sensitivity.py` and `a100_timing.py` (`scripts/colab/LEAF_A100.ipynb`, `scripts/a100_home.sh`): re-timing of the main comparison at other batch sizes and on an A100; not reported in the paper.

### CLIP: zero-shot ImageNet and image–text retrieval

`clip_b16` is OpenAI CLIP ViT-B/16 (timm `vit_base_patch16_clip_quickgelu_224.openai`) and `clip_l14` is OpenAI CLIP ViT-L/14 at 224 px (timm `vit_large_patch14_clip_quickgelu_224.openai`: patch 14, 257 tokens, 24 blocks).
- **Head.** Its head is the zero-shot classifier: the projection, then a scaled cosine to the ImageNet class embeddings of its own text tower (80 templates). Every method, probe and runner therefore works on it unchanged.
- **Planning.** LEAF is planned on the KL of the zero-shot distribution over the 5,000 probe images. The same sealed schedules are then evaluated on Flickr30k (1k test) and MS-COCO (5k test) retrieval without re-planning.
- **DiffRate.** DiffRate releases no CLIP schedules. The paper runs DiffRate's own search on CLIP with the official code (`python leaf/reproduction/experiments/diffrate_clip_search.py search --budgets 8.7`, about 6.4 GPU-hours per budget; results in `results/diffrate_clip_search/`); the transferred DeiT-B rates serve only to place the budgets. `tools/clip_paper_tables.py` writes the CLIP tables of the paper from the final timing session.

```bash
pip install -e '.[clip]'                                   # open_clip, for the text tower
leaf data prepare --split flickr_test --download           # Karpathy test splits (Hugging Face, about 1 GB)
leaf data prepare --split coco_test --download
leaf data clip-text --backbone clip_b16                    # once: ImageNet classifier + caption embeddings (CPU by default;
                                                           # add --device cuda on an idle card: minutes instead of ~25 min for L/14)
bash scripts/run_clip_comparison.sh                        # search -> plan (sealed) -> worker -> time -> report
bash scripts/run_clip_densify.sh                           # cost-only baseline points near every LEAF latency, second session

leaf data clip-text --backbone clip_l14 --device cuda      # ViT-L/14: the same two scripts with BACKBONES=clip_l14
BACKBONES=clip_l14 bash scripts/run_clip_comparison.sh
BACKBONES=clip_l14 bash scripts/run_clip_densify.sh
```

Results go to `results/clip_comparison/<backbone>/`: `report.json`, `table.tex` and `frontier.pdf` (`*_bs8` at batch 8).

## Repository layout

```text
leaf/                       the method (installable package)
  backbones.py                frozen backbones (timm; official MAE weights)
  model/                      the reduced ViT: stations, prune -> merge (M0, M3), the alpha score, Triton kernels
  forecast/                   forecast D: folded query-key forms, head groups, rank-r factors, Triton kernels
  planner/                    probes, separable surrogate, Lagrangian DP, budgets, GMAC planner, alpha planner
  latency/                    SALT, timing as one CUDA graph or on the CPU, SALT workloads
  evaluation/                 cached splits, evaluation runner, determinism, GPU coordination
  data/                       ImageNet download and cache builders
  api.py, cli.py              Python API and command line (`leaf --help`)
  reproduction/               the paper
    experiments/              one module per experiment (`leaf experiments list | run`)
    runners/                  campaign runners, exclusive timing, sealed-stage protocol
    ablation_models/          models and planners used only by ablations
    report/                   tables, figures, schedule books and release files (`leaf report ...`)
baselines/                  re-run baselines; official code vendored in third_party/ with its licence
assets/                     PNG figures and the schedule books (every schedule the paper evaluated, with its numbers)
schedules/                  released schedules (alpha form): leaf/ (the paper's method + planning inputs), every other campaign's
                            schedules, and kappa_form/ (the campaigns' own exports, read by the experiment code)
paper/                      created by `leaf report tables|figures` (generated tables/ and figures/); the manuscript sources are not distributed here
results/                    raw results of the campaigns (not in git; README.md, SUMMARY.json)
data/real/                  ImageNet-ReaL labels; data/cache is created by `leaf data prepare`
scripts/                    shell chains that ran the campaigns (GPU scheduling, timing windows)
tests/                      script tests (see tests/README.md)
tools/                      migration from the research repository (rename_map.py, migrate_results.py), alpha_form.py
```

## Names used in files and results

- **α and κ.**
  - Everything released is in the α form. `pda` is the method (α score) on top of `pdm` (M3 merge) or `pd` (M0 merge).
  - Campaigns run before the α score wrote `pd` / `pdm` configs with the unnormalised score `z(log P) + κ z(D)`, where κ = α/(1 − α) gives the same token order. Those configs name the raw result files (`config_sha`), so `results/` and `schedules/kappa_form/` keep them unchanged.
  - Every set has an α form: `schedules/<set>/`, the probe tables in `schedules/leaf/planning/`, and `results/probe_runs_alpha/`.
  - Re-scoring the paper's κ schedules in α on ImageNet val gives identical predictions for 194 of 196 schedules and one changed image for the other two (`leaf experiments run alpha_kappa_equivalence`).
- **Groups and arms of the main campaign:**
  - `leaf_alpha4/<budget>`: LEAF on the four-value α grid {0, ½, ⅔, ¾}, the paper's main campaign;
  - `ablation/<arm>/<budget>`;
  - `seed251|seed351/<budget>`: other calibration samples;
  - `ref/...`: baselines and references.
- **Ablation campaign groups** (`results/ablation_and_alpha_grid/<bb>/plans/<group>.json`):
  - `leaf` is the paper's method (all budgets on the seven-value α grid);
  - the other groups are `grid_theta`, `grid_theta_alpha`, `schedule_structure`, `shared_settings`, `forecast_only`, `alpha_grid`, `forecast_compression`, `probe_sample_size`, `theta_grid`, `final_station(_all)`, `fixed_station_count` and `surrogate_check`.
- **Budget ids.**
  - `DRx`: anchored at the released DiffRate schedule of x GMAC. The latency budget is 0.97 × its measured latency; the GMAC budget is its actual MACs.
  - `extra1`, `extra2`: faster than the fastest anchor (`extra2` is the faster).
  - `dense0.xx`: AugReg budgets, a fraction of the dense latency.
- **Operators.** M0 is the DiffRate merge, M3 the MLERP (norm-preserving) merge.

## Tests

```bash
python tests/test_leaf_cli.py          # CPU: planner and SALT reproduce the released schedules exactly
python tests/test_reduced_vit.py            # GPU: keeping all tokens reproduces the dense model
```

The full list is in [`tests/README.md`](tests/README.md).

## Limitations

- **One device class.** All latencies come from one consumer GPU and one desktop CPU. Another device needs its own SALT fit.
- **CLIP.** Only ViT-B/16 is registered so far. ViT-L/14 (257 tokens, patch 14) still needs the patch size and the head GMAC generalised in `leaf/planner/gmac.py` and `leaf/latency/salt.py`.
- **Batch 1.** At batch 1, small models are launch-bound on this GPU, and no reduction method is faster than the dense model.
- **Approximations.** The planner relies on two approximations, the separable distortion surrogate and SALT. Neither is certified for the measured accuracy or latency.
- **GPU-only timing command.** `leaf time-workloads` times on CUDA only. The paper's CPU measurements are in `leaf experiments run latency_pricing cpu_time`. On the CPU, the α score uses the full-rank reference forecast.
- **Earlier variant.** LEAF-M0, the earlier variant compared in the paper, is released as schedules (`schedules/leaf_m0/`, `leaf_m0/` in the schedule books) that `leaf evaluate` and `leaf benchmark` run; its campaign driver is not part of this repository (see PROVENANCE.md).

## Provenance and acknowledgements

This repository was created on 2026-10-21 from the research repository `efficient-vlm` (commit `50eaa23`, tag `pre-reorg-2026-10-20`). [`PROVENANCE.md`](PROVENANCE.md) maps every old file, result folder and key to its new name.

This code builds on timm and on the official implementations of ToMe, PiToMe, EViT, PPT, MCTF and DiffRate (`baselines/third_party/`, each with its licence and source commit). The merge rule M3 is MLERP from ToFu.

## Citation

If you use this code or the released schedules, please cite the manuscript (under review at *Knowledge-Based Systems*):

```bibtex
@unpublished{huynh2026leaf,
  title  = {{LEAF}: Latency-Budgeted Efficient Planning with Attention Forecasts for Training-Free Token Reduction},
  author = {Huynh, Canh Vien and Phan, Minh Nhat and Pham, Nhat Khoa and Nguyen, Van Hieu},
  note   = {Manuscript submitted to Knowledge-Based Systems; code at \url{https://github.com/huynhcanhvien/LEAF}},
  year   = {2026}
}
```

## License

LEAF's own code is licensed under the [Apache License 2.0](LICENSE). The vendored baseline code in `baselines/third_party/` keeps its own licences; the ToMe and PiToMe code is CC BY-NC 4.0 (non-commercial). See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
