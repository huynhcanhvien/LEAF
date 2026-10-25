# A100 sessions for the KBS revision (Colab or any single-A100 machine)

Notebook: `scripts/colab/LEAF_A100.ipynb`. Kit: `a100_kit.tar.gz`, made at home by
`python leaf/reproduction/experiments/a100_timing.py export` (tasks, the pricing design, the device script and the T4 kit).

| Phase | On the device | Time | Output | At home |
|---|---|---|---|---|
| `transfer` | `scripts/run_second_device.sh device-transfer` in BF16: the sealed latency-pricing arms at batch 4 and 32 | ≈ 1 h | `second_device_transfer.tar.gz` | `bash scripts/a100_home.sh transfer <archive>` → `results/second_device/compare_transfer.json` |
| `main` | `a100_timing.py smoke` then `time` + `pack`: every main-comparison arm of DeiT-S/B, ViT-B (MAE) in one session (batch 32; 4/128 for LEAF, DiffRate, dense) | ≈ 1.5–2 h | `a100_timing_<device>.tar.gz` | `bash scripts/a100_home.sh main <archive>` → `results/a100_timing/report_<device>.json` (pairs at device latency, accuracy reused) |
| `diffrate384` (optional) | DiffRate's official search on ViT-B/16-384, 3 epochs per budget, checkpointed | ≈ 8–11 h per budget | `diffrate_vitb384_search/` on Drive | `bash scripts/a100_home.sh diffrate <dir>` then worker / retime / report of `vitb384_comparison.py` |

Rules kept from the paper: one process on the timed card (the scripts abort if another appears), three interleaved rounds with the
middle round reversed, 10 eager warm-ups + CUDA-graph capture + 10 graph warm-ups + 50 timed replays (20 at batch 128), BF16.
Accuracy is never evaluated on the device; it comes from the sealed campaigns at home, so a device session only re-pairs the
same arms at the device's latencies. Cloud cards drift between sessions (the T4 ran 9–14 % faster in its second session), so every
comparison stays within one session, and the dense model is timed in every round as the drift reference.

Colab notes: Runtime → A100 GPU; mount Drive so the archives survive a disconnect; `transfer` + `main` fit in one session;
`diffrate384` needs the 1,024 training shards (`timm/imagenet-1k-wds`, ≈ 150 GB on local disk) and resumes from the
checkpoints copied to Drive after every budget.
