# Second accelerator on Kaggle (T4)

Repeats the latency-pricing test of the paper on an NVIDIA T4 with the same probes and a SALT model refitted on the
T4 (`leaf/reproduction/experiments/second_device.py`). The T4 has no BF16, so latency is timed in FP16
(`LEAF_DTYPE=float16`); accuracy is evaluated at home in BF16. Two Kaggle sessions, with one home step between them.

| Step | Where | Command | Output |
|---|---|---|---|
| 1 | Kaggle | notebook `second_device_t4.ipynb`, `PHASE='device-1'` | `second_device_1.tar.gz` |
| 2 | home | `bash scripts/run_second_device.sh home-1 second_device_1.tar.gz` | `second_device_2.tar.gz` |
| 3 | Kaggle | same notebook, `PHASE='device-2'`, dataset with `second_device_2.tar.gz` attached | `second_device_3.tar.gz` |
| 4 | home | `bash scripts/run_second_device.sh home-2 second_device_3.tar.gz` | `results/second_device/compare.json` |

Notebook settings: Accelerator *GPU T4 x2* (Kaggle has no x1; timing uses card 0 only and card 1 stays idle), Internet on, secret `GITHUB_TOKEN` (read access to the
repository) attached. Run it with *Save Version → Save & Run All*, so that closing the browser does not stop it.
Every timing keeps the paper protocol (10 eager warm-ups, CUDA-graph capture, 10 graph warm-ups, 50 timed replays,
three interleaved rounds with the middle round reversed).
