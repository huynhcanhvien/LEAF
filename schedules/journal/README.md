# Sealed configurations of the journal-version experiments

Written by `scripts/export_journal_configs.py` from the sealed campaign folders. Each folder holds the files that define the
evaluated arms (every arm's runtime config with its SHA-256, sealed before any evaluation), the timing design, the budgets and
the SALT models with their workloads. Place a folder under `results/<experiment>/` (same relative paths) and run the
`worker`, `time` and `report` stages of `leaf/reproduction/experiments/<experiment>.py` to evaluate exactly these arms;
the LEAF schedules themselves are also in `schedules/leaf/` in readable form.

| Folder | Paper | Files |
|---|---|---:|
| `clip_comparison/` | Section 6.5, Supplementary S5.2: CLIP ViT-B/16, zero-shot ImageNet and retrieval | 14 |
| `diffrate_clip_search/` | Section 6.5: DiffRate's own search on CLIP (searched rates per GMAC budget) | 6 |
| `search_baselines/` | Appendix B: random search and regularised evolution | 2 |
| `second_device/` | Section 6.3: transfer to a Tesla T4 (SALT refitted on the device) | 4 |
| `dp_verify/` | Section 7, Supplementary S5.7: planner candidates verified on the measured KL | 7 |
| `avgpool_final_station/` | Section 7, Supplementary S5.8: no final-block station on average-pooled heads | 2 |
| `calib_size/` | Section 7, Supplementary S5.5: 1,000 to 10,000 probe images | 3 |
| `probe_domain/` | Section 7, Supplementary S5.5: probes on 1,000 MS-COCO images | 2 |
| `vitb384_comparison/` | Section 6.1, Supplementary S5.12: ViT-B/16 at 384 pixels, all families | 6 |
| `station_discount/` | Section 7: per-station discount of the surrogate | 7 |
| `final_station_alpha7/` | Section 7: schedules without their final-block station | 7 |
| `salt_low_coverage/` | Section 6.3, Supplementary S5.6: SALT below the range of its timing data | 21 |
