# Sealed configurations of the journal-version experiments

Exported from the sealed campaign folders. Each folder holds the files that define the evaluated arms (every arm's
runtime config with its SHA-256, sealed before any evaluation), the timing design, the budgets and the SALT models with
their workloads. Any arm runs with `leaf evaluate | benchmark --schedule <folder>/<backbone>/plans.json --key <arm>`
(`leaf schedules <file>` lists the arms); its per-image results are in `release/`. The LEAF schedules themselves are
also in `schedules/leaf/` in readable form.

| Folder | Paper | Files |
|---|---|---:|
| `clip_comparison/` | Section 6.5: CLIP ViT-B/16, zero-shot ImageNet and retrieval | 14 |
| `diffrate_clip_search/` | Section 6.5, Supplementary S5.5: DiffRate's own search on CLIP (searched rates per GMAC budget) | 6 |
| `surrogate_check/` | Appendix A, Supplementary S2: 30 random schedules per budget at three budgets, with F, val KL and top-1 | 2 |
| `search_baselines/` | Appendix B, Supplementary S5.4: random search and regularised evolution | 2 |
| `second_device/` | Section 6.3: transfer to a Tesla T4 (SALT refitted on the device) | 4 |
| `dp_verify/` | Section 7, Supplementary S5.8: planner candidates verified on the measured KL | 7 |
| `avgpool_final_station/` | Section 7, Supplementary S5.9: no final-block station on average-pooled heads | 2 |
| `calib_size/` | Section 7, Supplementary S5.7: 1,000 to 10,000 probe images | 3 |
| `probe_domain/` | Section 7, Supplementary S5.7: probes on 1,000 MS-COCO images | 2 |
| `vitb384_comparison/` | Section 6.1, Supplementary S5.11: ViT-B/16 at 384 pixels, all families | 6 |
| `station_discount/` | Section 7, Supplementary S2: per-station discount of the surrogate | 7 |
| `final_station_alpha7/` | Section 7: schedules without their final-block station | 7 |
| `salt_low_coverage/` | Section 6.3, Supplementary S5.3: SALT below the range of its timing data | 21 |
