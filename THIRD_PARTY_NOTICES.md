# Third-party code

LEAF's own code is licensed under the Apache License 2.0 (`LICENSE`). The following files are vendored from other projects
and remain under their own licences. Each folder in `baselines/third_party/` holds the original `LICENSE` and a `SOURCE`
file with the upstream commit and the changes made.

| Folder | Upstream | Licence |
|---|---|---|
| `baselines/third_party/tome/` | [facebookresearch/ToMe](https://github.com/facebookresearch/ToMe) | CC BY-NC 4.0 (non-commercial) |
| `baselines/third_party/pitome/` | [hchautran/PiToMe](https://github.com/hchautran/PiToMe) | CC BY-NC 4.0 (non-commercial) |
| `baselines/third_party/evit/` | [youweiliang/evit](https://github.com/youweiliang/evit) | Apache-2.0 |
| `baselines/third_party/ppt/` | [xjwu1024/PPT](https://github.com/xjwu1024/PPT) | Apache-2.0 |
| `baselines/third_party/mctf/` | [mlvlab/MCTF](https://github.com/mlvlab/MCTF) | MIT |

The ToMe and PiToMe code is used only to re-run those baselines; the LEAF method (`leaf/`) does not depend on it.
Any use of the vendored ToMe or PiToMe code must comply with its non-commercial licence.

`leaf/model/segmenter_head.py` re-implements, without copying code, the mask-transformer decoder of Segmenter
([rstrudel/segmenter](https://github.com/rstrudel/segmenter), MIT) in the weight layout of MMSegmentation's
`SegmenterMaskTransformerHead` ([open-mmlab/mmsegmentation](https://github.com/open-mmlab/mmsegmentation), Apache-2.0),
and loads MMSegmentation's released ADE20K checkpoint. `leaf/data/ade20k.py` and `leaf/evaluation/segmentation.py` follow
MMSegmentation's test pipeline (keep-ratio resize, slide inference, `intersect_and_union`). ADE20K (SceneParsing) is
distributed by MIT CSAIL under its own terms.
