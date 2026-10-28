"""The reduced ViT. A station sits between the attention and the MLP of a block and removes tokens: prune the
lowest-scoring patches, then merge a share theta of the removed tokens into their most similar kept tokens.

  reduced_vit        ReducedViT / ReductionConfig: stations scored by (1 - alpha) z(log P) + alpha z(D), M0 (DiffRate)
                     merge; config base "pd"
  mlerp_vit          MlerpViT: the M3 (MLERP, norm-preserving) merge with Triton kernels; base "pdm", kernel "triton_norm"
  merge_variant_vit  MergeVariantViT: the merge variants of the ablations, generic kernels; base "pdm", other kernels
  ablation_vit       ExtraViT: the forecast-horizon, placebo and norm ablations; base "extra_pdm"
  token_ops          z-score, class-token attention P, the M0 compress operator
  merge_variants     merge-variant implementations (M0 delegates to DiffRate's operator)
  kernels/           Triton merge kernels: mean_merge (M0), mlerp_merge (M3 / N4)

A config of the method has kind "pda" and names its wrapper in "base"; both are part of the hashed configs that name
every result file, so they keep their short names: p = prune, d = forecast D, m = merge operator, a = alpha.
"""
