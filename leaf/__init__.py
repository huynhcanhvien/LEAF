"""LEAF: Latency-budgeted Exact planning with Attention Forecasts, training-free token reduction for frozen ViTs.

Subpackages:
  leaf.backbones     frozen timm / MAE backbones (Spec, SPECS, load)
  leaf.model         the reduced ViT: stations, prune -> merge (M0 / M3), the alpha score, Triton kernels
  leaf.forecast      the forecast D: folded query-key forms, head groups, rank-r factors, Triton kernels
  leaf.planner       probes, separable surrogate, Lagrangian DP, budgets, schedule configs
  leaf.latency       SALT (station-aware latency model), timing, SALT workloads
  leaf.evaluation    cached data splits, evaluation runner, determinism, GPU coordination
  leaf.data          ImageNet download and uint8 evaluation caches
  leaf.api           high-level helpers (load a schedule, build a model, evaluate)
  leaf.reproduction  the paper: experiments (leaf experiments) and its tables, figures and assets (leaf report)
Command line: leaf --help
"""
__version__ = "1.0.0"
