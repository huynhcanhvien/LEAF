"""Evaluation: cached splits, cached runs, latency, locked protocol. Importing it enables deterministic mode
(leaf/evaluation/determinism.py) before any CUDA work."""
from leaf.evaluation import determinism  # noqa: F401
