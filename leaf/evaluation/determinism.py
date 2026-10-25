"""Bit-exact evaluation: every accuracy/KL number is reproducible across runs, processes and our two GPUs.

Sources of run-to-run variation that were found and removed:
  - CUDA SVD and index_add_ (atomics) when building D            -> leaf/forecast/attention_forecast.py (CPU fp64 SVD, per-group sums)
  - LAPACK SVD depends on the CPU thread count                    -> pinned (grouped_future.SVD_THREADS)
  - scatter_reduce sum/mean with atomics (ToMe, PiToMe, MCTF, PPT
    official merge code; the scatter fallback of compress)        -> torch.use_deterministic_algorithms(True)
  - cuBLAS workspace-dependent reductions                          -> CUBLAS_WORKSPACE_CONFIG=:4096:8
The method's own path (SDPA, GEMMs, Triton D and merge kernels, sort/gather) is deterministic in both modes and
gives identical bits with or without this switch (tests/test_determinism.py), so earlier cached runs stay valid.

Latency is measured with each method's native kernels (`native()`): deterministic replacements of scatter_reduce
are slower and are not what the baselines' code runs; their outputs differ only by floating-point summation order.
"""
import contextlib
import os

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import torch


def enable():
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False


@contextlib.contextmanager
def native():
    """Default (non-deterministic) kernels, for latency measurements only."""
    prev = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(False)
    try: yield
    finally: torch.use_deterministic_algorithms(prev)


enable()
