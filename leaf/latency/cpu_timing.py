"""CPU latency of a model: pinned cores, fixed thread count, BF16 autocast, trimmed means over rounds.

Additive companion of leaf/latency/timing.py (unchanged). The earlier CPU attempt failed its stability
gate (CV > 17%) without core pinning. Pinning the process to a fixed set of physical cores and fixing
torch/OMP threads brings the between-round CV of DeiT-B bs1 below 1% on the Ryzen 9 9950X.
Every reduction path has a CPU fallback: grouped D (leaf/forecast/attention_forecast.py, non-Triton branch) and
M0/M3 merging (leaf/model/token_ops.compress, leaf/model/mlerp_vit.compress_norm -> compress_variant).
"""
import copy
import os
import statistics
import time

import torch

DEFAULT_CORES = tuple(range(16))      # logical CPUs 0-15 = 16 distinct physical cores on this machine


def pin(cores=DEFAULT_CORES, threads=None):
    """Pin this process to `cores` and fix intra-op threads; call before building models."""
    threads = threads or len(cores)
    os.sched_setaffinity(0, set(cores))
    torch.set_num_threads(threads)
    return dict(cores=list(cores), threads=threads, affinity=sorted(os.sched_getaffinity(0)),
                torch_threads=torch.get_num_threads(), omp=os.environ.get('OMP_NUM_THREADS'))


def cpu_inference_copy(model, dtype=torch.bfloat16):
    """CPU copy whose Linear/Conv weights are pre-cast to dtype (as evaluation.latency.inference_copy)."""
    m = copy.deepcopy(model).cpu().eval()
    for mod in m.modules():
        if isinstance(mod, (torch.nn.Linear, torch.nn.Conv2d)): mod.to(dtype)
        if hasattr(mod, 'prepare_inference'): mod.prepare_inference(dtype)
    return m


@torch.inference_mode()
def forward_cpu(model, x, dtype=torch.bfloat16):
    with torch.autocast('cpu', dtype=dtype):
        return model(x)


@torch.inference_mode()
def measure_latency_cpu_rounds(model, batch_size, rounds=5, iters=20, warmup=10, trim=2, img_size=224, dtype=torch.bfloat16):
    """Per-round trimmed-mean ms of a forward; returns rounds and their CV. `model` must already be on CPU."""
    x = torch.randn(batch_size, 3, img_size, img_size)
    for _ in range(warmup): forward_cpu(model, x, dtype)
    out = []
    for _ in range(rounds):
        ts = []
        for _ in range(iters):
            t = time.perf_counter(); forward_cpu(model, x, dtype); ts.append(time.perf_counter() - t)
        ts.sort()
        out.append(statistics.mean(ts[trim:len(ts) - trim]) * 1e3)
    mean = statistics.mean(out)
    return dict(bs=batch_size, device='cpu', dtype=str(dtype).replace('torch.', ''), rounds=out, ms_mean=mean,
                cv=statistics.pstdev(out) / mean, threads=torch.get_num_threads(),
                affinity=sorted(os.sched_getaffinity(0)), img_per_s=batch_size / mean * 1e3)


def measure_latency_cpu(model, batch_size, **kw):
    """One timing measurement (all rounds pooled), same keys as evaluation.latency.measure_latency where possible."""
    r = measure_latency_cpu_rounds(model, batch_size, **kw)
    return dict(r, ms_p50=statistics.median(r['rounds']))
