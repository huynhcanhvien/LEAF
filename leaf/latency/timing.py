"""Latency of a model: BF16 autocast with Linear/Conv weights pre-cast, CUDA-graph replay, one GPU alone.

A graph captured from the fp32 model would replay autocast's weight casts on every call, so graphs are
captured on a pre-cast copy (inference_copy). Measure all arms of a comparison in one session,
interleaved over rounds: latency drifts 0.3-0.5 ms between sessions.
"""
import copy
import gc
import os

import numpy as np
import torch

# Inference dtype of timed models: BF16 (the paper). LEAF_DTYPE=float16 for GPUs without BF16 (e.g. T4, sm_75); accuracy
# evaluation always uses BF16.
INFERENCE_DTYPE = getattr(torch, os.environ.get('LEAF_DTYPE', 'bfloat16'))


def inference_copy(model, dtype=INFERENCE_DTYPE):
    """Copy whose Linear/Conv weights are cast to dtype (what autocast computes, without per-call casts)."""
    m = copy.deepcopy(model).eval()
    for mod in m.modules():
        if isinstance(mod, (torch.nn.Linear, torch.nn.Conv2d)): mod.to(dtype)
        if hasattr(mod, 'prepare_inference'): mod.prepare_inference(dtype)
    return m


def input_size(model, default=224):
    """Input side of a model: its timm patch embedding's img_size (segmentation windows: 512), else default."""
    for mod in model.modules():
        size = getattr(getattr(mod, 'patch_embed', None), 'img_size', None)
        if size is not None: return int(size[0] if isinstance(size, (tuple, list)) else size)
    return default


def measure_latency(model, batch_size, **kw):
    """Mean / p50 / p90 ms per batch forward and images per second, with each method's native kernels
    (leaf/evaluation/determinism.py)."""
    from leaf.evaluation.determinism import native
    with native(): return _latency_cast(model, batch_size, **kw)


@torch.no_grad()
def _latency_cast(model, batch_size, graph=True, iters=200, warmup=20, img_size=None, device='cuda', dtype=INFERENCE_DTYPE):
    m = inference_copy(model, dtype)
    img_size = img_size or input_size(m)
    x = torch.randn(batch_size, 3, img_size, img_size, device=device)
    torch.cuda.reset_peak_memory_stats()
    side = torch.cuda.Stream(); side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side), torch.autocast('cuda', dtype=dtype):
        for _ in range(warmup): m(x)
    torch.cuda.current_stream().wait_stream(side)
    if graph:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g), torch.autocast('cuda', dtype=dtype): m(x)
        run = g.replay
    else:
        def run():
            with torch.autocast('cuda', dtype=dtype): m(x)
    for _ in range(warmup): run()
    torch.cuda.synchronize(); times = []
    for _ in range(iters):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record(); run(); end.record(); end.synchronize(); times.append(start.elapsed_time(end))
    t = torch.tensor(times); mean = t.mean().item()
    return dict(bs=batch_size, mode='graph' if graph else 'eager', peak_vram_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 3),
                ms_mean=round(mean, 3), ms_p50=round(t.quantile(.5).item(), 3), ms_p90=round(t.quantile(.9).item(), 3),
                img_per_s=round(batch_size / mean * 1e3, 1))


def measure_latency_batches(model, batches=(1, 32, 128), repeats=3, graph=True, big=(20, 5)):
    """{bs: {ms_mean, ms_std, samples, ...}} or 'OOM'. big = (iters, warmup) for bs > 32."""
    out = {}
    for bs in batches:
        try:
            iters, warm = (50, 10) if bs <= 32 else big
            samples = [measure_latency(model, bs, graph=graph, iters=iters, warmup=warm) for _ in range(repeats)]
            ms = [x['ms_mean'] for x in samples]
            out[str(bs)] = dict(ms_mean=float(np.mean(ms)), ms_std=float(np.std(ms)), img_per_s=float(bs * 1000 / np.mean(ms)),
                                peak_vram_gb=max(x['peak_vram_gb'] for x in samples), samples=ms, mode=samples[0]['mode'])
        except torch.cuda.OutOfMemoryError:
            out[str(bs)] = 'OOM'
        gc.collect(); torch.cuda.empty_cache()
    return out


@torch.no_grad()
def gmacs(model, img_size=None, device='cuda'):
    """Multiply-accumulates per image (billions), every matmul including SDPA (FlopCounterMode: 2 FLOPs/MAC)."""
    from torch.utils.flop_counter import FlopCounterMode
    img_size = img_size or input_size(model)
    x = torch.randn(1, 3, img_size, img_size, device=device); model.eval()
    with FlopCounterMode(display=False) as fc: model(x)
    return fc.get_total_flops() / 2 / 1e9
