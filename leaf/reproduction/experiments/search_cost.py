"""Search cost (paper Table 'search cost'): forward and forward+backward throughput of the dense DeiT-S / DeiT-B.

python leaf/reproduction/experiments/search_cost.py measure   (GPU, nothing else on the card) -> results/search_cost/throughput.json

BF16 autocast, batch 64, synthetic inputs, mean over timed iterations after warm-up. "forward+backward" keeps the weights
frozen and back-propagates to the activations, as DiffRate's rate training does through the frozen network. The paper's
table (leaf.reproduction.report.tables.cost) reads the stored file; a new measurement differs from it by timing noise.
"""
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / 'results/search_cost/throughput.json'
MODELS = ('deit_small_patch16_224.fb_in1k', 'deit_base_patch16_224.fb_in1k')
BACKWARD = 'forward+backward (frozen weights, grad to activations)'


def throughput(model, backward, batch=64, warmup=10, iters=50):
    import torch
    x = torch.randn(batch, 3, 224, 224, device='cuda', requires_grad=backward)
    def step():
        with torch.autocast('cuda', dtype=torch.bfloat16):
            y = model(x)
        if backward: y.float().sum().backward()
    with torch.set_grad_enabled(backward):
        for _ in range(warmup): step()
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters): step()
        end.record(); torch.cuda.synchronize()
    return batch * iters / (start.elapsed_time(end) / 1000)


def measure():
    import timm
    from leaf.evaluation.runner import require_idle_gpu
    require_idle_gpu(idle=True)
    out = {}
    for name in MODELS:
        model = timm.create_model(name, pretrained=True).cuda().eval().requires_grad_(False)
        out[name] = {'forward': throughput(model, False), BACKWARD: throughput(model, True)}
        print(name, out[name], flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(out, indent=1))
    return out


if __name__ == '__main__':
    {'measure': measure}[(sys.argv[1:] or ['measure'])[0]]()
