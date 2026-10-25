"""(paper: Experimental setup, MCTF note; results/mctf_overhead/) Why MCTF is slow: DeiT-B, batch 32, BF16, CUDA graphs (the paper's timing path), decomposed."""
import sys, json, torch
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[3]))
from leaf.backbones import load
from leaf.latency.timing import measure_latency
from baselines import build
from baselines.methods import MCTF
import baselines.third_party.mctf.mctf as M

bb = sys.argv[1] if len(sys.argv) > 1 else 'deit_b'
vit, spec = load(bb)
ms = lambda m: round(min(measure_latency(m, 32, iters=50, warmup=10)['ms_mean'] for _ in range(3)), 2)
out = {}
out['dense (timm, fused SDPA)'] = ms(vit)
out['MCTF path, no block activated (explicit attention only)'] = ms(MCTF(vit, spec, 8, activate=set()))
for r in (8, 16):
    out[f'ToMe r={r}'] = ms(build(vit, spec, dict(kind='tome', r=r)))
    out[f'MCTF r={r} (official settings)'] = ms(MCTF(vit, spec, r))
    m = MCTF(vit, spec, r); m.base.update(tau_info=0, tau_size=0); out[f'MCTF r={r}, similarity only (no info/size weights)'] = ms(m)
    m = MCTF(vit, spec, r); m.base.update(bidirection=0); out[f'MCTF r={r}, one-way matching'] = ms(m)
out['MCTF r=0 (explicit attention + second qkv, no merge)'] = ms(MCTF(vit, spec, 0))
for k, v in out.items(): print(f'{v:8.2f}  {k}')
json.dump(out, open(f'{sys.argv[2] if len(sys.argv) > 2 else "/dev/stdout"}', 'w'), indent=1)
