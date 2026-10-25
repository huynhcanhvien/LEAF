"""(paper: Experimental setup, MCTF note; results/mctf_overhead/) Official MCTF model (mlvlab/MCTF @2004da0) vs our adapter: same weights, BF16, batch 32, CUDA graphs."""
import os, sys, json, torch
root = os.path.expanduser('~/.cache/leaf/official/MCTF')
sys.path.insert(0, root)
from models.deit import deit_base
sys.path.pop(0)
for k in [k for k in sys.modules if k == 'models' or k.startswith('models.') or k == 'utils' or k.startswith('utils.')]: del sys.modules[k]
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[3]))
from leaf.backbones import load
from leaf.latency.timing import measure_latency
from baselines.methods import MCTF
vit, spec = load('deit_b')

class Official(torch.nn.Module):
    def __init__(self, r):
        super().__init__(); self.r = r
        self.m = deit_base(use_mctf=True, mctf_type=[r, 0, 0, 1, 1, 1, 20, 40, 1, 1, 0], task_type=[1, 0, 0, 0],
                           activate_layer=list(range(1, 12)), r_evals=[r])
        print(self.m.load_state_dict(vit.state_dict(), strict=False))
        self.m.eval().cuda()
    def forward(self, x):
        out = self.m(x, r_eval=self.r); return out['x_student'] if isinstance(out, dict) else out

ms = lambda m: round(min(measure_latency(m, 32, iters=50, warmup=10)['ms_mean'] for _ in range(3)), 2)
x = torch.randn(4, 3, 224, 224, device='cuda'); res = {}
for r in (8, 16):
    o, a = Official(r), MCTF(vit, spec, r)
    with torch.no_grad(): err = (o(x).float() - a(x).float()).abs().max().item()
    res[r] = dict(official=ms(o), ours=ms(a), max_abs_diff_fp32=err)
    print(r, res[r], flush=True)
json.dump(res, open(sys.argv[1], 'w'), indent=1)
