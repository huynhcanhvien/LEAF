"""CPU tests of the new control, GMAC-only DP and unbiased cost matching."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from types import SimpleNamespace
import numpy as np
import torch
from timm.models.vision_transformer import VisionTransformer
from baselines.tome_m3 import pack_matching,merge_reference,ToMeM3
from baselines.third_party.tome.merge import bipartite_soft_matching,merge_wavg
from baselines.methods import ToMe
from leaf.planner.surrogate import ProbeCurve
from leaf.planner.gmac import GmacPlanner,config_cost,make_budgets

def main():
    torch.set_num_threads(2); torch.manual_seed(151)
    for n in (7,16,17):
        for r in (0,1,3,99):
            x=torch.randn(2,n,8); size=torch.rand(2,n,1)+1; metric=torch.randn(2,n,5)
            packed,mass,dest,ids,keep=pack_matching(metric,x,size,r)
            if dest is None:
                assert torch.equal(x,packed) and keep==n; continue
            merge,_=bipartite_soft_matching(metric,r,True,False)
            out,os=merge_reference(packed,mass,dest,keep)
            mean=merge(x,'sum')/merge(torch.ones(2,n,1),'sum')
            peak=merge(x.norm(dim=-1,keepdim=True),'amax')
            ref=mean*peak/mean.norm(dim=-1,keepdim=True)
            assert torch.allclose(out,ref,atol=2e-6)
            assert torch.allclose(os,merge(size,'sum'),atol=1e-6)
            assert torch.equal(out[:,0],x[:,0]) and torch.equal(os[:,0],size[:,0])
    x=torch.tensor([[[1.,0.],[-1.,0.]]]); size=torch.ones(1,2,1)
    out,os=merge_reference(x,size,torch.zeros(1,1,dtype=torch.long),1)
    assert torch.equal(out,x[:,:1]) and float(os[0,0,0])==2
    spec=SimpleNamespace(pool='token')
    vit=VisionTransformer(img_size=32,patch_size=8,embed_dim=32,depth=3,num_heads=2,num_classes=7).eval().requires_grad_(False)
    img=torch.randn(2,3,32,32)
    with torch.no_grad():
        assert torch.equal(ToMeM3(vit,spec,0)(img),ToMe(vit,spec,0)(img))
    # Synthetic DP can run without any latency model or GPU context.
    spec=SimpleNamespace(n_tokens=17,depth=3,dim=16)
    dc=SimpleNamespace(rank=4,triton=False,at=lambda l:[0,0])
    cv={(th,k):ProbeCurve({l:{4:.01,8:.04,14:.2} for l in (1,2)},17) for th in (0.,.5,1.) for k in (0.,1.)}
    planner=GmacPlanner(spec,dc,cv,'M3')
    dense_cfg=dict(prune_keep=[17]*3,merge_keep=[17]*3,k={},groups={},rank=None)
    budget=config_cost(dense_cfg,spec)*.8
    row=planner.plan(budget)
    assert not row.get('infeasible') and row['gmac']<=budget+1e-9
    assert row['config']['kind']=='pdm' and 'latency' not in row['objective'].lower().replace('no latency inputs','')
    print('PASS: official ToMe routing/size parity, M3 norm/cancellation/CLS, zero-merge equivalence, GMAC-only DP budget')

if __name__=='__main__': main()
