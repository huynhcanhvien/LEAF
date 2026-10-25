"""Additive norm-target kernel. Existing M0/M3 kernels remain untouched."""
import torch
import triton
import triton.language as tl
from leaf.model.kernels.mean_merge import merge_inference
from leaf.model.kernels.mlerp_merge import token_norms,stable_sizes


@triton.jit
def norm_target(X, DEST, NORM, MEAN, OUT, NP:tl.constexpr,NM:tl.constexpr,
                C:tl.constexpr,CP:tl.constexpr,SRC:tl.constexpr,SP:tl.constexpr,RMS:tl.constexpr):
    row=tl.program_id(0);b=row//NM;d=row%NM
    src=tl.arange(0,SP);target=tl.load(DEST+b*SRC+src,src<SRC,-1)
    assigned=(target==d)&(src<SRC);count=tl.sum(assigned.to(tl.int32),0)
    bn=tl.load(NORM+b*NP+d);sn=tl.load(NORM+b*NP+NM+src,src<SRC,0.)
    peak=tl.maximum(bn,tl.max(tl.where(assigned,sn,0.),0))
    if RMS:desired=tl.sqrt((bn*bn+tl.sum(tl.where(assigned,sn*sn,0.),0))/(count+1))
    else:desired=(bn+tl.sum(tl.where(assigned,sn,0.),0))/(count+1)
    col=tl.arange(0,CP);y=tl.load(MEAN+row*C+col,col<C,0.)
    ynorm=tl.sqrt(tl.sum(y*y,0));direction=y
    if ynorm==0.:
        first=tl.min(tl.where(assigned&(sn==peak),NM+src,NP),0)
        first=tl.where(bn==peak,d,first)
        direction=tl.load(X+b*NP*C+first*C+col,col<C,0.)
        ynorm=tl.sqrt(tl.sum(direction*direction,0))
    out=direction*(desired/tl.maximum(ynorm,1.e-30))
    out=tl.where(count>0,out,y)
    tl.store(OUT+row*C+col,out,col<C)


def merge_target(x,size,destination,maximum,m,mode):
    b,p,c=x.shape;src=p-m
    if not src:return x,size
    norm=torch.empty((b,p),device=x.device)
    token_norms[(b*p,)](x,norm,p,c,triton.next_power_of_2(c),False,num_warps=4,enable_fp_fusion=False)
    mean,sizes=merge_inference(x,size,destination,maximum,m)
    stable_sizes[(b,triton.cdiv(m,128))](size,destination,maximum,sizes,p,m,src,False,128,num_warps=4,enable_fp_fusion=False)
    out=torch.empty_like(mean)
    norm_target[(b*m,)](x,destination,norm,mean,out,p,m,c,triton.next_power_of_2(c),src,
        triton.next_power_of_2(src),mode=='rms_norm',num_warps=4,enable_fp_fusion=False)
    return out,sizes
