"""Additive ToMe+MLERP control: original routing/schedule/size, M3 features.

This is not the complete ToFu method. The existing ToMe implementation is untouched.
"""
import torch
import torch.nn.functional as F
from baselines.methods import ToMe

def pack_matching(metric,x,size,r):
    n=x.shape[1]; r=min(r,(n-1)//2)
    if r<=0: return x,size,None,None,n
    with torch.no_grad():
        metric=metric/metric.norm(dim=-1,keepdim=True)
        a,b=metric[...,::2,:],metric[...,1::2,:]
        scores=a@b.transpose(-1,-2); scores[...,0,:]=-torch.inf
        node_max,node_idx=scores.max(-1)
        edge=node_max.argsort(dim=-1,descending=True)[...,None]
        unm=edge[...,r:,:].sort(dim=1)[0]; src=edge[...,:r,:]
        dst=node_idx[...,None].gather(-2,src)
        unm_ids=2*unm.squeeze(-1); src_ids=2*src.squeeze(-1)
        dst_ids=torch.arange(1,n,2,device=x.device).expand(x.shape[0],-1)
        ids=torch.cat((unm_ids,dst_ids,src_ids),1)
        destination=dst.squeeze(-1)+unm_ids.shape[1]
    gather=lambda value:value.gather(1,ids[...,None].expand(-1,-1,value.shape[-1]))
    return gather(x),gather(size),destination,ids,n-r

def merge_reference(x,size,destination,keep):
    b,p,c=x.shape; src=p-keep
    if not src: return x,size
    assignment=F.one_hot(destination,keep).to(x.dtype).transpose(1,2)
    target,source=x[:,:keep],x[:,keep:]
    count=assignment.sum(-1,keepdim=True)
    mean=(target+assignment@source)/(1+count)
    norms=x.norm(dim=-1); assigned=assignment.bool()
    candidates=norms[:,None,keep:].expand(-1,keep,-1).masked_fill(~assigned,-torch.inf)
    peak,winner=torch.cat((norms[:,:keep,None],candidates),-1).max(-1)
    fallback=torch.where((winner==0)[...,None],target,source.gather(1,(winner-1).clamp_min(0)[...,None].expand(-1,-1,c)))
    direction=torch.where(mean.norm(dim=-1,keepdim=True)>0,mean,fallback)
    corrected=direction*(peak[...,None]/direction.norm(dim=-1,keepdim=True).clamp_min(1e-30))
    result=torch.where(count>0,corrected,target)
    mass=size[:,:keep]+assignment@size[:,keep:]
    return result,mass

class ToMeM3(ToMe):
    def reduce(self,l,x,size,info):
        r=self.r[l]
        if r<=0: return x,size
        packed,mass,destination,_,keep=pack_matching(info['kmean'],x,size,r)
        if destination is None: return packed,mass
        if x.is_cuda and not torch.is_grad_enabled():
            from leaf.model.kernels.mlerp_merge import merge_norm_inference
            maximum=x.new_ones(destination.shape)
            return merge_norm_inference(packed,mass,destination,maximum,keep,'M3')
        return merge_reference(packed,mass,destination,keep)
