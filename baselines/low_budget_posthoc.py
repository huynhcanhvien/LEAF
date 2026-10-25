"""Paper reimplementations for the post-hoc campaign, never silently substituted for official code."""
import math

import torch
import torch.nn.functional as F

from .core import TokenViT, attention


def bipartite_indices(metric, r, redundancy=False, cls_attention=None):
    """Match alternating tokens, protect CLS, and rank source removals."""
    r=min(int(r),(metric.shape[1]-1)//2)
    if r<0:
        raise ValueError('negative reduction')
    unit=F.normalize(metric.float(),dim=-1)
    sim=unit[:,::2] @ unit[:,1::2].transpose(-1,-2)
    if redundancy:
        scores=(sim-.7).clamp_min(0).sum(-1)
        dest=sim.argmax(-1)
    else:
        scores,dest=sim.max(-1)
        if cls_attention is not None:
            a=cls_attention.float()
            cutoff=a[:,1:].topk(max(1,math.ceil(.2*(a.shape[1]-1))),dim=-1).values[:,-1:]
            scores=scores-50*(a[:,::2]-cutoff)
    scores[:,0]=-torch.inf
    order=scores.argsort(-1,descending=True)
    selected=order[:,:r]
    untouched=order[:,r:].sort(-1).values
    return selected, untouched, dest.gather(1,selected)


def apply_bipartite(x, size, indices, mode):
    """Prune or merge matching groups; MLERP uses the maximum incoming feature norm."""
    src,unm,dst=indices
    a,b=x[:,::2],x[:,1::2]
    sa,sb=size[:,::2],size[:,1::2]
    def gather(t,idx):
        return t.gather(1,idx[...,None].expand(-1,-1,t.shape[-1]))
    keep=gather(a,unm);keep_size=gather(sa,unm)
    if mode=='prune':
        return torch.cat([keep,b],1),torch.cat([keep_size,sb],1)
    incoming=gather(a,src);incoming_size=gather(sa,src)
    index=dst[...,None].expand_as(incoming)
    if mode=='weighted':
        merged_size=sb.scatter_add(1,dst[...,None],incoming_size)
        merged=(b*sb).scatter_add(1,index,incoming*incoming_size)/merged_size
    elif mode=='mlerp':
        merged=b.scatter_reduce(1,index,incoming,reduce='mean',include_self=True)
        norms=b.float().norm(dim=-1,keepdim=True).scatter_reduce(1,dst[...,None],incoming.float().norm(dim=-1,keepdim=True),reduce='amax',include_self=True)
        merged=(F.normalize(merged.float(),dim=-1)*norms).to(x.dtype)
        merged_size=sb.scatter_add(1,dst[...,None],incoming_size)
    else:
        raise ValueError(mode)
    return torch.cat([keep,merged],1),torch.cat([keep_size,merged_size],1)


def per_block_r(depth,r):
    """r per block: a constant r, or [r, inflection] as in ToMe's parse_r (finer latency steps; a constant r is unchanged)."""
    from .methods import parse_r
    return parse_r(depth,(int(r[0]),float(r[1])) if isinstance(r,(list,tuple)) else int(r))


class RAPID(TokenViT):
    """RAPID equations 1–4; first four blocks prune, remaining blocks merge."""
    needs=('kmean','cls')

    def __init__(self,vit,spec,r):
        super().__init__(vit,spec)
        self.r=per_block_r(self.depth,r);self.prop=spec.pool!='avg'

    def forward(self,img):
        if not any(self.r):
            return self.vit(img)
        return super().forward(img)

    def reduce(self,l,x,size,info):
        if min(self.r[l],(x.shape[1]-1)//2)<=0:
            return x,size
        indices=bipartite_indices(info['kmean'],self.r[l],redundancy=l<4,cls_attention=info['cls'] if l>=4 else None)
        return apply_bipartite(x,size,indices,'prune' if l<4 else 'weighted')


class ToFu(TokenViT):
    """ToFu Algorithm 1, d=6, MLERP maximum-norm merge; paper reimplementation."""
    needs=('kmean',)

    def __init__(self,vit,spec,r,d=6):
        super().__init__(vit,spec)
        self.r,self.d=per_block_r(self.depth,r),int(d);self.prop=spec.pool!='avg'

    def forward(self,img):
        if not any(self.r):
            return self.vit(img)
        return super().forward(img)

    def reduce(self,l,x,size,info):
        if min(self.r[l],(x.shape[1]-1)//2)<=0:
            return x,size
        ids=bipartite_indices(info['kmean'],self.r[l])
        return apply_bipartite(x,size,ids,'prune' if l<self.d else 'mlerp')


def onemore_score(probs, v):
    """Received attention plus V statistic following the authors' released POMT metric."""
    received=probs[:,:,1:,1:].amax(1).sum(1)
    received=received/received.amax().clamp_min(torch.finfo(received.dtype).tiny)
    magnitude=v.amax(1).sum(-1).softmax(1)[:,1:]
    return received+magnitude


class OneMore(TokenViT):
    """One station at block index depth//4; n includes CLS and the inattentive token."""
    prop=False

    def __init__(self,vit,spec,n):
        super().__init__(vit,spec)
        self.n=int(n);self.site=self.depth//4
        if not 3<=self.n<=spec.n_tokens:
            raise ValueError('n must include CLS and at least two patch representatives')

    def forward(self,img):
        if self.n==self.spec.n_tokens:
            return self.vit(img)
        vit=self.vit
        x=vit.norm_pre(vit.patch_drop(vit._pos_embed(vit.patch_embed(img))))
        size=x.new_ones(x.shape[0],x.shape[1],1)
        for l,blk in enumerate(vit.blocks):
            out,info=attention(blk,blk.norm1(x),size,('attn',) if l==self.site else (),False)
            x=x+blk.drop_path1(blk.ls1(out))
            if l==self.site:
                score=onemore_score(info['attn'],info['v'])
                order=score.argsort(-1,descending=True)+1
                keep,drop=order[:,:self.n-2],order[:,self.n-2:]
                selected=x.gather(1,keep[...,None].expand(-1,-1,x.shape[-1]))
                inattentive=x.gather(1,drop[...,None].expand(-1,-1,x.shape[-1])).mean(1,keepdim=True)
                x=torch.cat([x[:,:1],selected,inattentive],1)
                size=torch.cat([size[:,:1],size.gather(1,keep[...,None]),size.gather(1,drop[...,None]).sum(1,keepdim=True)],1)
            x=x+blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(x))))
        x=vit.norm(x)
        if self.spec.pool=='avg':
            w=size[:,1:];x=(x[:,1:]*w).sum(1)/w.sum(1)
        else:
            x=x[:,0]
        return vit.head(vit.head_drop(vit.fc_norm(x)))


def utility_choice(counts,latencies,accuracies,alpha=.5):
    """Select n by POMT equations 1–3, with deterministic first-maximum ties."""
    import numpy as np
    n,l,a=np.asarray(counts),np.asarray(latencies,float),np.asarray(accuracies,float)
    if n.ndim!=1 or n.shape!=l.shape or n.shape!=a.shape or not len(n) or not np.isfinite(l).all() or not np.isfinite(a).all() or min(l)<=0 or max(a)<=0 or not 0<=alpha<=1:
        raise ValueError('invalid utility curves')
    utility=alpha*a/max(a)+(1-alpha)*(1-l/max(l))
    return dict(n=int(n[utility.argmax()]),alpha=alpha,utility=utility.tolist())


def build_posthoc(vit,spec,cfg):
    """Build a new baseline; the historical baseline registry is unchanged."""
    classes={'rapid':RAPID,'tofu':ToFu,'onemore':OneMore}
    kind=cfg['kind']
    return classes[kind](vit,spec,**{k:v for k,v in cfg.items() if k!='kind'})
