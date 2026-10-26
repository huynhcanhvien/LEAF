"""Fixed-path D-horizon/placebo and M3-norm ablations; no legacy changes."""
from copy import deepcopy
import math
import torch
import torch.nn.functional as F
from leaf.model.mlerp_vit import MlerpViT,MlerpConfig,compress_norm
from leaf.forecast.attention_forecast import AttentionForecast
from leaf.model.token_ops import cls_attention,zscore,compress
from leaf.model.merge_variants import gather,segment,_set_norm


def permute_d(zd,size,priority,conditional=False):
    body=zd[:,1:];p=body.shape[1]
    if conditional:
        bins=torch.floor(torch.log2(size[:,1:,0].clamp_min(1))).long().clamp(0,63)
        natural=bins.argsort(dim=1,stable=True)
        source=(bins*p+priority[None,:]).argsort(dim=1,stable=True)
        body=torch.zeros_like(body).scatter(1,natural,body.gather(1,source))
    else:body=body[:,priority]
    return torch.cat((torch.zeros_like(zd[:,:1]),body),1)


def compress_target(x,size,scores,p,m,mode):
    b,n,c=x.shape;p=min(n,p);m=min(p,m)
    if p==m:return compress_norm(x,size,scores,p,m,'M3')[:2]
    order=torch.cat((torch.zeros(b,1,device=x.device,dtype=torch.long),scores[:,1:].argsort(-1,descending=True)+1),1)[:,:p]
    y,s=gather(x,order),gather(size,order)
    metric=F.normalize(y.float(),dim=-1);sim=metric[:,m:]@metric[:,:m].transpose(1,2);sim[...,0]=-math.inf
    rho,dest=sim.max(-1);rho=rho.clamp_min(0)
    if x.is_cuda and not torch.is_grad_enabled():
        from leaf.model.kernels.norm_target_merge_kernel import merge_target
        return merge_target(y,s,dest,rho,m,mode)
    mean,os,_,_=compress(x,size,scores,p,m,fuse_merge=False)
    ids=torch.cat((torch.arange(m,device=x.device)[None].expand(b,-1),dest),1)
    norms=y.float().norm(dim=-1,keepdim=True)
    count=segment(torch.ones_like(norms),ids,m)
    desired=segment(norms.square() if mode=='rms_norm' else norms,ids,m)/count
    if mode=='rms_norm':desired=desired.sqrt()
    corrected=_set_norm(mean,y,ids,m,desired)
    return torch.where(count>1,corrected,mean),os


class ExtraViT(MlerpViT):
    def __init__(self,vit,cfg,pool='token'):
        cfg=deepcopy(cfg);self.ablation=cfg.pop('ablation');cfg.pop('kind',None)
        super().__init__(vit,MlerpConfig(**cfg),pool)
        variant=self.ablation['variant'];n=self.n_tokens
        for l in range(self.n_blocks):
            key=str(l)
            if key in self.future:
                if variant in ('future1','future3'):
                    horizon=1 if variant=='future1' else 3
                    end=min(self.n_blocks,l+1+horizon)
                    labels=list(self.cfg.groups[key][:(end-l-1)*vit.blocks[0].attn.num_heads])
                    f=AttentionForecast([vit.blocks[j] for j in range(l+1,end)],labels,self.cfg.rank)
                    if self.cfg.triton:f.use_triton()
                    self.future[key]=f
                if variant in ('shuffle','shuffle_size','identity'):
                    rng=torch.Generator().manual_seed(int(self.ablation.get('seed',1810))+l*1009+n)
                    self.register_buffer('priority_'+key,torch.randperm(n-1,generator=rng).to(vit.pos_embed.device),persistent=False)
            n=min(n,self.cfg.prune_keep[l],self.cfg.merge_keep[l])

    def score(self,l,x,size,raw_p):
        key=str(l);variant=self.ablation['variant']
        if key in self.future and variant in ('shuffle','shuffle_size','identity'):
            zd=self.future[key].zscored(x,size,None,1.)
            if variant!='identity':zd=permute_d(zd,size,getattr(self,'priority_'+key),variant=='shuffle_size')
            return zscore(raw_p.clamp_min(1e-12).log(),1)+self.cfg.k[key]*zd
        return super().score(l,x,size,raw_p)

    def forward(self,img):
        variant=self.ablation['variant']
        if variant not in ('mean_norm','rms_norm'):return super().forward(img)
        x=self.embed(img);size=x.new_ones(x.shape[0],x.shape[1],1)
        for l,block in enumerate(self.blocks):
            out,p=cls_attention(block,block.norm1(x),None if l==0 else size)
            x=x+block.drop_path1(block.ls1(out))
            if l in self.stations:x,size=compress_target(x,size,self.score(l,x,size,p),self.cfg.prune_keep[l],self.cfg.merge_keep[l],variant)
            x=x+block.drop_path2(block.ls2(block.mlp(block.norm2(x))))
        return self.head(x,size)
