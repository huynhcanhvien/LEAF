"""Independent CPU semantics and GPU kernel gates for follow-up ablations."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from leaf.reproduction.ablation_models.ablation_vit import permute_d,compress_target
from leaf.model.mlerp_vit import compress_norm


def cpu():
    torch.manual_seed(1810);torch.set_num_threads(4)
    z=torch.tensor([[0.,1.,2.,3.,4.,5.]])
    sizes=torch.tensor([[[1.],[1.],[1.],[2.],[2.],[4.]]]);perm=torch.tensor([4,3,2,1,0])
    a=permute_d(z,sizes,perm,False);b=permute_d(z,sizes,perm,True)
    assert torch.equal(a[:,1:].sort().values,z[:,1:].sort().values) and a[0,0]==0
    for idx in ([1,2],[3,4],[5]):assert torch.equal(b[:,idx].sort().values,z[:,idx].sort().values)
    assert not torch.equal(a,z)
    # Exactly aligned features: norm(mean) already equals mean norm, RMS is independently known.
    x=torch.tensor([[[7.,0.],[1.,0.],[3.,0.],[5.,0.]]]);s=torch.ones(1,4,1);scores=torch.tensor([[0.,4.,2.,1.]])
    a,sa=compress_target(x,s,scores,4,2,'mean_norm')
    b,sb=compress_target(x,s,scores,4,2,'rms_norm')
    assert torch.equal(a[:,0],x[:,0]) and torch.equal(b[:,0],x[:,0])
    assert torch.allclose(a[0,1],torch.tensor([3.,0.]))
    assert abs(float(b[0,1].norm())-(35/3)**.5)<1e-6
    assert torch.equal(sa,sb)
    # Cancellation fallback and singleton copy.
    y=torch.tensor([[[0.,0.],[1.,-1.],[-1.,1.]]]);sc=torch.tensor([[0.,2.,1.]])
    for mode in ('mean_norm','rms_norm'):
        a,_=compress_target(y,torch.ones(1,3,1),sc,3,2,mode)
        assert torch.equal(a[:,0],y[:,0]) and torch.allclose(a[:,1],y[:,1])
    print('PASS CPU: placebo distribution/size bins, independently known norm targets, CLS/singleton/cancellation')


@torch.no_grad()
def gpu():
    torch.manual_seed(1810);rows=[]
    # Compare new kernels to independently executed CPU reference, including min tokens.
    for c in (64,384,768):
        for p,m in ((17,3),(65,17),(197,3)):
            x=torch.randn(2,p,c);s=torch.rand(2,p,1)+1;sc=torch.randn(2,p)
            for mode in ('mean_norm','rms_norm'):
                ref=compress_target(x,s,sc,p,m,mode)
                a=compress_target(x.cuda(),s.cuda(),sc.cuda(),p,m,mode)
                b=compress_target(x.cuda(),s.cuda(),sc.cuda(),p,m,mode)
                err=float((a[0].cpu()-ref[0]).abs().max())
                assert torch.allclose(a[0].cpu(),ref[0],atol=2e-5,rtol=2e-5),(c,p,m,mode,err)
                assert torch.allclose(a[1].cpu(),ref[1],atol=1e-5,rtol=1e-5)
                assert torch.equal(a[0],b[0]) and torch.equal(a[1],b[1])
                rows.append(dict(channels=c,p=p,m=m,variant=mode,max_abs_error=err))
    print('PASS GPU norm target kernels',len(rows));return rows


if __name__=='__main__':cpu()
