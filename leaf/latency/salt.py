"""Additive nonnegative latency proxies; existing planners and latency models untouched."""
import math
import numpy as np
from scipy.optimize import nnls
from leaf.planner.schedule_caps import block_gmacs, canonical

CORE = ['qkv','attention','mlp','similarity','d_cls','d_token','K','N_D','K_m3']
PHYSICAL = CORE + ['sort_work','station_io','merge_io','m3_io']
FORMULAS = {
    'gmac_only':['total_gmac'],
    'current_refit':['original_gmac','K','N_D','K_m3'],
    'station_rich':['original_gmac','K','N_D','K_m3','station_io','merge_io','d_token','m3_io'],
    'split_mac':CORE,
    'split_station':PHYSICAL,
    'split_position':PHYSICAL+['site_late','removed_late','d_late'],
    'split_hinge':PHYSICAL+['pre_h64','pre_h128','post_h64','post_h128'],
    'split_tile32':PHYSICAL+['pad_attn32','pad_mlp32'],
    'split_tile64':PHYSICAL+['pad_attn64','pad_mlp64'],
}


def constants(spec):
    from leaf.backbones import head_gmac
    return dict(original_gmac=((spec.n_tokens-1)*spec.dim**2)/1e9+head_gmac(spec),
        total_gmac=((spec.n_tokens-1)*3*getattr(spec,'patch',16)**2*spec.dim)/1e9+head_gmac(spec))


def edge_features(spec,l,n,p,m,active=False,groups=16,rank=32,m3=True):
    d=spec.dim;p=min(n,p);m=min(p,m);cut=m<n;merge=m<p
    # Triton pads D's group axis to at least 16, even when fewer groups exist.
    gp=max(16,1<<(max(1,int(groups))-1).bit_length()) if active else 0
    r=(rank or d) if active else 0;dep=l/max(1,spec.depth-1)
    qkv=4*n*d*d/1e9;attn=2*n*n*d/1e9;mlp=8*m*d*d/1e9
    similarity=(p-m)*m*d/1e9 if merge else 0.
    dc=2*d*gp*r/1e9;dt=n*gp*d/1e9
    out=dict(original_gmac=block_gmacs(n,p,m,d),
        total_gmac=qkv+attn+mlp+similarity+n*d/1e9+dc+dt,
        qkv=qkv,attention=attn,mlp=mlp,similarity=similarity,d_cls=dc,d_token=dt,
        K=float(cut),N_D=float(active),K_m3=float(m3 and merge),
        sort_work=n*math.log2(max(2,n))/1e6 if cut else 0.,
        station_io=n*d/1e6 if cut else 0.,merge_io=(p-m)*d/1e6,
        m3_io=(p+m)*d/1e6 if m3 and merge else 0.,
        site_late=dep*cut,removed_late=dep*(n-m)/spec.n_tokens,d_late=dep*active)
    for h in (64,128):
        out[f'pre_h{h}']=max(0,n-h)/spec.n_tokens
        out[f'post_h{h}']=max(0,m-h)/spec.n_tokens
    for tile in (32,64):
        out[f'pad_attn{tile}']=(math.ceil(n/tile)*tile-n)*d*d/1e9
        out[f'pad_mlp{tile}']=(math.ceil(m/tile)*tile-m)*d*d/1e9
    return out


def features(cfg,spec):
    out=constants(spec);n=spec.n_tokens
    g=canonical(dict(p=cfg['prune_keep'],m=cfg['merge_keep']),n)
    for l,(p,m) in enumerate(zip(g['p'],g['m'])):
        active=l<spec.depth-1 and float(cfg.get('alpha',{}).get(str(l),0))>0
        groups=len(set(cfg.get('groups',{}).get(str(l),[]))) if active else 0
        row=edge_features(spec,l,n,p,m,active,groups,cfg.get('rank'),cfg.get('base')=='pdm')
        for k,v in row.items():out[k]=out.get(k,0.)+v
        n=min(n,p,m)
    return out


def matrix(rows,name):return np.array([[r['features'].get(k,0.) for k in FORMULAS[name]] for r in rows],float)


def fit(x,y,alpha):
    """Nonnegative slopes, free intercept, train-only column standardization."""
    x=np.asarray(x);y=np.asarray(y);mean=x.mean(0);scale=x.std(0);scale=np.where(scale>1e-12,scale,1.)
    z=(x-mean)/scale;target=y-y.mean()
    if alpha>0:
        z=np.vstack([z,np.sqrt(alpha)*np.eye(x.shape[1])]);target=np.r_[target,np.zeros(x.shape[1])]
    coef,_=nnls(z,target,maxiter=10000);coef=coef/scale
    return dict(intercept=float(y.mean()-mean@coef),coef=coef.tolist(),alpha=float(alpha))


def predict(model,x):return model['intercept']+np.asarray(x)@model['coef']


class SaltCost:
    def __init__(self,model,spec,dcfg):
        self.model=model;self.spec=spec;self.dcfg=dcfg;self.names=FORMULAS[model['formula']]
        self.weights=np.asarray(model['coef'])
        c=constants(spec);self.c0=float(model['intercept']+sum(c.get(k,0)*v for k,v in zip(self.names,self.weights)))
    def edge(self,l,n,p,m,active=False,groups=16):
        f=edge_features(self.spec,l,n,p,m,active,groups,self.dcfg.rank,True)
        return float(sum(f.get(k,0)*v for k,v in zip(self.names,self.weights)))
    def __call__(self,cfg):
        f=features(cfg,self.spec)
        return float(self.model['intercept']+sum(f.get(k,0)*v for k,v in zip(self.names,self.weights)))


def plan_with_salt(curves,spec,dcfg,cost,budget):
    """Same inner-min, multiplier sweep and ratio projection as the existing paper DP."""
    from leaf.planner.surrogate import JointTables,lookup
    from leaf.planner.lagrange import lagrange_dp
    from leaf.planner.configs import norm_config
    jt=JointTables(curves,spec.n_tokens,spec.depth,step=getattr(spec,'count_step',4),dim=spec.dim)
    times=np.zeros_like(jt.mac)
    for l in range(spec.depth):
        for i,ni in enumerate(jt.counts):
            for j in range(i,len(jt.counts)):
                ni,nj=int(ni),int(jt.counts[j]);cut=j>i
                pp=nj+round((ni-nj)*jt.th[l,i,j]) if cut else ni
                active=cut and l<spec.depth-1 and jt.alpha[l,i,j]>0
                groups=len(set(dcfg.at(l))) if active else 0
                times[l,i,j]=cost.edge(l,ni,pp,nj,active,groups)
    def state(sites,counts):
        th,alphas=lookup(jt,sites,counts)
        return dict(sites=list(sites),counts=list(counts),thetas=th,alphas=alphas)
    def cfg(st):return norm_config(st,spec,dcfg,'M3')
    def fhat(st):
        n=spec.n_tokens;value=0.
        for l,c,th,a in zip(st['sites'],st['counts'],st['thetas'],st['alphas']):
            value+=jt.curves[(th,a)].transition(l,n,c);n=c
        return float(value)
    def project(raw):
        sites=raw['sites'];n=spec.n_tokens;ratios=[]
        for c in raw['counts']:ratios.append(c/n);n=c
        def at(eta):
            n=spec.n_tokens;counts=[]
            for r in ratios:
                c=max(3,min(n-1,round(n*r**eta)))
                if c>=n:return None
                counts.append(c);n=c
            st=state(sites,counts);config=cfg(st);t=cost(config)
            return dict(state=st,config=config,predicted_ms=t,fhat=fhat(st))
        samples=[(float(e),at(float(e))) for e in np.geomspace(.001,20,96)]
        hit=next((i for i,(_,v) in enumerate(samples) if v is not None and v['predicted_ms']<=budget-.1),None)
        if hit is None:return None
        hi,best=samples[hit];lo=samples[hit-1][0] if hit else .001
        for _ in range(40):
            mid=(lo+hi)/2;v=at(mid)
            if v is not None and v['predicted_ms']<=budget-.1:hi=mid;best=v
            else:lo=mid
        return best
    lo,hi=0.,1.;candidates=[];seen=set()
    for _ in range(24):
        raw=lagrange_dp(jt,times,times,hi)
        if raw is None:break
        if cost(cfg(state(raw['sites'],raw['counts'])))<=budget:break
        hi*=4
    for _ in range(32):
        mid=(lo+hi)/2;raw=lagrange_dp(jt,times,times,mid)
        if raw is None or cost(cfg(state(raw['sites'],raw['counts'])))>budget:lo=mid
        else:hi=mid
    for factor in (.5,.7,.85,1.,1.15,1.3,1.6):
        raw=lagrange_dp(jt,times,times,hi*factor)
        if raw is None:continue
        result=project(raw)
        if result is None:continue
        key=str(result['state'])
        if key not in seen:seen.add(key);candidates.append(result)
    if not candidates:return dict(infeasible=True,budget_ms=budget)
    best=min(candidates,key=lambda v:(v['fhat'],v['predicted_ms'],str(v['state'])))
    # Additivity on final off-grid schedule is essential to the DP interpretation.
    n=spec.n_tokens;edge_sum=cost.c0
    for l,(p,m) in enumerate(zip(best['config']['prune_keep'],best['config']['merge_keep'])):
        active=l<spec.depth-1 and float(best['config']['alpha'].get(str(l),0))>0
        groups=len(set(best['config']['groups'].get(str(l),[]))) if active else 0
        edge_sum+=cost.edge(l,n,p,m,active,groups);n=min(n,p,m)
    assert abs(edge_sum-best['predicted_ms'])<1e-7,(edge_sum,best['predicted_ms'])
    return dict(best,budget_ms=budget,n_candidates=len(candidates),
        certificate='fixed-multiplier DP exact; constrained multiplier search and ratio projection uncertified')


# ---- fitting procedure of the SALT models (copied unchanged from experiments/t_proxy_compare.py, which keeps its own) ----
ALPHAS=(0.,.01,.1,1.,10.,100.)   # ridge strengths tried by fit_salt()


def folds(rows,count=5,seed=151,blocked=False):
    groups=sorted({r['group'] for r in rows});count=min(count,len(groups))
    assert count>=2
    if blocked:
        groups.sort(key=lambda g:np.mean([r['gmac'] for r in rows if r['group']==g]))
        chunks=np.array_split(groups,count);assignment={g:i for i,chunk in enumerate(chunks) for g in chunk}
    else:
        order=np.random.default_rng(seed).permutation(groups)
        assignment={g:i%count for i,g in enumerate(order)}
    out=[]
    for f in range(count):
        te=np.array([assignment[r['group']]==f for r in rows]);tr=~te
        assert {r['group'] for r,t in zip(rows,tr) if t}.isdisjoint({r['group'] for r,t in zip(rows,te) if t})
        out.append((np.where(tr)[0],np.where(te)[0]))
    return out


def fit_salt(rows,name,seed=701):
    x=matrix(rows,name);y=np.array([r['ms'] for r in rows]);splits=folds(rows,3,seed)
    scores=[]
    for alpha in ALPHAS:
        errors=[]
        for tr,te in splits:
            model=fit(x[tr],y[tr],alpha);pred=predict(model,x[te]);errors.extend(abs(pred-y[te])/y[te])
        scores.append((float(np.mean(errors)),alpha))
    _,alpha=min(scores,key=lambda v:(v[0],-v[1]))
    return dict(fit(x,y,alpha),formula=name,feature_names=FORMULAS[name],n=len(rows),
        groups=len({r['group'] for r in rows}),tuning=[dict(alpha=a,mape=100*e) for e,a in scores])
