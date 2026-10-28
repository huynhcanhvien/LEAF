# Paper-final planner copied unchanged for additive merge experiments.
"""Final method: separable Lagrange DP and model-only ratio projection.

For fixed multipliers the grid DP is exact. Searching multipliers and adjusting
integer counts does not certify optimality at an arbitrary requested budget.
No measured latency projection, best-path list or true-KL candidate selection is used.
"""
import numpy as np
from leaf.planner.surrogate import lookup
from leaf.planner.schedule_caps import full_cost
from leaf.backbones import head_gmac


def lagrange_dp(jt, times, gmac, mu_t, mu_g=0., fixed_K=None):
    """Exact DP of Fhat + mu_T*T + mu_G*G, K free unless explicitly constrained."""
    if mu_t < 0 or mu_g < 0: raise ValueError('multipliers must be nonnegative')
    depth, width = jt.depth, len(jt.counts)
    if fixed_K is None:
        cost=np.full(width,np.inf); cost[0]=0.; back=[]
        allowed=np.triu(np.ones((width,width),bool))
        for l in range(depth):
            edge=jt.kl[l]+mu_t*times[l]+mu_g*gmac[l]
            edge=np.where(allowed,edge,np.inf)
            if l==0: edge=np.where(np.eye(width,dtype=bool),edge,np.inf)
            values=cost[:,None]+edge
            arg=values.argmin(0); cost=values[arg,np.arange(width)]; back.append(arg)
        j=int(cost.argmin())
        if not np.isfinite(cost[j]): return None
        value=float(cost[j]); path=[]
        for l in reversed(range(depth)):
            i=int(back[l][j])
            if j>i: path.append((l,int(jt.counts[j])))
            j=i
        path.reverse()
        return dict(sites=[l for l,_ in path],counts=[n for _,n in path],value=value,
                    mu_t=float(mu_t),mu_g=float(mu_g))
    K_max = depth - 1 if fixed_K is None else fixed_K
    if not 0 <= K_max < depth: raise ValueError('invalid K')
    cost = np.full((width,K_max+1), np.inf); cost[0,0] = 0.; back=[]
    for l in range(depth):
        new = np.full_like(cost,np.inf); arg = np.full((width,K_max+1,2),-1,int)
        for i in range(width):
            for j in range(i,width):
                cut = int(j>i)
                if cut and l==0: continue
                edge = jt.kl[l,i,j] + mu_t*times[l,i,j] + mu_g*gmac[l,i,j]
                if not np.isfinite(edge): continue
                for K in range(cut,K_max+1):
                    v = cost[i,K-cut] + edge
                    if v < new[j,K]: new[j,K] = v; arg[j,K] = (i,K-cut)
        cost=new; back.append(arg)
    if fixed_K is None: j,K = np.unravel_index(cost.argmin(),cost.shape)
    else: K=fixed_K; j=int(cost[:,K].argmin())
    if not np.isfinite(cost[j,K]): return None
    value=float(cost[j,K]); path=[]
    for l in reversed(range(depth)):
        i,prev_K=back[l][j,K]
        if j>i: path.append((l,int(jt.counts[j])))
        j,K=int(i),int(prev_K)
    path.reverse()
    return dict(sites=[l for l,_ in path],counts=[n for _,n in path],value=value,
                mu_t=float(mu_t),mu_g=float(mu_g))


def metrics(p, st):
    cfg=p.cfg(st)
    return dict(latency_ms=float(p.lat(cfg['prune_keep'],cfg['merge_keep'],alphas=cfg['alpha'])),
                full_gmac=full_cost(cfg['prune_keep'],cfg['merge_keep'],cfg['alpha'],p.spec.n_tokens,
                                    p.spec.dim,{l:len(set(v)) for l,v in cfg['groups'].items()},p.dcfg.rank,getattr(p.spec,'patch',16),
                                    head_gmac(p.spec))['full'])


def project(p, raw, budget, gmac_cap=None, margin=.1):
    """Uniform keep-ratio exponent; re-lookup settings for each adjusted state."""
    target=budget-margin
    if target<=0: return None
    sites,counts=raw['sites'],raw['counts']; n=p.spec.n_tokens; ratios=[]
    for c in counts: ratios.append(c/n); n=c
    def at(eta):
        out=[]; n=p.spec.n_tokens
        for r in ratios:
            c=max(3,min(n-1,round(n*r**eta)))
            if c>=n: return None
            out.append(c); n=c
        th,alphas=lookup(p.jt,sites,out)
        st=dict(sites=list(sites),counts=out,thetas=th,alphas=alphas)
        ms=metrics(p,st)
        if not all(np.isfinite(v) for v in ms.values()): return None
        return st,ms
    def feasible(v):
        return v is not None and v[1]['latency_ms']<=target and (gmac_cap is None or v[1]['full_gmac']<=gmac_cap)
    # Explicit scan brackets integer rounding and rejects impossible schedules.
    samples=[(float(e),at(float(e))) for e in np.geomspace(.001,20.,96)]
    hit=next((i for i,(_,v) in enumerate(samples) if feasible(v)),None)
    if hit is None: return None
    hi,v=samples[hit]; lo=samples[hit-1][0] if hit else .001
    best=v
    for _ in range(40):
        mid=(lo+hi)/2; v=at(mid)
        if feasible(v): hi=mid; best=v
        else: lo=mid
    return dict(state=best[0],**best[1],fhat=float(p.fhat(best[0])))


def plan(p, latency_budget, gmac_cap=None, fixed_K=None):
    if latency_budget<=0 or (gmac_cap is not None and gmac_cap<=0): raise ValueError('invalid budget')
    # These tables use the same inner-min (theta,alpha) choices as the error tables.
    times,gm=transition_costs(p)
    candidates=[]; seen=set()
    # mu_G=0 is exactly the latency-only case. A finite multiplier sweep is not
    # a certificate of a constrained optimum; every fixed pair is solved exactly.
    for mu_g in ((0.,.02,.1,.5,2.,10.) if gmac_cap is not None else (0.,)):
        lo,hi=0.,1.
        def solve(mu): return lagrange_dp(p.jt,times,gm,mu,mu_g,fixed_K)
        for _ in range(24):
            r=solve(hi)
            if r is None: break
            st=p.state(r['sites'],r['counts'])
            if metrics(p,st)['latency_ms']<=latency_budget: break
            hi*=4
        for _ in range(32):
            mid=(lo+hi)/2; r=solve(mid)
            if r is None or metrics(p,p.state(r['sites'],r['counts']))['latency_ms']>latency_budget: lo=mid
            else: hi=mid
        for f in (.5,.7,.85,1.,1.15,1.3,1.6):
            r=solve(hi*f)
            if r is None: continue
            v=project(p,r,latency_budget,gmac_cap)
            if v is None: continue
            key=(tuple(v['state']['sites']),tuple(v['state']['counts']))
            if key in seen: continue
            seen.add(key); candidates.append(dict(v,grid=r))
    if not candidates: return dict(state=None,infeasible=True,latency_budget=latency_budget,gmac_cap=gmac_cap)
    best=min(candidates,key=lambda v:(v['fhat'],v['latency_ms'],v['full_gmac']))
    return dict(best,latency_budget=latency_budget,gmac_cap=gmac_cap,candidates=len(candidates),
                selection='separable surrogate only',measured_projection=False,
                certificate='fixed-multiplier grid optimum only; projection and multiplier sweep are uncertified')


def transition_costs(p):
    """Selected-setting transition costs, including actual grouped-D sizes."""
    import math
    from leaf.planner.schedule_caps import block_gmacs
    jt=p.jt; times=np.zeros_like(jt.mac); gm=np.zeros_like(jt.mac)
    for l in range(jt.depth):
        for i,ni in enumerate(jt.counts):
            ni=int(ni)
            for j in range(i,len(jt.counts)):
                nj=int(jt.counts[j]); cut=j>i
                pp=nj+round((ni-nj)*jt.th[l,i,j]) if cut else ni
                active=cut and jt.alpha[l,i,j]>0 and l<jt.depth-1
                G=len(set(p.dcfg.at(l))) if active else 0
                times[l,i,j]=p.lat.block(ni,pp,nj)
                gm[l,i,j]=block_gmacs(ni,pp,nj,p.spec.dim)
                if cut:
                    times[l,i,j]+=p.lat.station(ni,pp,nj,active,groups=G)
                    scalar=ni*(math.ceil(math.log2(max(ni,2)))+12)+(pp-nj)*(p.spec.dim+2)+nj*p.spec.dim
                    if active:
                        r=p.dcfg.rank or p.spec.dim
                        gm[l,i,j]+=(2*p.spec.dim*G*r+ni*G*p.spec.dim)/1e9
                        scalar+=ni*G*8+ni*p.spec.dim*5
                    gm[l,i,j]+=scalar/1e9
    return times,gm
