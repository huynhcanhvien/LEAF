"""GMAC-only counterpart of the paper's separable DP. No latency calls."""
import numpy as np
from leaf.planner.surrogate import JointTables,lookup,ProbeCurve
from leaf.planner.lagrange import lagrange_dp
from leaf.planner.configs import config_from_state
from leaf.planner.schedule_caps import full_cost,block_gmacs
from leaf.backbones import head_gmac

def config_cost(cfg,spec):
    groups={k:len(set(v)) for k,v in cfg.get('groups',{}).items()}
    row=full_cost(cfg['prune_keep'],cfg['merge_keep'],cfg.get('k'),spec.n_tokens,spec.dim,groups,cfg.get('rank'),getattr(spec,'patch',16),head_gmac(spec))
    # Same convention for all arms: matrix/conv MACs + grouped D, not scalar/comparison equivalents.
    return float(row['backbone_full']+row['d_gmac'])

def dense_cost(spec):
    n,d=spec.n_tokens,spec.dim
    return float(((n-1)*3*getattr(spec,'patch',16)**2*d+spec.depth*(12*n*d*d+2*n*n*d))/1e9+head_gmac(spec))

def make_budgets(spec,old_budgets,old_jobs):
    dr={name:config_cost(row['config'],spec) for name,row in old_jobs.items() if name.startswith('DR')}
    if not dr:
        return [dict(id=b['id'],gmac=dense_cost(spec)*float(b['id'][5:]),source='fraction of dense matrix MACs') for b in old_budgets]
    values=sorted(dr.values()); gap=(values[-1]-values[0])/(len(values)-1)
    return [dict(id=b['id'],gmac=dr[b['id']] if b['id'] in dr else values[0]-gap*(2 if b['id']=='extra2' else 1),
                 source='DR actual matrix MACs; extras extend mean DR GMAC gap') for b in old_budgets]

class GmacPlanner:
    def __init__(self,spec,dcfg,curves,mode):
        self.spec,self.dcfg,self.mode=spec,dcfg,mode
        self.jt=JointTables(curves,spec.n_tokens,spec.depth,step=getattr(spec,'count_step',4),dim=spec.dim)
        self.mac=np.zeros_like(self.jt.mac)
        for l in range(spec.depth):
            for i,ni in enumerate(self.jt.counts):
                for j in range(i,len(self.jt.counts)):
                    ni,nj=int(ni),int(self.jt.counts[j]); cut=j>i
                    pp=nj+round((ni-nj)*self.jt.th[l,i,j]) if cut else ni
                    value=block_gmacs(ni,pp,nj,spec.dim)
                    if cut and l<spec.depth-1 and self.jt.kk[l,i,j]>0:
                        G=len(set(dcfg.at(l))); rank=dcfg.rank or spec.dim
                        value+=(2*spec.dim*G*rank+ni*G*spec.dim)/1e9
                    self.mac[l,i,j]=value
        self.constant=((spec.n_tokens-1)*3*getattr(spec,'patch',16)**2*spec.dim)/1e9+head_gmac(spec)
    def cfg(self,state):
        cfg=config_from_state(state,self.spec,self.dcfg)
        return cfg if self.mode=='M0' else dict(cfg,kind='pdm',merge=dict(mode='M3',kernel='triton_norm'))
    def state(self,sites,counts):
        th,ks=lookup(self.jt,sites,counts)
        return dict(sites=list(sites),counts=list(counts),thetas=list(th),ks=list(ks))
    def fhat(self,state):
        n=self.spec.n_tokens; value=0.
        for l,c,t,k in zip(state['sites'],state['counts'],state['thetas'],state['ks']):
            value+=self.jt.curves[(t,k)].transition(l,n,c); n=c
        return float(value)
    def plan(self,budget):
        candidates=[]; seen=set()
        def candidate(raw):
            if raw is None: return
            state=self.state(raw['sites'],raw['counts'])
            # Same model-only ratio projection as paper; no measured feedback.
            ratios=[]; n=self.spec.n_tokens
            for c in state['counts']: ratios.append(c/n); n=c
            def at(eta):
                n=self.spec.n_tokens; counts=[]
                for ratio in ratios:
                    c=max(3,min(n-1,round(n*ratio**eta)))
                    if c>=n: return None
                    counts.append(c); n=c
                st=self.state(state['sites'],counts); cfg=self.cfg(st); cost=config_cost(cfg,self.spec)
                return st,cfg,cost
            value=(state,self.cfg(state),config_cost(self.cfg(state),self.spec))
            if value[2]>budget+1e-10:
                samples=[(float(eta),at(float(eta))) for eta in np.geomspace(.001,20,96)]
                hit=next((idx for idx,(_,v) in enumerate(samples) if v is not None and v[2]<=budget),None)
                if hit is None: return
                hi,value=samples[hit]; lo=samples[hit-1][0] if hit else .001
                for _ in range(40):
                    mid=(lo+hi)/2; v=at(mid)
                    if v is not None and v[2]<=budget: hi=mid; value=v
                    else: lo=mid
            key=str(value[0])
            if key not in seen:
                seen.add(key); candidates.append(dict(state=value[0],config=value[1],gmac=value[2],fhat=self.fhat(value[0])))
        lo,hi=0.,1.
        raw=lagrange_dp(self.jt,self.mac,self.mac,0.); candidate(raw)
        for _ in range(24):
            raw=lagrange_dp(self.jt,self.mac,self.mac,hi); candidate(raw)
            if raw is not None and config_cost(self.cfg(self.state(raw['sites'],raw['counts'])),self.spec)<=budget: break
            hi*=4
        for _ in range(48):
            mid=(lo+hi)/2; raw=lagrange_dp(self.jt,self.mac,self.mac,mid); candidate(raw)
            if raw is not None and config_cost(self.cfg(self.state(raw['sites'],raw['counts'])),self.spec)<=budget: hi=mid
            else: lo=mid
        for mu in np.geomspace(1e-8,max(hi*4,1e-7),80): candidate(lagrange_dp(self.jt,self.mac,self.mac,float(mu)))
        if not candidates: return dict(infeasible=True,budget=budget)
        best=min(candidates,key=lambda v:(v['fhat'],v['gmac'],str(v['state'])))
        return dict(best,budget=budget,candidates=len(candidates),objective='GMAC only; no latency inputs; surrogate selection only',certificate='fixed-multiplier DP exact; constrained projection not certified globally optimal')
