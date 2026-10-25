"""Meaningful CPU regression tests for new operator-aware campaign planning."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from types import SimpleNamespace
import numpy as np
from leaf.planner.surrogate import ProbeCurve
from leaf.planner.plan import OperatorCost,operator_config,plan_latency
from leaf.latency.salt import SaltCost,FORMULAS,plan_with_salt


def main():
    spec=SimpleNamespace(n_tokens=17,depth=3,dim=16)
    dc=SimpleNamespace(rank=4,triton=False,at=lambda l:[0,0])
    names=FORMULAS['station_rich']
    model=dict(formula='station_rich',feature_names=names,coef=[1000.,.01,.005,.02,.001,.001,.001,.002],intercept=.05)
    cfg=dict(kind='pdm',merge=dict(mode='M3',kernel='triton_norm'),prune_keep=[17,15,7],merge_keep=[17,9,3],
             k={'1':2.},groups={'1':[0,0]},rank=4,triton=False,fuse_merge=True)
    original=SaltCost(model,spec,dc)
    m3=OperatorCost(model,spec,dc,'M3');m0=OperatorCost(model,spec,dc,'M0')
    assert abs(original(cfg)-m3(cfg))<1e-12
    assert m0(cfg)<m3(cfg), 'M0 must not pay M3 overhead'
    for cost,mode in ((m3,'M3'),(m0,'M0')):
        n=17;value=cost.c0
        c=operator_config(cfg,mode)
        for l,(p,m) in enumerate(zip(c['prune_keep'],c['merge_keep'])):
            active=str(l) in c['k'];value+=cost.edge(l,n,p,m,active,1 if active else 0);n=min(n,p,m)
        assert abs(value-cost(c))<1e-10
    off=operator_config(cfg,'M3',False)
    assert not off['k'] and not off['groups'] and off['rank'] is None
    assert m3(off)<m3(cfg), 'D-off must not pay D overhead'
    cv={(th,k):ProbeCurve({l:{4:.01/(1+k),8:.04/(1+k),14:.2/(1+k)} for l in (1,2)},17)
        for th in (0.,.5,1.) for k in (0.,1.)}
    expected=plan_with_salt(cv,spec,dc,original,.5)
    a=plan_latency(cv,spec,dc,model,.5)
    assert a['state']==expected['state'] and a['config']==expected['config'], 'default M3 planner must preserve schedules'
    b=plan_latency(cv,spec,dc,model,.5,'M0',False)
    assert not b.get('infeasible') and b['config']['kind']=='pd' and not b['config']['k']
    assert b['predicted_ms']<=.4+1e-8
    c=plan_latency(cv,spec,dc,model,.001)
    assert c.get('infeasible') and not c['global_infeasibility_proven']
    print('PASS: default planner parity, M0/M3 additive costs, D-off cost and settings, low-budget constraints, honest infeasibility')


if __name__=='__main__':main()
