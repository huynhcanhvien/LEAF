"""CPU protocol tests; --paper-final also reproduces every locked M0 budget."""
from common import check, finish
import argparse
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from leaf.reproduction.experiments.merge_norm_runner import pair_design, latency_rounds, paired_tests
from leaf.reproduction.ablation_models.merge_operator_planner import NormLat
from leaf.latency.profile_model import ProfileLatencyModel
from leaf.backbones import SPECS


def main(real=False):
    from leaf.reproduction.experiments.merge_norm_evaluation import probe_metadata
    metadata=probe_metadata('M3',1,189,.5,1.)
    check('probe metadata does not override Runner fields',not set(metadata)&{'id','config','n','top1','kl','batch_size','seconds','gpu'})
    jobs={name:dict(config=dict(kind=name)) for name in ('dense','M0/latency/DR8.7','DR8.7','M3/latency/DR8.7','N4/latency/DR8.7')}
    means={'dense':30.,'M0/latency/DR8.7':20.,'DR8.7':21.,'M3/latency/DR8.7':20.02,'N4/latency/DR8.7':22.}
    rows={name:{i:{'32':mean+offset} for i,offset in enumerate((-.01,0.,.01))} for name,mean in means.items()}
    pairs,noise=pair_design(jobs,rows)
    check('session tau has paper-final floor', abs(noise['tau_ms']-.05)<1e-9)
    check('tolerant M3 pairs with current and DR', all(q['eligible'] for q in pairs if q['method']=='M3'))
    check('slower N4 pairs are ineligible', all(not q['eligible'] for q in pairs if q['method']=='N4'))
    class Fake:
        def cached(self,cfg,split):
            correct=np.zeros(1000,dtype=bool);correct[:600 if cfg['kind'].startswith('M3') else 500]=True
            return dict(correct=correct,kl=np.ones(1000))
    tests=paired_tests(Fake(),jobs,rows,pairs,'val')
    check('Holm applied within method across references', all('holm' in q for q in tests))
    check('paired improvement detected', all(q['delta']>9.9 and q['holm']<.05 for q in tests if q['method']=='M3'))
    for rich in (False,True):
        kind='gmac_station' if rich else 'gmac'
        model=dict(fit={'32':dict(chosen=kind,**{kind:dict(coef=[0.,1.,.2,.1]+([.1,.2,.3] if rich else []))})},
                   profile=dict(grid=[197,3],ms={'32/attn':[1.,1.],'32/mlp':[1.,1.]}))
        spec=SPECS['deit_b'];p=[197,160,197,120]+[197]*8;m=[197,150,197,120]+[197]*8
        a=ProfileLatencyModel(model,spec);b=NormLat(model,spec,.025)
        check(f'latency surcharge only on actual merge (rich={rich})',abs(b(p,m,ks={})-a(p,m,ks={})-.025)<1e-9)
        check(f'zero surcharge exactly legacy latency (rich={rich})',NormLat(model,spec,0.)(p,m,ks={})==a(p,m,ks={}))
    if real:
        from leaf.reproduction.experiments.merge_norm_evaluation import context,m0_regression,ORIGINAL
        with tempfile.TemporaryDirectory() as directory:
            args=SimpleNamespace(root=directory,baseline_root=ORIGINAL,data_cache='data/cache')
            for b in ('deit_b','mae_b'):
                r,d,base=context(args,b)
                m0_regression(args,r,d,base)
                check(b+' all M0 schedules match paper-final bit-for-bit', (r.root/'m0_planner_regression.json').exists())
    else:
        print('Full cached-paper-final regression: run with --paper-final')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--paper-final',action='store_true');args=p.parse_args()
    main(args.paper_final);finish()
