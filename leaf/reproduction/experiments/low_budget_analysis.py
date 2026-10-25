"""Auditable paired analysis for the additive low-budget experiment."""
import csv
import json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest, norm
from leaf.evaluation.runner import load_json,save,config_hash
from leaf.evaluation.data import mcnemar,holm


def write_csv(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def bootstrap_factorial(arrays,seed=1810,repeats=5000):
    # Exact empirical distribution of per-image paired interaction values.
    values=100*(arrays['m3_pd'].astype(int)-arrays['m3_p'].astype(int)-arrays['m0_pd'].astype(int)+arrays['m0_p'].astype(int))
    unique,counts=np.unique(values,return_counts=True);n=len(values)
    draws=np.random.default_rng(seed).multinomial(n,counts/n,size=repeats)@unique/n
    return dict(interaction_pp=float(values.mean()),ci_low=float(np.quantile(draws,.025)),
                ci_high=float(np.quantile(draws,.975)),bootstrap_samples=repeats,seed=seed,n=n,
                note='paired-image percentile interval; exploratory, no multiplicity-adjusted interaction test')


def analyze(root,bbs,ablation):
    root=Path(root);rows=[];lookup={};jobs_by_bb={};cache={};tests=[];coverage=[];effects=[]
    def arrays(bb,cfg,split):
        key=(bb,config_hash(cfg),split)
        if key not in cache:
            taskid=bb+'-'+config_hash(cfg);p=root/'completed'/split/f'{taskid}.json'
            if not p.exists() or load_json(p)['status']!='ok':return None
            with np.load(load_json(p)['arrays']) as a:cache[key]=a['correct'].astype(bool)
        return cache[key]
    for bb in bbs:
        jobs=load_json(root/bb/'plans.json')['jobs'];jobs_by_bb[bb]=jobs
        for name,job in jobs.items():
            row=dict(backbone=bb,name=name,group=job.get('group'),axis=job.get('axis'),status=job.get('status','configured'),
                     nominal_latency_ms=job.get('budget',{}).get('latency_ms'),nominal_gmac=job.get('budget',{}).get('gmac'),
                     predicted_ms=job.get('predicted_ms'),prediction_main_ms=job.get('prediction_main_ms'))
            if 'config' not in job:
                rows.append(row);lookup[bb,name]=row;continue
            cfg=job['config'];id=bb+'-'+config_hash(cfg)
            row.update(config_sha=config_hash(cfg),kind=cfg['kind'],config_json=json.dumps(cfg,sort_keys=True))
            details=job.get('details',{})
            for field in ('K','N_D','final_tokens','total_gmac','backbone_full_gmac','d_gmac'):row[field]=details.get(field)
            row['token_path_json']=json.dumps(details.get('token_path'))
            row['feature_extrapolation_json']=json.dumps(job.get('feature_extrapolation',{}))
            for split in ('confirm','val','v2'):
                p=root/'completed'/split/f'{id}.json'
                result=load_json(p)
                row[split+'_status']=result['status'];row[split+'_top1']=result.get('top1');row[split+'_kl']=result.get('kl')
            profile=root/bb/'profiles'/f'{config_hash(cfg)}.json'
            if row.get('total_gmac') is None and profile.exists():row['total_gmac']=load_json(profile).get('counted_gmac')
            times=[]
            for rnd in range(3):
                p=root/bb/'latency'/f'{id}#{rnd}.json'
                if p.exists():
                    measurement=load_json(p)
                    if measurement['clean'] and 'ms_mean' in measurement['measurement']:times.append(measurement['measurement']['ms_mean'])
            if len(times)==3:
                row.update(latency_ms=float(np.mean(times)),latency_sd=float(np.std(times,ddof=1)),rounds_json=json.dumps(times))
                if job.get('axis')=='latency':
                    row['overshoot_ms']=row['latency_ms']-job['budget']['latency_ms']
                    row['budget_pass']=row['overshoot_ms']<=0
            rows.append(row);lookup[bb,name]=row
    def contrast(bb,a,b,family,axis=None):
        ja,jb=jobs_by_bb[bb][a],jobs_by_bb[bb][b]
        if 'config' not in ja or 'config' not in jb:return
        ra,rb=lookup[bb,a],lookup[bb,b]
        eligible=True;tau=None
        if axis=='latency':
            if ra.get('latency_ms') is None or rb.get('latency_ms') is None:return
            tau=max(.05,2*np.sqrt((ra['latency_sd']**2+rb['latency_sd']**2)/2))
            eligible=abs(ra['latency_ms']-rb['latency_ms'])<=tau
        if axis=='gmac':
            if ra.get('total_gmac') is None or rb.get('total_gmac') is None:return
            tau=max(.001,.005*ra['total_gmac']);eligible=abs(ra['total_gmac']-rb['total_gmac'])<=tau
        for split in ('val','v2'):
            aa,ba=arrays(bb,ja['config'],split),arrays(bb,jb['config'],split)
            if aa is None or ba is None:continue
            p,w,l=mcnemar(aa,ba)
            dominance=False
            if axis=='latency' and aa.mean()>ba.mean() and eligible:
                dominance=all(x<=y-tau for x,y in zip(json.loads(ra['rounds_json']),json.loads(rb['rounds_json'])))
            tests.append(dict(backbone=bb,arm=a,reference=b,family=family,axis=axis or 'direct',split=split,
                eligible=bool(eligible),tolerance=tau,p=p,wins=w,losses=l,delta_pp=100*float(aa.mean()-ba.mean()),
                accuracy_positive_and_faster_all_rounds=dominance,config_equal=ja['config']==jb['config']))
    for bb in bbs:
        jobs=jobs_by_bb[bb]
        refs=[n for n,j in jobs.items() if j.get('group')=='baseline' and 'config' in j]
        families=sorted({jobs[n]['config']['kind'] if not (jobs[n]['config']['kind']=='pd') else 'diffrate' for n in refs})
        for name,j in jobs.items():
            if j.get('group')!='full' or 'config' not in j:continue
            a=lookup[bb,name]
            for axis in ('latency','gmac'):
                for family in families:
                    eligible=[]
                    for ref in refs:
                        cfg=jobs[ref]['config'];kind='diffrate' if cfg['kind']=='pd' else cfg['kind']
                        if kind!=family:continue
                        b=lookup[bb,ref]
                        if axis=='latency' and a.get('latency_ms') is not None and b.get('latency_ms') is not None:
                            tau=max(.05,2*np.sqrt((a['latency_sd']**2+b['latency_sd']**2)/2))
                            if abs(a['latency_ms']-b['latency_ms'])<=tau:eligible.append((abs(a['latency_ms']-b['latency_ms']),b['latency_ms'],ref))
                        elif axis=='gmac' and a.get('total_gmac') is not None and b.get('total_gmac') is not None:
                            tau=max(.001,.005*a['total_gmac'])
                            if abs(a['total_gmac']-b['total_gmac'])<=tau:eligible.append((abs(a['total_gmac']-b['total_gmac']),b['total_gmac'],ref))
                    ref=min(eligible)[2] if eligible else None
                    coverage.append(dict(backbone=bb,arm=name,baseline_family=family,axis=axis,reference=ref,matched=bool(ref)))
                    if ref:contrast(bb,name,ref,'baseline_'+axis,axis)
        if bb not in ablation:continue
        for level in ('low90','low80','low70'):
            full=f'full/latency/{level}'
            for group in ('replan','fixed'):
                names={arm:f'{group}/{level}/{arm}' for arm in ('m0_p','m0_pd','m3_p')};names['m3_pd']=full
                for a,b in (('m3_pd','m3_p'),('m0_pd','m0_p'),('m3_pd','m0_pd'),('m3_p','m0_p'),('m3_pd','m0_p'),('m0_pd','m3_p')):
                    contrast(bb,names[a],names[b],'D_M3_'+group,'latency' if group=='replan' else None)
                for split in ('val','v2'):
                    cfgs={arm:jobs.get(name,{}).get('config') for arm,name in names.items()}
                    if not all(cfgs.values()):continue
                    ar={arm:arrays(bb,cfg,split) for arm,cfg in cfgs.items()}
                    if any(a is None for a in ar.values()):continue
                    effects.append(dict(backbone=bb,level=level,group=group,split=split,
                        D_under_M0_pp=100*float(ar['m0_pd'].mean()-ar['m0_p'].mean()),
                        D_under_M3_pp=100*float(ar['m3_pd'].mean()-ar['m3_p'].mean()),
                        M3_under_P_pp=100*float(ar['m3_p'].mean()-ar['m0_p'].mean()),
                        M3_under_PD_pp=100*float(ar['m3_pd'].mean()-ar['m0_pd'].mean()),**bootstrap_factorial(ar)))
            contrast(bb,full,f'pricing/{level}/no_overhead','pricing','latency')
        # Cross every latency point against every GMAC-control point; eligibility is measured, not ordinal label.
        for la in ('low90','low80','low70'):
            for ga in ('low90','low80','low70'):
                contrast(bb,f'full/latency/{la}',f'full/gmac/{ga}','pricing','latency')
    for family in sorted({t['family'] for t in tests}):
        for split in ('val','v2'):
            # Include unmatched contrasts in multiplicity correction too, but never use them for iso-cost claims.
            group=[t for t in tests if t['family']==family and t['split']==split]
            holm(group)
    for t in tests:t['measured_dominance']=bool(t['eligible'] and t.get('holm',1)<.05 and t['accuracy_positive_and_faster_all_rounds'])
    write_csv(root/'all_arms.csv',rows);write_csv(root/'paired_comparisons.csv',tests)
    write_csv(root/'coverage.csv',coverage);write_csv(root/'factorial_effects.csv',effects)
    save(root/'analysis.json',dict(rows=rows,tests=tests,coverage=coverage,effects=effects,
         protocol='Holm across all contrasts/backbone/budgets in predeclared family/split; cost-only eligibility; unmatched adjusted p not used for iso-cost claims'))


def audit_statistics(root):
    data=load_json(Path(root)/'analysis.json')
    for t in data['tests']:
        b,c=t['wins'],t['losses'];n=b+c
        p=1. if not n else float(2*norm.sf(max(0,abs(b-c)-1)/np.sqrt(n))) if n>2000 else float(binomtest(b,n,.5).pvalue)
        assert abs(p-t['p'])<1e-10
    for family in {t['family'] for t in data['tests']}:
        for split in ('val','v2'):
            group=sorted([t for t in data['tests'] if t['family']==family and t['split']==split],key=lambda t:t['p'])
            h=0.
            for i,t in enumerate(group):
                h=max(h,min(1.,t['p']*(len(group)-i)));assert abs(h-t['holm'])<1e-12
