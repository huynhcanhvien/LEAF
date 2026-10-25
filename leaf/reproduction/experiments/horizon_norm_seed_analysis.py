"""Complete paired follow-up report, with independent numerical/statistical audit."""
from pathlib import Path
import json
import csv
import numpy as np
from scipy.stats import binomtest,norm
from leaf.evaluation.runner import load_json,save,config_hash
from leaf.evaluation.data import mcnemar,holm


def csv_write(path,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w') as stream:
        out=csv.DictWriter(stream,fieldnames=fields);out.writeheader();out.writerows(rows)


def analyze(root,bbs,levels):
    root=Path(root);rows=[];stats=[];look={};jobs_by_bb={}
    for bb in bbs:
        jobs=load_json(root/bb/'plans.json')['jobs'];jobs_by_bb[bb]=jobs
        for name,job in jobs.items():
            row=dict(backbone=bb,name=name,group=job['group'],level=job['level'],seed=job.get('seed'),
                     predicted_ms=job.get('predicted_ms'),budget_ms=job['budget']['latency_ms'],status=job.get('status','configured'))
            if 'config' not in job:rows.append(row);continue
            cfg=job['config'];id=bb+'-'+config_hash(cfg)
            row.update(config_sha=config_hash(cfg),config_json=json.dumps(cfg,sort_keys=True),
                       fixed_path=job.get('fixed_path',job['group']=='reference'))
            for split in ('confirm','val','v2'):
                result=load_json(root/'completed'/split/f'{id}.json')
                row[split+'_top1']=result['top1'];row[split+'_kl']=result['kl']
            times=[load_json(root/bb/'latency'/f'{id}#{i}.json')['measurement']['ms_mean'] for i in range(3)]
            row.update(latency_ms=float(np.mean(times)),latency_sd=float(np.std(times,ddof=1)),rounds_json=json.dumps(times))
            if job['group']=='calibration':row['overshoot_ms']=row['latency_ms']-row['budget_ms']
            rows.append(row);look[bb,name]=row
    def pair(bb,a,b,family):
        ja,jb=jobs_by_bb[bb][a],jobs_by_bb[bb][b]
        if 'config' not in ja or 'config' not in jb:return
        ra,rb=look[bb,a],look[bb,b]
        tau=max(.05,2*np.sqrt((ra['latency_sd']**2+rb['latency_sd']**2)/2))
        for split in ('val','v2'):
            aa=load_json(root/'completed'/split/f"{bb}-{config_hash(ja['config'])}.json")
            ba=load_json(root/'completed'/split/f"{bb}-{config_hash(jb['config'])}.json")
            with np.load(aa['arrays']) as z:x=z['correct'].astype(bool)
            with np.load(ba['arrays']) as z:y=z['correct'].astype(bool)
            p,w,l=mcnemar(x,y)
            stats.append(dict(backbone=bb,arm=a,reference=b,family=family,split=split,p=p,wins=w,losses=l,
                delta_pp=100*float(x.mean()-y.mean()),delta_latency_ms=ra['latency_ms']-rb['latency_ms'],
                tolerance_ms=float(tau),near_latency=bool(abs(ra['latency_ms']-rb['latency_ms'])<=tau),
                same_config=ja['config']==jb['config'],interpretation='fixed-path effect' if family!='calibration' else 'replanned; iso-latency only when eligible'))
    for bb in bbs:
        for level in levels:
            full=f'reference/{level}/full';m0=f'reference/{level}/M0_PD'
            for variant in ('future1','future3'):pair(bb,f'{variant}/{level}',full,'D_horizon')
            for variant in ('shuffle','shuffle_size'):pair(bb,f'{variant}/{level}',f'identity/{level}','D_placebo')
            pair(bb,f'identity/{level}',full,'D_implementation_control')
            for variant in ('mean_norm','rms_norm'):
                pair(bb,f'{variant}/{level}',full,'M3_norm');pair(bb,f'{variant}/{level}',m0,'M3_norm')
            pair(bb,f'mean_norm/{level}',f'rms_norm/{level}','M3_norm')
            for seed in (251,351):pair(bb,f'calibration/{level}/seed{seed}',full,'calibration')
    for family in {r['family'] for r in stats}:
        for split in ('val','v2'):holm([r for r in stats if r['family']==family and r['split']==split])
    csv_write(root/'all_arms.csv',rows);csv_write(root/'paired_comparisons.csv',stats)
    save(root/'analysis.json',dict(rows=rows,tests=stats,disclosure='All fixed-path effects reported with measured latency; seed replans require near_latency for iso-latency interpretation; no seed/model selection on accuracy'))


def verify(root,bbs,levels):
    from leaf.reproduction.experiments import horizon_norm_seed_ablation as main
    from leaf.reproduction.experiments.gmac_planning import hash_file,sha
    root=Path(root);main.check();sealed=load_json(root/'plans_sealed.json')
    assert hash_file(root/'tasks.json')==sealed['tasks_sha']
    checks=0;probe_checks=0;timing_checks=0
    labels={s:np.load(main.REPO/f'data/cache/{s}_labels.npy') for s in main.SPLITS}
    for bb,h in sealed['plans'].items():assert hash_file(root/bb/'plans.json')==h
    for task in load_json(root/'tasks.json'):
        for split,n in main.SPLITS.items():
            row=load_json(root/'completed'/split/f"{task['id']}.json");path=Path(row['arrays']);meta=load_json(path.with_suffix('.json'))
            assert hash_file(path)==row['arrays_sha'] and hash_file(path.with_suffix('.json'))==row['metadata_sha']
            assert meta['config']==task['config'] and meta['n']==n and meta['batch_size']==32
            with np.load(path) as a:
                assert all(a[k].shape==(n,) for k in ('pred','correct','kl')) and np.isfinite(a['kl']).all()
                assert np.array_equal(a['correct'],a['pred']==labels[split])
                assert abs(row['top1']-100*float(a['correct'].mean()))<1e-9
            checks+=1
        for rnd in range(3):
            row=load_json(root/task['backbone']/'latency'/f"{task['id']}#{rnd}.json")
            assert row['task']==task and row['round']==rnd and row['measurement']['ms_mean']>0
            timing_checks+=1
    pool_labels=np.load(main.REPO/'data/cache/calib_bal_labels.npy')
    for seed in main.SEEDS:
        m=load_json(root/'calibration'/f'{seed}.json');digest=m.pop('sha256');assert sha(m)==digest
        idx=np.array(m['indices']);assert len(idx)==5000 and len(np.unique(idx))==5000
        assert np.array_equal(pool_labels[idx],np.array(m['labels']))
        assert np.all(np.bincount(pool_labels[idx],minlength=1000)==5)
        for bb in bbs:
            for probe in load_json(root/bb/f'probe_means_seed{seed}.json'):
                path=Path(probe['arrays']);meta=load_json(path.with_suffix('.json'))
                assert hash_file(path)==probe['arrays_sha'] and hash_file(path.with_suffix('.json'))==probe['metadata_sha']
                assert meta['calibration_sha']==digest and meta['batch_size']==32 and meta['n']==5000
                with np.load(path) as a:
                    assert np.array_equal(a['correct'],a['pred']==pool_labels[idx]) and np.isfinite(a['kl']).all()
                probe_checks+=1
    previous=load_json(main.PREVIOUS/'design.json')
    for p,info in previous['data'].items():assert hash_file(p)==info['sha256']
    for p,info in previous['teachers'].items():assert hash_file(p)==info['sha256']
    data=load_json(root/'analysis.json')
    for t in data['tests']:
        b,c=t['wins'],t['losses'];n=b+c
        p=1. if n==0 else float(2*norm.sf(max(0,abs(b-c)-1)/np.sqrt(n))) if n>2000 else float(binomtest(b,n,.5).pvalue)
        assert abs(p-t['p'])<1e-10
    for family in {t['family'] for t in data['tests']}:
        for split in ('val','v2'):
            tests=sorted([t for t in data['tests'] if t['family']==family and t['split']==split],key=lambda t:t['p']);adj=0.
            for i,t in enumerate(tests):adj=max(adj,min(1.,t['p']*(len(tests)-i)));assert abs(adj-t['holm'])<1e-12
    save(root/'audit.json',dict(passed=True,arrays_checked=checks,probe_arrays_checked=probe_checks,
         timing_samples_checked=timing_checks,tests_checked=len(data['tests']),old_code_unchanged=True,independent_human_review=False))
