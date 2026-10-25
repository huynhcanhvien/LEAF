"""CPU-only final audit/export enrichment after the locked campaign completes.

Does not change inference code, frozen manifests, comparisons or cached predictions.
"""
import argparse
import datetime
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from leaf.evaluation.runner import config_hash,load_json,save
from leaf.reproduction.runners.sealed_stage_protocol import Stage
from leaf.reproduction.experiments.merge_norm_runner import pair_design,paired_tests
from leaf.reproduction.experiments.merge_norm_evaluation import ROOT,ORIGINAL,GATE,BACKBONES,context,validate


def finish(args):
    root=Path(args.root)
    if load_json(root/'status.json')['state']!='complete':raise RuntimeError('campaign is not complete yet')
    validate(args)
    audit={}
    for b in BACKBONES:
        r,_,_=context(args,b);stage=Stage(r,'stage2');frozen=stage.sealed();jobs=frozen['jobs']
        rows=stage.lat_table();pairs,noise=pair_design(jobs,rows)
        assert load_json(stage.dir/'pairs.json')['pairs']==pairs
        assert load_json(stage.dir/'pairs.json')['sha256']==frozen['sha256']
        assert load_json(stage.dir/'noise.json')==noise
        latfiles=list((stage.dir/'lat').glob('*.json'))
        assert len(latfiles)==len(jobs)*3
        for p in latfiles:
            row=load_json(p);assert row['frozen_sha']==frozen['sha256'] and row['config']==jobs[row['name']]['config']
        counts={}
        for split,n in (('half',5000),('confirm',20018),('val',50000),('v2',10000)):
            paths=list((r.root/split/'runs').glob('*.json'));counts[split]=len(paths)
            for path in paths:
                meta=load_json(path)
                assert meta['n']==n and meta['batch_size']==32 and config_hash(meta['config'])==path.stem
                with np.load(path.with_suffix('.npz')) as z:
                    assert all(z[k].shape==(n,) for k in ('pred','correct','kl'))
                    assert np.isfinite(z['kl']).all()
        comparisons=load_json(r.root/'comparisons.json')
        for split in ('val','v2'):
            fresh=paired_tests(r,jobs,rows,pairs,split)
            assert fresh==comparisons[split], (b,split,'paired comparison differs')
        exported=Path('schedules/kappa_form/merge_variants')/f'{b}.json';doc=load_json(exported)
        for name,entry in doc['schedules'].items():
            mode,method,budget=name.split('/')
            search=load_json(r.root/'search'/mode/f'{method}_{budget}.json')
            assert entry['config']==jobs[name]['config']==search['config']
            entry['planner']=dict(mode=mode,surrogate=search['fhat'],predicted_latency_ms=search['latency_ms'],
                                  full_gmac=search['full_gmac'],calib_kl=search['calib_kl'],merge_overhead_ms=search['merge_overhead_ms'])
            entry['measured']['latency_rounds_ms']=[rows[name][i]['32'] for i in range(3)]
            entry['measured']['session_tau_ms']=noise['tau_ms']
            entry['measured']['kl_v2']=float(r.cached(entry['config'],'v2')['kl'].mean())
            entry['equal_mark']='=' if entry.get('equal_to') else ''
        save(exported,doc)
        audit[b]=dict(caches_validated=counts,latency_rows=len(latfiles),frozen_sha=frozen['sha256'],tau_ms=noise['tau_ms'],
                      schedules_exported=len(doc['schedules']),comparisons_reproduced=True)
    result=dict(time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),audit=audit,report='results/reports/merge-variants-stage2-2026-10-16.md')
    save(root/'final_audit.json',result)
    report=Path(result['report']);text=report.read_text()
    summary_heading='## Summary against current LEAF at measured latency'
    if summary_heading not in text:
        claims=load_json(root/'decision.json')['claims']
        summary=['',summary_heading,'',
            'The means below cover only the points that can be paired under the locked latency rule; they are not a pooled test. Δ is in percentage points. Holm keeps the family (backbone, method, split) unchanged, covering both reference families and both budget types.', '',
            '| Backbone | Operator | Paired points | Mean Δval | Val significant wins/losses | Mean ΔV2 | V2 significant wins/losses | Meets val claim rule |',
            '|---|---|---:|---:|---:|---:|---:|---|']
        for b in BACKBONES:
            comparisons=load_json(root/b/'comparisons.json')
            for mode in ('M3','N4'):
                val=[q for q in comparisons['val'] if q['method']==mode and q['reference']=='LEAF-M0']
                v2=[q for q in comparisons['v2'] if q['method']==mode and q['reference']=='LEAF-M0']
                eligible=[q for q in val if q['eligible']]
                independent=[q for q in v2 if q['eligible']]
                mean=lambda rows: f"{np.mean([q['delta'] for q in rows]):+.3f}" if rows else '—'
                signals=lambda rows: f"{sum(q['delta']>0 and q['holm']<.05 for q in rows)}/{sum(q['delta']<0 and q['holm']<.05 for q in rows)}"
                summary.append(f"| {b} | {mode} | {len(eligible)}/{len(val)} | {mean(eligible)} | {signals(eligible)} | {mean(independent)} | {signals(independent)} | {claims[b][mode]} |")
        insert=text.index('\n',text.index('# '))
        text=text[:insert]+'\n'+'\n'.join(summary)+'\n'+text[insert:]
    heading='## Integrity check after completion'
    if heading not in text:
        text+='\n'+heading+'\n\n'
        text+='Checked shape/count/batch/config/NPZ for every new cache, the manifest SHA and every latency row; recomputed McNemar/Holm match exactly. Exported schedules include the surrogate, predicted latency/three measurement rounds, GMAC and calib KL.\n\n'
        text+='| Backbone | Half cache | Confirm cache | Val cache | V2 cache | Latency rows | Exported schedules |\n|---|---:|---:|---:|---:|---:|---:|\n'
        for b,row in audit.items():
            c=row['caches_validated'];text+=f"| {b} | {c['half']} | {c['confirm']} | {c['val']} | {c['v2']} | {row['latency_rows']} | {row['schedules_exported']} |\n"
    report.write_text(text)
    print(json.dumps(result,indent=1))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default=ROOT);parser.add_argument('--baseline-root',default=ORIGINAL)
    parser.add_argument('--gate-root',default=GATE);parser.add_argument('--data-cache',default=str(Path('data/cache').resolve()))
    finish(parser.parse_args())
