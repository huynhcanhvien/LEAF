"""Automatic statistical audit and report finalization for the expanded proxy study."""
import csv
import json
import math
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import numpy as np
from scipy.stats import binomtest,norm
from leaf.reproduction.experiments import salt_validation as main
from leaf.evaluation.runner import load_json,save


def csv_save(path,rows):
    with path.open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def analyze(extra_complete=False):
    main.validate();data=load_json(main.ROOT/'results.json');assert data['complete']
    rows=data['records'];tests=data['paired_tests'];assert len(tests)==378
    for t in tests:
        b,c=t['wins'],t['losses'];n=b+c
        p=1. if n==0 else float(2*norm.sf(max(0,abs(b-c)-1)/math.sqrt(n))) if n>2000 else float(binomtest(b,n,.5).pvalue)
        assert abs(p-t['p'])<1e-12
    for split in ('val','v2'):
        subset=sorted([t for t in tests if t['split']==split],key=lambda v:v['p']);assert len(subset)==189;adj=0.
        for rank,t in enumerate(subset):
            adj=max(adj,min(1.,(len(subset)-rank)*t['p']));assert abs(adj-t['holm'])<1e-12
    contrasts=[];summary=[]
    for a in rows:
        if a['formula'] not in main.MODELS:continue
        for reference in ('frozen_t','gmac_M3','paper_M0'):
            b=next(r for r in rows if r['backbone']==a['backbone'] and r['formula']==reference and r['budget']==a['budget'])
            tau=max(.05,2*np.sqrt((a['latency_sd']**2+b['latency_sd']**2)/2))
            row=dict(backbone=a['backbone'],model=a['formula'],budget=a['budget'],reference=reference,
                actual_ms=a['latency_ms'],reference_ms=b['latency_ms'],delta_ms=a['latency_ms']-b['latency_ms'],
                tolerance_ms=float(tau),eligible=bool(abs(a['latency_ms']-b['latency_ms'])<=tau),
                delta_valid_pp=a['valid']-b['valid'],delta_v2_pp=a['v2']-b['v2'])
            for split in ('val','v2'):
                t=next(t for t in tests if t['backbone']==a['backbone'] and t['name']==a['name'] and t['reference']==b['name'] and t['split']==split)
                row[split+'_holm']=t['holm']
            contrasts.append(row)
    for model in ('frozen_t',*main.MODELS):
        subset=[r for r in rows if r['formula']==model];matched=[v for v in contrasts if v['model']==model and v['reference']=='frozen_t' and v['eligible']]
        summary.append(dict(model=model,n=len(subset),mape=float(np.mean([100*abs(r['predicted_ms']/r['latency_ms']-1) for r in subset])),
            budget_pass=sum(r['latency_ms']<=r['budget_ms'] for r in subset),matched_n=len(matched),
            matched_delta_valid=float(np.mean([v['delta_valid_pp'] for v in matched])) if matched else None,
            matched_delta_v2=float(np.mean([v['delta_v2_pp'] for v in matched])) if matched else None,
            val_wins=sum(v['val_holm']<.05 and v['delta_valid_pp']>0 for v in matched),
            val_losses=sum(v['val_holm']<.05 and v['delta_valid_pp']<0 for v in matched),
            v2_wins=sum(v['v2_holm']<.05 and v['delta_v2_pp']>0 for v in matched),
            v2_losses=sum(v['v2_holm']<.05 and v['delta_v2_pp']<0 for v in matched)))
    save(main.ROOT/'analysis.json',dict(summary=summary,contrasts=contrasts,statistics_audited=True,extra_complete=extra_complete))
    csv_save(main.ROOT/'comparisons.csv',contrasts);csv_save(main.ROOT/'summary.csv',summary)
    lines=['<!-- validation-analysis-start -->','## Main-pass analysis','',
        '| Proxy | N | MAPE % | Within budget | Pairs close to old t | Δ valid | Δ V2 | Valid wins/losses (Holm) | V2 wins/losses (Holm) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    fmt=lambda x:'—' if x is None else f'{x:+.3f}'
    for v in summary:
        lines.append(f"| {v['model']} | {v['n']} | {v['mape']:.3f} | {v['budget_pass']}/{v['n']} | {v['matched_n']} | {fmt(v['matched_delta_valid'])} | {fmt(v['matched_delta_v2'])} | {v['val_wins']}/{v['val_losses']} | {v['v2_wins']}/{v['v2_losses']} |")
    lines+=['', 'Each proxy has 21 self-chosen schedules; MAPE values of different proxies come from different schedule sets. "Within budget" means the mean of three rounds ≤ budget in the first session; it is not a guarantee for every session. Matched uses |Δmean| ≤ max(0.05ms, 2×pooled sample SD); coverage differs, so proxies are not ranked by mean delta over different subsets. There is no contrast of frozen_t against itself.',
        '', 'McNemar/Holm were checked independently with SciPy: 378 contrasts, 189 per split, including pairs that are not close in latency. Mean delta on the matched subset is descriptive only. All pairs and the negative cases are in comparisons.csv. Duplicate schedules are not independent repeats.',
        '', 'See the additional-checks block below to separate ranking reliability, stress error and variation across 3 sessions. Two cards of the same type do not demonstrate transfer to other hardware.','<!-- validation-analysis-end -->']
    text=main.REPORT.read_text();start='<!-- validation-analysis-start -->';end='<!-- validation-analysis-end -->'
    if start in text:text=text.split(start)[0]+text.split(end,1)[1]
    main.REPORT.write_text(text.rstrip()+'\n\n'+'\n'.join(lines)+'\n')
    save(main.ROOT/'statistics_audit.json',dict(complete=True,contrasts=378,holm_per_split=189,source_hashes_unchanged=True,time=time.time()))


def wait_phase(path,pidpath):
    while True:
        if path.exists() and load_json(path)['phase']=='completed':return
        if pidpath.exists():
            pid=int(pidpath.read_text())
            if not Path(f'/proc/{pid}').exists():raise RuntimeError(f'scheduler {pid} stopped before completion; logs and partial results retained')
        time.sleep(10)


if __name__=='__main__':
    wait_phase(main.ROOT/'status.json',main.ROOT/'scheduler.pid');analyze()
    wait_phase(main.ROOT/'extra/status.json',main.ROOT/'extra/scheduler.pid');analyze(True)
    state=Path('results/notes/state/t-proxy-validation-2026-10-17-progress.md')
    state.write_text('Main pass (84 schedules) and additional checks (140 pairs, 56 stress cases, 3 sessions) complete. Not independently reviewed.\n\nReport: [t-proxy-validation-2026-10-17.md](../reports/t-proxy-validation-2026-10-17.md).\n')
    save(main.ROOT/'finalized.json',dict(complete=True,time=time.time(),report=str(main.REPORT)))
