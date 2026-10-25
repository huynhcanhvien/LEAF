"""CPU-only final exports and interpretation for the completed t/M3 pilot."""
import csv
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import numpy as np
from leaf.reproduction.experiments import salt_m3_pilot as p


def main():
    p.validate(); doc=p.load_json(p.ROOT/'results.json'); assert doc['complete']
    rows=doc['records']; contrasts=[]; errors=[]; profile_rows=[]
    for arm in rows:
        if not arm['arm'].startswith('t_M3/'):continue
        bb=arm['backbone']; bid=arm['arm'].split('/')[1]
        ref=next(v for v in rows if v['backbone']==bb and v['arm']=='paper/latency/'+bid)
        samples=lambda name:np.array([p.load_json(p.ROOT/bb/'latency'/f"{name.replace('/','|')}#{rnd}.json")['measurement']['ms_mean'] for rnd in range(3)])
        a,b=samples(arm['arm']),samples(ref['arm'])
        tau=max(.05,2*float(np.sqrt((a.var(ddof=1)+b.var(ddof=1))/2)))
        err=100*(arm['predicted_ms']/arm['latency_ms']-1);errors.append(abs(err))
        comparison=dict(backbone=bb,budget=bid,predicted_ms=arm['predicted_ms'],actual_ms=arm['latency_ms'],
            paper_ms=ref['latency_ms'],delta_latency_ms=arm['latency_ms']-ref['latency_ms'],
            close_latency=bool(abs(arm['latency_ms']-ref['latency_ms'])<=tau),tolerance_ms=tau,
            valid=arm['valid'],paper_valid=ref['valid'],delta_valid_pp=arm['valid']-ref['valid'],
            v2=arm['v2'],paper_v2=ref['v2'],delta_v2_pp=arm['v2']-ref['v2'],prediction_error_pct=err)
        for split in ('val','v2'):
            test=next(v for v in doc['paired_tests'] if v['backbone']==bb and v['arm']==arm['arm'] and v['reference']==ref['arm'] and v['split']==split)
            comparison[f'{split}_holm']=test['holm']
        contrasts.append(comparison)
        plan=p.load_json(p.ROOT/bb/'plans.json')['jobs'][arm['arm']]
        profile_rows.append(dict(backbone=bb,budget=bid,search_seconds=plan['search_seconds'],
            cached_accuracy=bool(p.load_json(p.ROOT/bb/'val/runs'/f"{p.config_hash(plan['config'])}.json").get('reused_from'))))
    for name,records in [('all_arms.csv',rows),('paper_comparisons.csv',contrasts),('paired_tests.csv',doc['paired_tests'])]:
        with (p.ROOT/name).open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    p.save(p.ROOT/'summary.json',dict(mape_pct=float(np.mean(errors)),max_absolute_error_pct=max(errors),
        comparisons=contrasts,search=profile_rows,notes='Tolerance is descriptive; same-budget pairs fixed before accuracy. Holm spans all 18 pilot contrasts per split. No claim of universal optimality or cross-device validity.'))
    lines=['','## Evaluation after completion','',
        f'Mean absolute relative error of t over 9 schedules: **{np.mean(errors):.2f}%**, maximum **{max(errors):.2f}%**. Tested only on 3 backbones/3 budgets; the ranking among schedules with nearly the same GMAC has not been verified by this small test.',
        '', '| Backbone | Budget | t / measured ms | Paper M0 ms | Δ valid (pp) | Δ V2 (pp) | Close in latency |',
        '|---|---|---:|---:|---:|---:|---|']
    for v in contrasts:
        lines.append(f"| {v['backbone']} | {v['budget']} | {v['predicted_ms']:.3f} / {v['actual_ms']:.3f} | {v['paper_ms']:.3f} | {v['delta_valid_pp']:+.3f} | {v['delta_v2_pp']:+.3f} | {'Yes' if v['close_latency'] else 'No'} |")
    lines+=['',
        '“Close in latency”: mean difference ≤ max(0.05 ms, 2×pooled sample SD) over three timing rounds. This is a closeness check; it does not prove statistical equivalence. Pairs that differ in latency are not used to claim iso-latency wins.',
        '', 'Versus paper M0: valid accuracy is numerically higher at 9/9 points; after Holm over all 18 contrasts per split, AugReg-S dense0.5 and MAE-B DR8.7 have significant valid gains. Only AugReg-S dense0.5 has a significant V2 gain. No point in the pilot is significantly worse than paper M0. The other points do not give enough evidence of an accuracy gain; V2 has many small negative deltas.',
        '', 'AugReg-S dense0.5: valid +1.082 points, V2 +1.250 points; latency 4.759 vs 4.738 ms. MAE-B DR8.7: valid +0.246 points, V2 +0.190 points (V2 not significant); latency 17.557 vs 17.520 ms.',
        '', f"After the probe is loaded, DP takes {min(v['search_seconds'] for v in profile_rows):.3f}–{max(v['search_seconds'] for v in profile_rows):.3f} seconds/schedule. {sum(v['cached_accuracy'] for v in profile_rows)}/9 t schedules reuse historical accuracy for the exact same config. The cost of building the probe and of the original fit/profile is not included in this search time.",
        '', 'The current model uses the number of stations and the number of active D steps; it has no varying D term or extended station features yet; rank stays at 32. M3 is the MLERP operator; the whole system is still the DP of this project. Old code was not modified. Accuracy runs were split across the two GPUs by backbone; workers that finished early stopped. Latency ran on GPU 0 while GPU 1 was idle, after all accuracy runs.',
        '', 'Artifacts: `all_arms.csv`, `paper_comparisons.csv`, `paired_tests.csv`, `summary.json`, `audit.json`. The audit checks 81 named arrays, config/batch/n/finite, and the SHA of old code/input/manifest. Not independently reviewed.']
    for repo in {Path.cwd(),p.MAIN}:
        report=repo/'results/reports'/p.REPORT
        original=report.read_text().split('\n## Evaluation after completion')[0]
        report.write_text(original+'\n'.join(lines)+'\n')
    print('SUMMARY',float(np.mean(errors)),max(errors),profile_rows)


if __name__=='__main__':main()
