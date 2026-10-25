"""Final CPU/GPU interpretation, coefficient exports, and taxonomy update."""
import csv
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import numpy as np
from leaf.reproduction.experiments import salt_fit as cpu
from leaf.reproduction.experiments import salt_formulas as gpu
from leaf.latency.salt import FORMULAS
from leaf.evaluation.runner import load_json,save


def export(name,rows):
    with (cpu.ROOT/name).open('w') as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    assert load_json(gpu.ROOT/'results.json')['complete'];gpu.validate()
    data=load_json(cpu.ROOT/'dataset.json');comparison=load_json(cpu.ROOT/'comparison.json');forward=load_json(cpu.ROOT/'forward.json')
    strict_path=cpu.ROOT/'workload_cv/comparison.json'
    strict_comparison=load_json(strict_path) if strict_path.exists() else comparison
    result=load_json(gpu.ROOT/'results.json');records=result['records'];contrasts=[];summary=[];coefficients=[]
    for bb in cpu.SPECS:
        for formula in FORMULAS:
            model_path=cpu.ROOT/'workload_cv'/bb/f'{formula}.json'
            if not model_path.exists():model_path=cpu.ROOT/'cv'/bb/f'{formula}.json'
            model=load_json(model_path)['full_model']
            for feature,value in [('intercept',model['intercept'])]+list(zip(model['feature_names'],model['coef'])):
                coefficients.append(dict(backbone=bb,formula=formula,feature=feature,coefficient=value,alpha=model['alpha'],fit_source=str(model_path)))
    for a in records:
        if a['formula'] not in FORMULAS:continue
        target=load_json(gpu.ROOT/a['backbone']/'plans.json')['jobs'][a['name']]['budget']['latency_ms']
        for ref in ('frozen_t','gmac_M3','paper_M0'):
            b=next(v for v in records if v['backbone']==a['backbone'] and v['budget']==a['budget'] and v['formula']==ref)
            tau=max(.05,2*np.sqrt((a['latency_sd']**2+b['latency_sd']**2)/2))
            c=dict(backbone=a['backbone'],formula=a['formula'],budget=a['budget'],reference=ref,
                actual_ms=a['latency_ms'],reference_ms=b['latency_ms'],delta_ms=a['latency_ms']-b['latency_ms'],tolerance_ms=float(tau),
                eligible=bool(abs(a['latency_ms']-b['latency_ms'])<=tau),valid=a['valid'],v2=a['v2'],
                delta_valid_pp=a['valid']-b['valid'],delta_v2_pp=a['v2']-b['v2'],
                predicted_ms=a['predicted_ms'],absolute_prediction_error_pct=100*abs(a['predicted_ms']/a['latency_ms']-1))
            c.update(budget_ms=target,budget_pass=bool(a['latency_ms']<=target),budget_overshoot_ms=max(0.,a['latency_ms']-target))
            for split in ('val','v2'):
                t=next(t for t in result['paired_tests'] if t['backbone']==a['backbone'] and t['name']==a['name'] and t['reference']==ref+'/'+a['budget'] and t['split']==split)
                c[split+'_holm']=t['holm']
            contrasts.append(c)
    for formula in FORMULAS:
        cases=[v for v in contrasts if v['formula']==formula and v['reference']=='frozen_t']
        matched=[v for v in cases if v['eligible']]
        summary.append(dict(formula=formula,n=len(cases),new_schedule_mape=float(np.mean([v['absolute_prediction_error_pct'] for v in cases])),
            mean_valid_same_budget=float(np.mean([v['delta_valid_pp'] for v in cases])),mean_v2_same_budget=float(np.mean([v['delta_v2_pp'] for v in cases])),
            budget_pass_n=sum(v['budget_pass'] for v in cases),max_budget_overshoot_ms=max(v['budget_overshoot_ms'] for v in cases),
            matched_n=len(matched),matched_mean_valid=float(np.mean([v['delta_valid_pp'] for v in matched])) if matched else None,
            matched_mean_v2=float(np.mean([v['delta_v2_pp'] for v in matched])) if matched else None,
            matched_val_wins=sum(v['val_holm']<.05 and v['delta_valid_pp']>0 for v in matched),
            matched_val_losses=sum(v['val_holm']<.05 and v['delta_valid_pp']<0 for v in matched),
            matched_v2_wins=sum(v['v2_holm']<.05 and v['delta_v2_pp']>0 for v in matched),
            matched_v2_losses=sum(v['v2_holm']<.05 and v['delta_v2_pp']<0 for v in matched)))
    export('coefficients.csv',coefficients);export('gpu_comparisons.csv',contrasts);export('gpu_summary.csv',summary)
    save(cpu.ROOT/'final_summary.json',dict(cpu=comparison,cpu_workload_grouped=strict_comparison,gpu=summary,contrast_count=len(contrasts),disclosure='Matched subset descriptive; Holm over all 162 planned comparisons per split. No accuracy selection.'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=list(FORMULAS);x=np.arange(len(names));width=.25
    fig,axes=plt.subplots(1,2,figsize=(16,5.5))
    series=[('Workload-grouped nested CV',[next(v['cv_mape'] for v in strict_comparison if v['formula']==n) for n in names]),
        ('Six pilot configurations',[np.mean([v['error_pct'] for v in forward if v['formula']==n]) for n in names]),
        ('New DP schedules',[next(v['new_schedule_mape'] for v in summary if v['formula']==n) for n in names])]
    for i,(label,values) in enumerate(series):axes[0].bar(x+(i-1)*width,values,width,label=label)
    axes[0].set_ylabel('MAPE (%)');axes[0].legend(fontsize=8);axes[0].set_title('Prediction error across evaluation settings')
    accuracy=[100*next(v['near1_accuracy'] for v in comparison if v['formula']==n) for n in names]
    axes[1].bar(x,accuracy);axes[1].set_ylim(0,105);axes[1].set_ylabel('Correct pair order (%)')
    axes[1].set_title('M3 within 1% GMAC: 29 unique pairs, two CV repeats')
    for ax in axes:ax.set_xticks(x,names,rotation=45,ha='right');ax.grid(axis='y',alpha=.2)
    fig.tight_layout();plotdir=cpu.ROOT/'plots';plotdir.mkdir(exist_ok=True)
    for ext in ('png','svg'):fig.savefig(plotdir/f'proxy-summary.{ext}',dpi=160)
    plt.close(fig)
    lines=['<!-- final-analysis-start -->','## Combined evaluation: CPU and GPU experiments','',
        'General formula: t(S) = b + Σ_l wᵀφ(l,n_in,p,m,D,merge). All slopes are non-negative; the intercept is free. φ for split_mac contains 4nd², 2n²d, 8md², (p−m)md, 2dG_pad·rank, nG_pad·d and the station/D/M3 indicators. split_station adds n·log2(n), n·d, (p−m)·d and the M3 I/O norm; tile32/64 add d²·(ceil(n/q)q−n) for attention/MLP. Per-backbone coefficients are in `coefficients.csv`; group/rank are mostly fixed, so individual coefficients should not be read as universal hardware laws.',
        '', '| Formula | MAPE on 6 new pilots % | MAPE on new GPU schedules % | Mean latency ≤ budget | Pairs close in latency to old t | Δ valid on close pairs | Δ V2 on close pairs | Valid wins/losses (Holm) | V2 wins/losses (Holm) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for v in summary:
        f=[r['error_pct'] for r in forward if r['formula']==v['formula']]
        fmt=lambda x:'—' if x is None else f'{x:+.3f}'
        lines.append(f"| {v['formula']} | {np.mean(f):.3f} | {v['new_schedule_mape']:.3f} | {v['budget_pass_n']}/6 | {v['matched_n']}/6 | {fmt(v['matched_mean_valid'])} | {fmt(v['matched_mean_v2'])} | {v['matched_val_wins']}/{v['matched_val_losses']} | {v['matched_v2_wins']}/{v['matched_v2_losses']} |")
    lines+=['',
        'The 6-pilot MAPE tests structures absent from historical training; the pilot latencies had already been seen, so it is still a retrospective check. The new-GPU-schedule MAPE uses the schedules that each model itself just found, without refitting coefficients on the new timings. Δ on the close-latency subset is not an average over all six budgets and cannot be compared directly between models with different coverage.',
        '', 'A pair is close in latency if |Δmean| ≤ max(0.05 ms, 2×pooled SD); three timing rounds in the same session/backbone. This is a closeness criterion; it does not prove statistical equivalence. Holm runs over all 162 contrasts per split, including comparisons with frozen_t, GMAC-M3 and paper M0. Aliases are not independent repeats; no other GPU type or other rank has been tested.',
        '', 'Stricter grouping check: we found two AugReg-B clusters with the same caps/D workload but different κ, so the original config-based CV could still place them in different folds. We grouped this workload and reran all 9 formulas on AugReg-B, keeping the old results in cv/. New macro results: current_refit **0.745%**, split_tile64 **0.414%** (original config-grouped version: 0.741% and 0.416%). Raw data in workload_cv/. The three GPU backbones have no such grouping conflict, so the GPU fits/schedules are not affected.',
        '', 'Forward test on six pilots: split_tile64 MAPE 0.946% sits below current_refit at 1.300%. Ranking of M3 within GMAC1%: 53/58 pair instances (91.4%) vs current_refit 57/58 (98.3%); only 29 unique pairs, repeated over two seeds, which is not enough to claim that a model with more features is always better for search.',
        '', 'All GPU schedules are kept; confirm/valid were not used for selection. Per-configuration tables and the negative cases are in `gpu_comparisons.csv`; OOF/fold predictions and alpha are in `cv/`. Old code is unchanged; the extension is in `leaf/latency/salt.py`. The model used in the paper has not been replaced automatically.',
        '', 'The CPU audit checks 945 folds: no train/test structure overlap, the new-structure pilots are fully separate from historical data, slopes are non-negative, and no OOF prediction is non-positive. The data keep the historical protocol, including two images shared between calibration and confirm per the earlier audit; valid/V2 were already seen in earlier campaigns.',
        '', '### Experimental conclusion',
        '', '**Next, prioritize testing split_tile32 and station_rich; do not replace the current default t yet.** On six schedules chosen by each model itself, split_tile32 has MAPE 0.451% and station_rich 0.675%, vs current_refit 0.779%; both new models have mean latency ≤ budget at 6/6 points in this session, current_refit at 5/6. These are six small points on one hardware setup, not a budget guarantee for every run. Notably, split_tile64 wins the historical CV, but on the new schedules it has MAPE 0.741% and keeps the budget at 5/6 points, no better than tile32.',
        '', 'The position formula split_position has severe outliers on DeiT-S: DR2.3 predicts 6.497 ms but measures 7.724 ms; DR2.7 predicts 7.459 ms but measures 8.964 ms. DP picks schedules with only 2 and 1 stations and final token counts of 100 and 127, while the fit data have 3–11 stations and final token counts in {3,4,6}. The features are outside the fit region; the result is consistent with the search exploiting extrapolation error. Additivity of the DP cost and the full-config prediction was asserted while searching each schedule. Raw data in extrapolation_diagnostics.json. The higher accuracy of these slow schedules is not used to claim an iso-latency improvement.',
        '', 'There is one accuracy signal where latency is truly close: AugReg-S dense0.5, split_station reaches valid 69.412% vs old t 68.454%, latency 4.816 vs 4.790 ms; +0.958 points valid passes Holm, V2 +0.640 points does not pass Holm. For split_tile32, the mean delta over all six budget labels is +0.210 points valid and +0.405 points V2, but only 2/6 pairs are close enough in latency and there is no significant V2 win after Holm. This does not support the conclusion that tile32 improves accuracy at every speed level.',
        '', '![Comparison of prediction error and near-GMAC ranking](../results/t_proxy_comparison/2026-10-17/plots/proxy-summary.png)',
        '', 'Artifacts: `comparison.csv`, `forward.csv`, `coefficients.csv`, `gpu_summary.csv`, `gpu_comparisons.csv`, `final_summary.json`; raw data and per-image audit in `gpu/`.','<!-- final-analysis-end -->']
    text=cpu.REPORT.read_text().split('<!-- final-analysis-start -->')[0].rstrip();cpu.REPORT.write_text(text+'\n\n'+'\n'.join(lines)+'\n')
    tax=cpu.REPORT.parent/'REPORT_TAXONOMY.md';text=tax.read_text();prefix=f'| [{cpu.REPORT.name}]'
    replacement=f'| [{cpu.REPORT.name}]({cpu.REPORT.name}) | 9 t formulas on 248 configs/7 backbones; after grouping by workload, split_tile64 reduces CV MAPE 0.745→0.414%, but near-GMAC ranking is not better than the simple form. Ran 54 schedules/3 backbones, confirm/valid/V2 on two GPUs, and final latency; see matched coverage and negative results in the report. Rank32/historical data. | CPU/GPU and audit complete; not independently reviewed |'
    tax.write_text('\n'.join(replacement if line.startswith(prefix) else line for line in text.splitlines())+'\n')
    print(json_dump(summary))


def json_dump(value):
    import json
    return json.dumps(value,indent=1)


if __name__=='__main__':main()
