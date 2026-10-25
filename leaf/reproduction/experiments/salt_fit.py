"""CPU-only, grouped nested validation of additive latency formulas. Saves every step."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import numpy as np
from scipy.stats import spearmanr
from leaf.backbones import SPECS
from leaf.latency.salt import FORMULAS,features,matrix,fit,predict
from leaf.planner.gmac import config_cost
from leaf.evaluation.runner import load_json,save,config_hash

SOURCE=Path('results/gmac_planning')
PILOT=Path('results/salt_m3_pilot')
ROOT=Path('results/salt_fit')
REPORT=Path('results/reports/t-proxy-formulas-2026-10-17.md')
ALPHAS=(0.,.01,.1,1.,10.,100.)


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def structural(cfg):return config_hash({k:v for k,v in cfg.items() if k not in ('kind','merge')})
def immutable(path,value):
    value=json.loads(json.dumps(value))
    if path.exists():assert load_json(path)==value,path
    else:save(path,value)


def dataset():
    path=ROOT/'dataset.json'
    if path.exists():return load_json(path)
    inputs={};rows=[];forward=[]
    for bb in SPECS:
        frozen=SOURCE/bb/'frozen.json';inputs[str(frozen.resolve())]=digest(frozen)
        jobs=load_json(frozen)['jobs'];unique={}
        for name,job in jobs.items():
            cfg=job['config']
            if cfg['kind'] not in ('pd','pdm'):continue
            if cfg['kind']=='pdm':assert cfg['merge']['mode']=='M3'
            key=config_hash(cfg)
            if key not in unique:unique[key]=dict(config=cfg,names=[])
            unique[key]['names'].append(name)
        for key,job in unique.items():
            values=[];name=sorted(job['names'])[0]
            for rnd in range(3):
                p=SOURCE/bb/'latency'/f"{name.replace('/','|')}#{rnd}.json"
                row=load_json(p);inputs[str(p.resolve())]=digest(p)
                values.append(row['measurements']['32']['ms_mean'])
            cfg=job['config'];spec=SPECS[bb]
            rows.append(dict(backbone=bb,key=key,group=structural(cfg),config=cfg,names=job['names'],
                mode=cfg['kind'],gmac=config_cost(cfg,spec),ms=float(np.mean(values)),sd=float(np.std(values,ddof=1)),features=features(cfg,spec)))
    for bb in ('deit_s','augreg_s','mae_b'):
        frozen=PILOT/bb/'plans.json';inputs[str(frozen.resolve())]=digest(frozen)
        training={r['group'] for r in rows if r['backbone']==bb}
        for name,job in load_json(frozen)['jobs'].items():
            if not name.startswith('t_M3/'):continue
            cfg=job['config']
            # Strictly exclude any pilot structure present in training, even under M0.
            if structural(cfg) in training:continue
            values=[]
            for rnd in range(3):
                p=PILOT/bb/'latency'/f"{name.replace('/','|')}#{rnd}.json"
                inputs[str(p.resolve())]=digest(p);values.append(load_json(p)['measurement']['ms_mean'])
            forward.append(dict(backbone=bb,key=config_hash(cfg),group=structural(cfg),config=cfg,name=name,
                ms=float(np.mean(values)),sd=float(np.std(values,ddof=1)),features=features(cfg,SPECS[bb])))
    data=dict(rows=rows,forward=forward,inputs=inputs)
    save(path,data);return data


def folds(rows,count=5,seed=151,blocked=False):
    groups=sorted({r['group'] for r in rows});count=min(count,len(groups))
    assert count>=2
    if blocked:
        groups.sort(key=lambda g:np.mean([r['gmac'] for r in rows if r['group']==g]))
        chunks=np.array_split(groups,count);assignment={g:i for i,chunk in enumerate(chunks) for g in chunk}
    else:
        order=np.random.default_rng(seed).permutation(groups)
        assignment={g:i%count for i,g in enumerate(order)}
    out=[]
    for f in range(count):
        te=np.array([assignment[r['group']]==f for r in rows]);tr=~te
        assert {r['group'] for r,t in zip(rows,tr) if t}.isdisjoint({r['group'] for r,t in zip(rows,te) if t})
        out.append((np.where(tr)[0],np.where(te)[0]))
    return out


def tune(rows,name,seed=701):
    x=matrix(rows,name);y=np.array([r['ms'] for r in rows]);splits=folds(rows,3,seed)
    scores=[]
    for alpha in ALPHAS:
        errors=[]
        for tr,te in splits:
            model=fit(x[tr],y[tr],alpha);pred=predict(model,x[te]);errors.extend(abs(pred-y[te])/y[te])
        scores.append((float(np.mean(errors)),alpha))
    _,alpha=min(scores,key=lambda v:(v[0],-v[1]))
    return dict(fit(x,y,alpha),formula=name,feature_names=FORMULAS[name],n=len(rows),
        groups=len({r['group'] for r in rows}),tuning=[dict(alpha=a,mape=100*e) for e,a in scores])


def ranking(rows,pred,tol):
    correct=total=0
    for i,a in enumerate(rows):
        if a['mode']!='pdm':continue
        for j in range(i+1,len(rows)):
            b=rows[j]
            if b['mode']!='pdm' or a['group']==b['group']:continue
            if abs(a['gmac']-b['gmac'])>tol*max(a['gmac'],b['gmac']):continue
            noise=max(.05,2*np.sqrt((a['sd']**2+b['sd']**2)/2))
            if abs(a['ms']-b['ms'])<=noise:continue
            total+=1;correct+=int((pred[i]-pred[j])*(a['ms']-b['ms'])>0)
    return dict(correct=correct,n=total,accuracy=correct/total if total else None)


def metrics(rows,pred):
    y=np.array([r['ms'] for r in rows]);pred=np.asarray(pred);mask=np.array([r['mode']=='pdm' for r in rows])
    return dict(n=len(rows),mape=100*float(np.mean(abs(pred-y)/y)),mae=float(np.mean(abs(pred-y))),
        m3_n=int(mask.sum()),m3_mape=100*float(np.mean(abs(pred[mask]-y[mask])/y[mask])),
        spearman=float(spearmanr(pred,y).statistic),near1=ranking(rows,pred,.01),near5=ranking(rows,pred,.05),
        negative_predictions=int((pred<=0).sum()))


def evaluate(rows,name):
    x=matrix(rows,name);predictions=[];blocked=None
    for protocol,seeds in [('random',(151,173)),('blocked',(151,))]:
        for seed in seeds:
            pred=np.full(len(rows),np.nan);audit=[]
            for f,(tr,te) in enumerate(folds(rows,5,seed,protocol=='blocked')):
                train=[rows[i] for i in tr];model=tune(train,name,701+f)
                pred[te]=predict(model,x[te]);audit.append(dict(fold=f,train=[rows[i]['key'] for i in tr],test=[rows[i]['key'] for i in te],alpha=model['alpha']))
            assert np.isfinite(pred).all()
            result=dict(protocol=protocol,seed=seed,predictions=pred.tolist(),metrics=metrics(rows,pred),folds=audit)
            if protocol=='random':predictions.append(result)
            else:blocked=result
    return dict(random=predictions,blocked=blocked,full_model=tune(rows,name))


def aggregate(data):
    rows=data['rows'];summaries=[]
    for name in FORMULAS:
        files=[ROOT/'cv'/bb/f'{name}.json' for bb in SPECS]
        if not all(p.exists() for p in files):continue
        results=[load_json(p) for p in files];random_metrics=[run['metrics'] for result in results for run in result['random']]
        blocked=[v['blocked']['metrics'] for v in results]
        near=lambda key:dict(correct=sum(v[key]['correct'] for v in random_metrics),n=sum(v[key]['n'] for v in random_metrics))
        a,b=near('near1'),near('near5')
        summaries.append(dict(formula=name,features=len(FORMULAS[name]),
            cv_mape=float(np.mean([v['mape'] for v in random_metrics])),
            cv_m3_mape=float(np.mean([v['m3_mape'] for v in random_metrics])),
            blocked_mape=float(np.mean([v['mape'] for v in blocked])),
            near1_correct=a['correct'],near1_n=a['n'],near1_accuracy=a['correct']/a['n'] if a['n'] else None,
            near5_correct=b['correct'],near5_n=b['n'],near5_accuracy=b['correct']/b['n'] if b['n'] else None))
    save(ROOT/'comparison.json',summaries)
    return summaries


def refresh(data,finished=False):
    summary=aggregate(data)
    lines=['# Comparison of t proxy formulas — 2026-10-17','',
        'Status: '+('CPU done; next, run M3 schedules with the formulas.' if finished else 'CPU running; no conclusion yet.')+' GPU is not used in the formula-comparison stage.',
        '',f"Data: {len(data['rows'])} unique configs / 7 backbones, latency bs32 BF16, three rounds; only PD M0/M3, not mixed with baselines that use different routing. {len(data['forward'])} pilot configs whose structures are absent from the historical data, for an additional check. Accuracy is not used to fit/select formulas.",
        '', 'Structures with kind/merge removed are grouped into the same fold (M0/M3 on the same schedule are not split across train/test). Outer 5-fold × 2 seeds; inner 3-fold selects regularization on train. An extra 5-fold split by GMAC range tests extrapolation. Each backbone is fit separately; all slopes ≥ 0, intercept free. Every formula is additive over blocks/stations and can be used in the DP.',
        '', '| Formula | Features | CV MAPE % | M3 MAPE % | Blocked MAPE % | Correct M3 order within GMAC 1% | Within 5% |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for v in summary:
        a=f"{100*v['near1_accuracy']:.1f}% ({v['near1_correct']}/{v['near1_n']})" if v['near1_n'] else '—'
        b=f"{100*v['near5_accuracy']:.1f}% ({v['near5_correct']}/{v['near5_n']})" if v['near5_n'] else '—'
        lines.append(f"| {v['formula']} | {v['features']} | {v['cv_mape']:.3f} | {v['cv_m3_mape']:.3f} | {v['blocked_mape']:.3f} | {a} | {b} |")
    lines+=['', 'CV MAPE is a macro-average over backbones/repeats. Ranking pairs include only M3 schedules with different structures, GMAC difference ≤1%/5%, and latency difference > max(0.05ms, 2×pooled SD). Pairs repeated over two seeds are not independent; the number of pairs is small. OOF ranking may compare predictions from different folds/models.',
        '', 'Models: gmac_only; current_refit = GMAC + number of stations + number of D steps + M3 stations; station_rich adds station/merge/D/M3 workload; split_mac separates QKV/attention/MLP/similarity/D; split_station adds sort and an I/O proxy; position adds station depth/cut amount; hinge adds token64/128 thresholds; tile32/64 add an assumed padding workload. Tile is an experimental basis; it does not claim that the actual kernel uses exactly that tile.',
        '', 'Limitations: one GPU type/bs32/BF16, rank mostly 32, and D groups padded to 16; transfer to other rank/G/devices is not yet shown. A good MAPE does not guarantee that DP will not exploit errors outside the measured region. A formula that wins CV does not necessarily yield schedules that win on accuracy/latency.',
        '', f'Raw: `{ROOT.resolve()}`. Dataset/input SHA, CV/folds/predictions/full models are saved right after each step. Next on GPU: all 9 formulas × 3 backbones × 2 budgets, confirm→valid/V2, final latency at bs32; this report will be updated.']
    REPORT.write_text('\n'.join(lines)+'\n')
    tax=REPORT.parent/'REPORT_TAXONOMY.md';text=tax.read_text();prefix=f'| [{REPORT.name}]'
    replacement=f'| [{REPORT.name}]({REPORT.name}) | Compares 9 t formulas on 7 backbones with nested CV by structure; checks M3 ranking near GMAC and new pilots. Next: GPU scheduling/confirm/valid/V2, final timing. Rank32/current hardware. | '+('CPU done; GPU next' if finished else 'CPU running')+' |'
    tax.write_text('\n'.join(replacement if line.startswith(prefix) else line for line in text.splitlines())+'\n')


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    immutable(ROOT/'design.json',dict(formulas=FORMULAS,alphas=ALPHAS,seeds=[151,173],folds=5,inner_folds=3,
        script_sha=digest(__file__),formula_sha=digest('leaf/latency/salt.py'),
        future_gpu=dict(backbones=['deit_s','augreg_s','mae_b'],budgets_per_backbone=2,all_formulas=True),
        protocol='group structure excluding operator; nested CV; no accuracy inputs; all trials saved'))
    data=dataset();refresh(data)
    for bb in SPECS:
        rows=[r for r in data['rows'] if r['backbone']==bb]
        for name in FORMULAS:
            dest=ROOT/'cv'/bb/f'{name}.json'
            if dest.exists():continue
            save(ROOT/'status.json',dict(phase='cv',backbone=bb,formula=name,time=time.time()))
            start=time.monotonic();result=evaluate(rows,name);save(dest,dict(result,seconds=time.monotonic()-start))
            print('CV',bb,name,result['random'][0]['metrics']['mape'],flush=True);refresh(data)
    forward=[]
    for bb in SPECS:
        rows=[v for v in data['forward'] if v['backbone']==bb]
        if not rows:continue
        for name in FORMULAS:
            model=load_json(ROOT/'cv'/bb/f'{name}.json')['full_model']
            preds=predict(model,matrix(rows,name))
            for row,pred in zip(rows,preds):forward.append(dict(backbone=bb,name=row['name'],formula=name,ms=row['ms'],predicted_ms=float(pred),error_pct=100*float(abs(pred-row['ms'])/row['ms'])))
    save(ROOT/'forward.json',forward)
    for name,records in [('comparison.csv',aggregate(data)),('forward.csv',forward)]:
        with (ROOT/name).open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    assert all(digest(path)==expected for path,expected in data['inputs'].items())
    save(ROOT/'cpu_done.json',dict(time=time.time(),configs=len(data['rows']),forward=len(data['forward']),inputs_unchanged=True))
    save(ROOT/'status.json',dict(phase='cpu_complete',time=time.time()));refresh(data,True)


if __name__=='__main__':main()
