"""Data/cache and locked inference protocol for section 7.2 (M3/N4 only)."""
from pathlib import Path
import numpy as np
import torch

from leaf.evaluation.runner import Runner, load_json
from leaf.reproduction.runners.merge_variant_runner import MergeVariantRunner, cached_arrays
from leaf.evaluation.data import load_split, SUBSETS, mcnemar, holm


class NormCampaignRunner(MergeVariantRunner):
    def __init__(self, backbone, root, original_root, data_cache, teacher_cache):
        super().__init__(backbone, root, 32, data_cache, teacher_cache)
        self.original_root = Path(original_root)
        self.stage_sha = None

    def data(self, split):
        if split not in ('half','confirm','val','v2'):
            raise ValueError('unknown campaign split')
        if split not in self._data:
            base, sl=SUBSETS.get(split,(split,slice(None)))
            im,lb=load_split(base,self.data_cache)
            teacher=np.load(self.teacher_cache/f'{base}_{self.name}_bs32_logits.npy',mmap_mode='r')
            if teacher.shape[0]!=len(lb): raise RuntimeError('teacher count mismatch')
            self._data[split]=im[sl],lb[sl],torch.from_numpy(np.array(teacher[sl],dtype=np.float32))
        return self._data[split]

    def cached(self, cfg, split):
        if cfg['kind']!='pdm':
            # Baseline cache is strictly read-only. Never recompute historical val/V2.
            counts={'half':5000,'confirm':20018,'val':50000,'v2':10000}
            try:
                return cached_original(self.original_root,cfg,split,counts[split])
            except FileNotFoundError:return None
        return super().cached(cfg,split)

    def run(self,cfg,split,meta=None):
        if cfg['kind']!='pdm':
            arr=self.cached(cfg,split)
            if arr is None:raise RuntimeError('missing historical baseline cache; refusing to rerun')
            return arr
        if split in ('val','v2'):
            from leaf.reproduction.runners.sealed_stage_protocol import Stage
            stage=Stage(self,'stage2')
            frozen=stage.sealed()
            pairs=load_json(stage.dir/'pairs.json')
            if pairs['sha256']!=frozen['sha256'] or cfg not in [j['config'] for j in frozen['jobs'].values()]:
                raise RuntimeError('heldout evaluation requires frozen manifest and matching pairs')
        return super().run(cfg,split,meta=meta)


def cached_original(root,cfg,split,n):
    # root is exactly the original backbone/bs32 cache directory.
    root=Path(root)
    return cached_arrays(root.parent,root.name,cfg,split,n,batch=32) if split in ('half','val') else _read_other(root,cfg,split,n)


def _read_other(root,cfg,split,n):
    from leaf.evaluation.runner import config_hash
    p=root/split/'runs'/config_hash(cfg)
    meta=load_json(p.with_suffix('.json'))
    if meta.get('config')!=cfg or meta.get('batch_size')!=32 or meta.get('n')!=n:raise RuntimeError('baseline metadata mismatch')
    with np.load(p.with_suffix('.npz')) as z:arr={k:z[k] for k in ('correct','pred','kl')}
    if any(v.shape!=(n,) for v in arr.values()) or not np.isfinite(arr['kl']).all():raise RuntimeError('baseline arrays invalid')
    return arr


def latency_rounds(rows,name):
    if set(rows.get(name,{}))!={0,1,2}:raise RuntimeError('incomplete three-round session')
    return np.array([rows[name][i]['32'] for i in range(3)],dtype=float)


def pair_design(jobs,rows):
    dense=float(latency_rounds(rows,'dense').mean())
    sd=max(float(latency_rounds(rows,n).std(ddof=1)) for n in jobs)
    tau=max(2*sd,.05*min(1.,dense/15.))
    pairs=[]
    for name,job in sorted(jobs.items()):
        if not name.startswith(('M3/','N4/')):continue
        ms=float(latency_rounds(rows,name).mean())
        for reference,prefix in (('LEAF-M0','M0/'),('DiffRate','DR')):
            refs={n:float(latency_rounds(rows,n).mean()) for n in jobs if n.startswith(prefix)}
            eligible=[n for n in refs if refs[n]>=ms-tau-1e-12]
            ref=min(eligible,key=lambda n:(refs[n],n)) if eligible else max(refs,key=lambda n:(refs[n],n))
            pairs.append(dict(arm=name,ref=ref,reference=reference,method=name.split('/')[0],
                eligible=bool(eligible),ms=ms,ref_ms=refs[ref],tau_ms=tau))
    return pairs,dict(tau_ms=tau,sd_ms=sd,floor_ms=.05*min(1.,dense/15.),batch=32)


def paired_tests(runner,jobs,rows,pairs,split):
    result=[]
    for pair in pairs:
        a=runner.cached(jobs[pair['arm']]['config'],split);b=runner.cached(jobs[pair['ref']]['config'],split)
        if a is None or b is None:raise RuntimeError('missing comparison arrays')
        p,wins,losses=mcnemar(a['correct'].astype(bool),b['correct'].astype(bool))
        arm=latency_rounds(rows,pair['arm']);ref=latency_rounds(rows,pair['ref'])
        result.append(dict(pair,split=split,delta=100*float(a['correct'].mean()-b['correct'].mean()),
            kl=float(a['kl'].mean()),ref_kl=float(b['kl'].mean()),p=p,wins=wins,losses=losses,
            same_or_faster=bool((arm<=ref+pair['tau_ms']).all())))
    # Both reference families and latency/latency+GMAC budgets share this family.
    for method in ('M3','N4'):
        holm([row for row in result if row['method']==method])
    return result
