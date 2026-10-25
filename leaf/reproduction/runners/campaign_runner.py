"""Isolated evaluation dispatch, checkpoint-locked teachers, batch32 caches."""
from pathlib import Path
import gc
import shutil
import numpy as np
import torch
from leaf.evaluation.runner import Runner,build as old_build,load_json,save,config_hash
from leaf.reproduction.runners.merge_variant_runner import MergeVariantRunner
from leaf.evaluation.data import load_split,SUBSETS,iterate
from leaf.latency.timing import inference_copy
from baselines.tome_m3 import ToMeM3

def build(vit,spec,cfg):
    if cfg['kind']=='tome_m3':
        r=tuple(cfg['r']) if isinstance(cfg['r'],list) else cfg['r']
        return ToMeM3(vit,spec,r)
    return old_build(vit,spec,cfg)

class CampaignRunner(MergeVariantRunner):
    def __init__(self,backbone,root,design):
        self.campaign=Path(root); self.design=design; self.binding=design['backbones'][backbone]
        super().__init__(backbone,self.campaign/backbone,32,design['data_cache'],self.campaign/'teachers',self.binding['timm_id'])
        self.input_root=Path(self.binding['root']); self.reuse_roots=[self.input_root]
        if backbone in ('deit_b','mae_b'):
            self.reuse_roots.append(Path(design['merge_root'])/backbone)
    def inference_model(self,cfg):
        _=self.vit
        if getattr(self,'_cast',None) is None: self._cast=inference_copy(self.vit)
        model=build(self.vit,self.spec,cfg)
        if cfg['kind']=='dense': return self._cast
        model.vit=self._cast
        for mod in model.modules():
            if mod is not self._cast and hasattr(mod,'prepare_inference'): mod.prepare_inference(torch.bfloat16)
        return model.eval()
    def data(self,split):
        if split not in ('half','odd','calib','confirm','val','v2'): raise ValueError(split)
        if split not in self._data:
            _=self.vit
            base,sl=SUBSETS.get(split,(split,slice(None)))
            images,labels=load_split(base,self.data_cache)
            path=self.teacher_cache/f'{base}_{self.name}_bs32_logits.npy'
            if not path.exists(): raise RuntimeError(f'missing locked batch32 teacher: {path}')
            teacher=np.load(path,mmap_mode='r')
            assert len(teacher)==len(labels)
            self._data[split]=images[sl],labels[sl],torch.from_numpy(np.array(teacher[sl],dtype=np.float32))
        return self._data[split]
    def teachers(self):
        _=self.vit; self.teacher_cache.mkdir(parents=True,exist_ok=True)
        for split in ('calib_bal','calib','confirm','val','v2'):
            dest=self.teacher_cache/f'{split}_{self.name}_bs32_logits.npy'
            if dest.exists(): continue
            old=Path(self.design['original_root'])/'teachers'/dest.name
            if old.exists():
                shutil.copy2(old,dest); save(dest.with_suffix('.json'),dict(reused_from=str(old),batch=32)); continue
            images,labels=load_split(split,self.data_cache)
            tmp=dest.with_suffix('.tmp.npy')
            mm=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.float16,shape=(len(labels),1000))
            offset=0
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                for x,_ in iterate(images,labels,32,'cuda',self.spec.mean,self.spec.std):
                    mm[offset:offset+len(x)]=self.vit(x).float().half().cpu().numpy(); offset+=len(x)
            mm.flush(); del mm; tmp.replace(dest)
            save(dest.with_suffix('.json'),dict(batch=32,checkpoint=self.binding['timm_id'],n=len(labels)))
    def run(self,cfg,split,meta=None):
        p=self.path(cfg,split)
        if not p.with_suffix('.json').exists():
            expected=dict(half=5000,odd=5000,confirm=20018,val=50000,v2=10000)[split]
            for base in self.reuse_roots:
                old=base/split/'runs'/config_hash(cfg); record=old.with_suffix('.json')
                if not record.exists() or not old.with_suffix('.npz').exists(): continue
                row=load_json(record)
                if row.get('config')!=cfg or row.get('batch_size')!=32 or row.get('n')!=expected: continue
                # Historical roots/teacher provenance are SHA bound by design before launch.
                arr=dict(np.load(old.with_suffix('.npz')))
                if any(arr[k].shape!=(expected,) for k in ('kl','pred','correct')) or not np.isfinite(arr['kl']).all(): continue
                p.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(old.with_suffix('.npz'),p.with_suffix('.npz'))
                save(p.with_suffix('.json'),dict(row,reused_from=str(old),campaign_sha=self.design['sha256'])); break
        return Runner.run(self,cfg,split,meta=dict(meta or {},campaign_sha=self.design['sha256']))
