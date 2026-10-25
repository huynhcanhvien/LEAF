"""Locked evaluation protocol of one stage (a set of arms compared together).

    arms.json   {name: {config, state, meta}}            written by the stage's search
    confirm     every arm on confirm (20k train images)
    freeze      frozen.json = arms + sha256 over {name: config}; never rewritten
    lat         interleaved latency rounds (bs 1/32/128, CUDA graph, GPU alone), reversed order on odd rounds
    pairs       pairs.json locked before valid: arm vs reference (rule given by the stage)
    valid       val 50k; v2 (ImageNet-V2) once, only for manifests that already have locked pairs
    tests       McNemar on top-1, Holm within the family, paired KL z; dominance = delta > 0, Holm p < .05,
                faster in every latency round
"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from leaf.evaluation.data import holm, mcnemar, paired_z
from leaf.evaluation.runner import load_json, save


def digest(jobs):
    return hashlib.sha256(json.dumps({n: v['config'] for n, v in jobs.items()}, sort_keys=True).encode()).hexdigest()


class Stage:
    def __init__(self, runner, name):
        self.r = runner; self.dir = runner.root / name; self.name = name

    def arms(self): return load_json(self.dir / 'arms.json')

    def sealed(self):
        f = load_json(self.dir / 'frozen.json'); assert digest(f['jobs']) == f['sha256'], self.dir
        return f

    def confirm(self, shard=(0, 1)):
        for i, (n, v) in enumerate(sorted(self.arms().items())):
            if i % shard[1] == shard[0]: self.r.run(v['config'], 'confirm', meta=dict(arm=f'{self.name}/{n}'))

    def freeze(self):
        p = self.dir / 'frozen.json'
        if p.exists(): print('already frozen'); return load_json(p)['sha256']
        js = self.arms(); save(p, dict(sha256=digest(js), time=time.strftime('%F %T'), jobs=js)); return digest(js)

    def latency(self, rounds=3, batches=(1, 32, 128), repeats=3, big=(20, 5), guard=None):
        from leaf.latency.timing import measure_latency_batches
        from leaf.evaluation.runner import build, require_idle_gpu
        f = self.sealed(); out = self.dir / 'lat'; names = sorted(f['jobs'])
        for r in range(rounds):
            for n in (names if r % 2 == 0 else names[::-1]):
                path = out / f"{n.replace('/', '|')}#{r}.json"
                if path.exists(): continue
                require_idle_gpu(idle=True)
                if guard is not None: guard()
                model = build(self.r.vit, self.r.spec, f['jobs'][n]['config'])
                row = dict(name=n, round=r, config=f['jobs'][n]['config'], latency=measure_latency_batches(model, batches, repeats=repeats, big=big),
                           time=time.strftime('%F %T'), frozen_sha=f['sha256'])
                save(path, row); del model
                print(json.dumps(dict(name=n, round=r, ms={b: (v if v == 'OOM' else round(v['ms_mean'], 3)) for b, v in row['latency'].items()})), flush=True)

    def lat_table(self):
        rows = {}
        for p in (self.dir / 'lat').glob('*.json'):
            r = load_json(p); rows.setdefault(r['name'], {})[r['round']] = {b: v['ms_mean'] for b, v in r['latency'].items() if v != 'OOM'}
        return rows

    @staticmethod
    def ms(rows, n, b='32'):
        return np.array([rows[n][r].get(b, np.nan) for r in sorted(rows[n])]) if n in rows else np.array([np.nan])

    def lock_pairs(self, pairs, rule):
        p = self.dir / 'pairs.json'
        if p.exists(): print('already locked'); return load_json(p)
        out = dict(sha256=self.sealed()['sha256'], time=time.strftime('%F %T'), pairs=pairs, rule=rule); save(p, out); return out

    def pairs_to_slower(self, refs, exclude=()):
        """Each arm vs the fastest reference whose mean bs32 latency is >= its own (none slower: not eligible)."""
        f = self.sealed(); rows = self.lat_table(); ref_ms = {n: self.ms(rows, n).mean() for n in refs}; out = []
        for n in sorted(f['jobs']):
            if n in refs or n in exclude: continue
            m = self.ms(rows, n).mean(); ref = min((d for d in ref_ms if ref_ms[d] >= m), key=lambda d: ref_ms[d], default=None)
            out.append(dict(arm=n, ref=ref or max(ref_ms, key=ref_ms.get), eligible=ref is not None, ms=float(m),
                            ref_ms=float(ref_ms[ref or max(ref_ms, key=ref_ms.get)])))
        return out

    def valid(self, split='val', shard=(0, 1)):
        assert (self.dir / 'pairs.json').exists(), 'pairs must be locked before valid'
        for i, (n, v) in enumerate(sorted(self.sealed()['jobs'].items())):
            if i % shard[1] == shard[0]: self.r.run(v['config'], split, meta=dict(arm=f'{self.name}/{n}'))

    def tests(self, split='val'):
        f = self.sealed(); pr = load_json(self.dir / 'pairs.json'); rows = self.lat_table(); out = []
        for q in pr['pairs']:
            a, b = self.r.cached(f['jobs'][q['arm']]['config'], split), self.r.cached(f['jobs'][q['ref']]['config'], split)
            if a is None or b is None: continue
            ca, cb = self.r.cached(f['jobs'][q['arm']]['config'], 'confirm'), self.r.cached(f['jobs'][q['ref']]['config'], 'confirm')
            d32 = self.ms(rows, q['arm']) - self.ms(rows, q['ref'])
            out.append(dict(q, d=100 * (a['correct'].mean() - b['correct'].mean()), p=mcnemar(a['correct'].astype(bool), b['correct'].astype(bool))[0],
                            dkl=100 * (a['kl'].mean() / b['kl'].mean() - 1), zv=paired_z(a['kl'], b['kl']),
                            zc=paired_z(ca['kl'], cb['kl']) if ca is not None and cb is not None else np.nan, d32=d32,
                            faster=int((d32 < 0).sum()), rounds=len(d32)))
        for fam in {t.get('family') for t in out}:          # Holm within each comparison family
            holm([t for t in out if t.get('eligible', True) and t.get('family') == fam])
        for t in out: t['dominates'] = bool(t.get('eligible', True) and t['d'] > 0 and t.get('holm', 1) < .05 and (t['d32'] < 0).all())
        return out
