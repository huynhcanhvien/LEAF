"""Why the separable distortion estimate suits LEAF: does the score (P + D with alpha) and the merge (MLERP with sizes)
make F-hat more accurate?

Part A (outcome). The 180 sealed random schedules of the surrogate check (ablation_and_alpha_grid.h_schedules, DeiT-B and
MAE-B, 2-8 stations, random blocks, counts and (theta, alpha) from the method's grids) get twins that differ in one
component only, with the same blocks, token counts and merge shares:
  alpha0   every station ranks by present attention alone (alpha = 0)            F-hat from the M3 curves at alpha = 0
  M0       mean merge instead of MLERP (same ranking, assignment and sizes)       F-hat from the M0 curves
  prune    every station prunes (theta = 0), so no merge and no sizes            F-hat from the M3 curves at theta = 0
F-hat of every twin is computed and sealed (twins.json) before its F is measured, on the same 5,000 probe images.
How well F-hat matches F (ratio F/F-hat, |log F/F-hat|, Spearman within a budget) is compared with the drawn schedules.

Part B (mechanism), on 1,000 probe images of the dense model and of single stations:
  ranking  does the score of an early block rank tokens like the later blocks will read them? Spearman of the score
           S(alpha) with the class-token attention of all later blocks (Phi, eq. target), and overlap of the tokens that
           two stations in different blocks would remove (nestedness; chance level = the removed share)
  merge    after one merge-only station, at every later block: norm of each merged token relative to the same token
           in the dense model, and the class-token attention on each cluster relative to the dense attention on its
           members (total variation), for M3, M0 and M3 without sizes

Stages: twins (CPU) | measure --backbones deit_b (GPU) | mech --backbones deit_b (GPU) | report (CPU)
"""
import argparse
import gc
import json
from pathlib import Path
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
import leaf.reproduction.experiments.ablation_and_alpha_grid as A
import leaf.reproduction.experiments.station_discount as SD

ROOT = REPO / 'results/separability_check'
BBS = ('deit_b', 'mae_b')
VARIANTS = ('alpha0', 'M0', 'prune')
ALPHAS = (0., .5, 2 / 3, .75, 1.)
Q = (.25, .5)


def save(p, obj):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix('.tmp'); tmp.write_text(json.dumps(obj, indent=1) + '\n'); tmp.replace(p)


def load(p): return json.loads(Path(p).read_text())


# ----------------------------------------------------------------------------------------------- part A
def twin(c, st, variant):
    """(config, F-hat) of the twin of a drawn schedule state."""
    from leaf.planner.configs import config_from_state
    from leaf.planner.plan import operator_config
    from leaf.planner.alpha import to_alpha_config
    sites, counts, thetas, alphas = st['sites'], st['counts'], list(st['thetas']), list(st['alphas'])
    mode, curves = 'M3', c.base['M3']
    if variant == 'alpha0': alphas = [0.] * len(sites)
    elif variant == 'prune': thetas = [0.] * len(sites)
    elif variant == 'M0': mode, curves = 'M0', c.base['M0']
    cfg = config_from_state(dict(sites=sites, counts=counts, thetas=thetas, ks=alphas), c.spec, c.d)
    cfg = to_alpha_config(operator_config(cfg, mode, True))
    n, fhat = c.spec.n_tokens, 0.
    for l, cnt, th, a in zip(sites, counts, thetas, alphas): fhat += float(curves[(th, a)].transition(l, n, cnt)); n = cnt
    return cfg, fhat


def twins(bbs=BBS):
    from leaf.evaluation.runner import config_hash
    from leaf.reproduction.experiments.gmac_planning import sha
    for bb in bbs:
        dest = ROOT / bb / 'twins.json'
        if dest.exists(): print('sealed', dest); continue
        c = A.Ctx(bb); rows = []
        for r in SD.h_terms(bb):
            row = dict(id=r['id'], budget=r['budget'], K=len(r['st']['sites']), st=r['st'], drawn=dict(F=r['F'], fhat=sum(r['f'])))
            for v in VARIANTS:
                cfg, fhat = twin(c, r['st'], v)
                row[v] = dict(config=cfg, sha=config_hash(cfg), fhat=fhat, same=config_hash(cfg) == config_hash(r['config']))
            rows.append(row)
        save(dest, dict(backbone=bb, rows=rows, sha=sha(rows), time=time.time()))
        print('sealed', bb, len(rows), 'schedules x', len(VARIANTS), 'twins')


def measure(bbs):
    import torch
    from leaf.reproduction.runners.exclusive_timing import guard_foreign
    guard_foreign()
    for bb in bbs:
        doc = load(ROOT / bb / 'twins.json'); r = A.runner(bb); out = ROOT / bb / 'F'
        for i, row in enumerate(doc['rows']):
            for v in VARIANTS:
                t = row[v]; p = out / f"{t['sha']}.json"
                if p.exists(): continue
                if t['same']: save(p, dict(F=row['drawn']['F'], reused=True)); continue
                arr = r.run(t['config'], 'half')
                save(p, dict(F=None if arr is None else float(arr['kl'].mean())))
            print(bb, i + 1, '/', len(doc['rows']), flush=True)
        del r; gc.collect(); torch.cuda.empty_cache()


# ----------------------------------------------------------------------------------------------- part B
def _images(spec, n):
    import torch
    from leaf.evaluation.data import SUBSETS, load_split
    base, sl = SUBSETS['half']; im, _ = load_split(base); im = im[sl][:n]
    mean = torch.tensor(spec.mean, device='cuda').view(1, 3, 1, 1) * 255; std = torch.tensor(spec.std, device='cuda').view(1, 3, 1, 1) * 255
    for i in range(0, len(im), 10):
        x = torch.from_numpy(np.array(im[i:i + 10])).cuda()
        yield (x.permute(0, 3, 1, 2).float() - mean) / std


def _dense_pass(vit, x):
    """Per block: residual stream after attention (where a station acts) and the head-mean CLS attention row."""
    import torch
    from leaf.model.token_ops import cls_attention
    h, ys, ps = vit.norm_pre(vit.patch_drop(vit._pos_embed(vit.patch_embed(x)))), [], []
    for l, blk in enumerate(vit.blocks):
        out, p = cls_attention(blk, blk.norm1(h), None)
        h = h + blk.drop_path1(blk.ls1(out)); ys.append(h); ps.append(p)
        h = h + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(h))))
    return ys, ps


def _rank(v):
    import torch
    return v.argsort(-1).argsort(-1).float()


def _spearman(a, b):
    import torch
    ra, rb = _rank(a), _rank(b); ra = ra - ra.mean(-1, keepdim=True); rb = rb - rb.mean(-1, keepdim=True)
    return (ra * rb).sum(-1) / (ra.norm(dim=-1) * rb.norm(dim=-1))


def mech_ranking(bb, n=1000):
    import torch
    from leaf.backbones import load as load_bb
    from leaf.forecast.attention_forecast import AttentionForecast
    from leaf.forecast.kernels.alpha_score import zscored_alpha
    from leaf.model.token_ops import zscore
    c = A.Ctx(bb); vit, spec = load_bb(bb); B = len(vit.blocks)
    fcs = {l: AttentionForecast([vit.blocks[m] for m in range(l + 1, B)], c.d.at(l), c.d.rank).cuda() for l in range(1, B - 1)}
    for f in fcs.values(): f.use_triton()
    sp = {f'{l}/{a:.3f}': [] for l in fcs for a in ALPHAS}; nest = {}
    with torch.no_grad():
        for x in _images(spec, n):
            ys, ps = _dense_pass(vit, x); size = x.new_ones(x.shape[0], ys[0].shape[1], 1)
            S = {}
            for l, f in fcs.items():
                phi = torch.stack(ps[l + 1:]).sum(0)[:, 1:]                      # class-token attention of all later blocks
                for a in ALPHAS:
                    s = zscore(ps[l].clamp_min(1e-12).log(), 1) if a == 0 else zscored_alpha(f, ys[l], size, ps[l], a)
                    S[(l, a)] = s[:, 1:].float()
                    sp[f'{l}/{a:.3f}'] += _spearman(S[(l, a)], phi).tolist()
            m = S[(1, 0.)].shape[1]
            for q in Q:
                k = int(q * m)
                for a in ALPHAS:
                    for l1 in fcs:
                        low1 = S[(l1, a)].argsort(-1)[:, :k]
                        for l2 in fcs:
                            if l2 <= l1: continue
                            low2 = torch.zeros_like(S[(l2, a)], dtype=torch.bool).scatter_(1, S[(l2, a)].argsort(-1)[:, :k], True)
                            nest.setdefault(f'{q}/{a:.3f}/{l1}/{l2}', []).extend(low2.gather(1, low1).float().mean(1).tolist())
    out = dict(spearman_phi={k: float(np.mean(v)) for k, v in sp.items()}, nested={k: float(np.mean(v)) for k, v in nest.items()}, n=n)
    save(ROOT / bb / 'mech_ranking.json', out); return out


def _station_pass(vit, x, l1, keep, mode, sizes=True):
    """One merge-only station in block l1 (ranking by present attention); per later block, norms of the tokens and the
    CLS attention, with the original indices of every cluster."""
    import torch
    from leaf.model.token_ops import cls_attention, zscore
    from leaf.model.mlerp_vit import compress_norm
    h = vit.norm_pre(vit.patch_drop(vit._pos_embed(vit.patch_embed(x)))); size = h.new_ones(h.shape[0], h.shape[1], 1)
    rec, members = {}, None
    for l, blk in enumerate(vit.blocks):
        if l > l1: rec[l] = dict(x=h.float(), p=cls_attention(blk, blk.norm1(h), size)[1])
        out, p = cls_attention(blk, blk.norm1(h), None if l == 0 else size)
        h = h + blk.drop_path1(blk.ls1(out))
        if l == l1:
            n = h.shape[1]; s = zscore(p.clamp_min(1e-12).log(), 1)
            h, size, order, dst = compress_norm(h, size, s, n, keep, mode)
            members = (order, dst)
            if not sizes: size = torch.ones_like(size)
        h = h + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(h))))
    return rec, members


def mech_merge(bb, n=1000, blocks=(2, 4, 6), keep_frac=.6):
    import torch
    from leaf.backbones import load as load_bb
    vit, spec = load_bb(bb); B = len(vit.blocks); res = {}
    with torch.no_grad():
        for x in _images(spec, n):
            hd = vit.norm_pre(vit.patch_drop(vit._pos_embed(vit.patch_embed(x))))
            from leaf.model.token_ops import cls_attention
            dense = {}
            for l, blk in enumerate(vit.blocks):
                dense[l] = dict(x=hd.float(), p=cls_attention(blk, blk.norm1(hd), None)[1])
                out, _ = cls_attention(blk, blk.norm1(hd), None); hd = hd + blk.drop_path1(blk.ls1(out))
                hd = hd + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(hd))))
            N = hd.shape[1]; keep = int(round(keep_frac * N))
            for l1 in blocks:
                for name, mode, sizes in (('M3', 'M3', True), ('M0', 'M0', True), ('M3_nosize', 'M3', False)):
                    rec, (order, dst) = _station_pass(vit, x, l1, keep, mode, sizes)
                    b = x.shape[0]
                    # cluster id (reduced position) of every original token
                    cid = torch.full((b, N), -1, device=x.device, dtype=torch.long)
                    cid.scatter_(1, order[:, :keep], torch.arange(keep, device=x.device).expand(b, -1))
                    cid.scatter_(1, order[:, keep:], dst)
                    merged = torch.zeros(b, keep, device=x.device).scatter_add_(1, dst, torch.ones_like(dst, dtype=torch.float)) > 0
                    for m in range(l1 + 1, B):
                        dr, rr = dense[m], rec[m]
                        dx = dr['x'].gather(1, order[:, :keep, None].expand(-1, -1, dr['x'].shape[-1]))
                        ratio = (rr['x'].norm(dim=-1) / dx.norm(dim=-1))[merged]
                        mass = torch.zeros(b, keep, device=x.device).scatter_add_(1, cid, dr['p'].float())
                        tv = .5 * (rr['p'].float() - mass).abs().sum(-1)
                        key = f'{name}/{l1}/{m}'
                        d = res.setdefault(key, dict(log_norm_ratio=[], tv=[]))
                        d['log_norm_ratio'].append(float(ratio.log().mean())); d['tv'] += tv.tolist()
    out = {k: dict(norm_ratio=float(np.exp(np.mean(v['log_norm_ratio']))), tv=float(np.mean(v['tv']))) for k, v in res.items()}
    save(ROOT / bb / 'mech_merge.json', dict(blocks=list(blocks), keep_frac=keep_frac, n=n, results=out)); return out


# ----------------------------------------------------------------------------------------------- report
def report(bbs=BBS):
    from scipy.stats import spearmanr, wilcoxon
    out = {}
    for bb in bbs:
        doc = load(ROOT / bb / 'twins.json'); rows = doc['rows']; rep = {}
        for v in ('drawn',) + VARIANTS:
            F, Fh, bud, K = [], [], [], []
            for r in rows:
                f = r['drawn']['F'] if v == 'drawn' else (load(ROOT / bb / 'F' / f"{r[v]['sha']}.json")['F'] if (ROOT / bb / 'F' / f"{r[v]['sha']}.json").exists() else None)
                if f is None: continue
                F.append(f); Fh.append(r[v if v != 'drawn' else 'drawn']['fhat']); bud.append(r['budget']); K.append(r['K'])
            F, Fh, K = np.array(F), np.array(Fh), np.array(K)
            if not len(F): continue
            lr = np.log(F / Fh)
            within = {b: float(spearmanr(F[np.array(bud) == b], Fh[np.array(bud) == b])[0]) for b in sorted(set(bud))}
            rep[v] = dict(n=len(F), ratio_median=float(np.exp(np.median(lr))), abs_log_ratio_median=float(np.median(np.abs(lr))),
                          spearman=float(spearmanr(F, Fh)[0]), spearman_within=within,
                          ratio_by_K={f'{a}-{b}': float(np.exp(np.median(lr[(K >= a) & (K <= b)]))) for a, b in ((2, 4), (5, 6), (7, 8)) if ((K >= a) & (K <= b)).any()})
        # paired: |log F/F-hat| of the drawn schedule vs its twin (schedules where the twin differs)
        for v in VARIANTS:
            d, t = [], []
            for r in rows:
                p = ROOT / bb / 'F' / f"{r[v]['sha']}.json"
                if r[v]['same'] or not p.exists() or load(p)['F'] is None: continue
                d.append(abs(np.log(r['drawn']['F'] / r['drawn']['fhat']))); t.append(abs(np.log(load(p)['F'] / r[v]['fhat'])))
            if d: rep[f'paired_{v}'] = dict(n=len(d), drawn=float(np.median(d)), twin=float(np.median(t)), p=float(wilcoxon(d, t).pvalue),
                                             twin_worse=int(sum(b > a for a, b in zip(d, t))))
        for name in ('mech_ranking', 'mech_merge'):
            if (ROOT / bb / f'{name}.json').exists(): rep[name] = load(ROOT / bb / f'{name}.json')
        out[bb] = rep
    save(ROOT / 'report.json', out)
    for bb, rep in out.items():
        print('==', bb)
        for v in ('drawn',) + VARIANTS:
            if v in rep: r = rep[v]; print(f"{v:7s} n={r['n']:3d} F/Fhat {r['ratio_median']:.2f} |log| {r['abs_log_ratio_median']:.2f} rho {r['spearman']:.3f} within {[round(x, 2) for x in r['spearman_within'].values()]} byK {r['ratio_by_K']}")
        for v in VARIANTS:
            if f'paired_{v}' in rep: print('paired', v, rep[f'paired_{v}'])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=('twins', 'measure', 'mech', 'report'))
    ap.add_argument('--backbones', default=','.join(BBS))
    a = ap.parse_args(); bbs = a.backbones.split(',')
    if a.stage == 'twins': twins(bbs)
    elif a.stage == 'measure': measure(bbs)
    elif a.stage == 'mech':
        for bb in bbs: mech_ranking(bb); mech_merge(bb)
    else: report(bbs)


if __name__ == '__main__':
    main()
