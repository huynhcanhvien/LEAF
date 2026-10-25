"""DiffRate's own rate search on CLIP ViT-B/16 (mentor comment L737): like-for-like DiffRate schedules instead of the
DeiT-B rates transferred to CLIP.

The search is the official DiffRate code (github.com/OpenGVLab/DiffRate at DIFFRATE_COMMIT; the repository has no
licence file, so it is fetched into DIFFRATE_REPO and never copied into this repository) applied to the CLIP image
tower with its zero-shot ImageNet head (leaf/model/clip_head.py), with the settings of the official search command:
3 epochs on the ImageNet training set, arch-lr 0.01 -> 0.001 (cosine, per epoch), batch 1024 (4 GPUs x 256),
granularity 4, FLOP loss weight 5, drop path 0.1, RandAugment rand-m9-mstd0.5-inc1, colour jitter 0.4, random erasing
0.25, mixup 0.8 / cutmix 1.0 with label smoothing 0.1 and repeated augmentation (3 copies of every image per global
batch). Deviations, none of which changes what is optimised: one GPU per budget with gradient accumulation to the same
global batch of 1024; the network weights are frozen (DiffRate only steps the rate parameters, so their gradients were
unused); CLIP's normalisation instead of ImageNet's; the training shards are streamed (timm/imagenet-1k-wds).

The adapter for timm 1.0 (the official code targets timm 0.4): blocks expose drop_path1/drop_path2 and LayerScale
(identity for CLIP) -- DiffRate's block uses self.drop_path, so it is aliased to the block's drop path; the attention
has q/k norms (identity for CLIP), checked to be identities.

Stages (python leaf/reproduction/experiments/diffrate_clip_search.py <stage>):
  fetch   clone the official code at DIFFRATE_COMMIT into DIFFRATE_REPO
  check   (GPU) adapter: keep-all equals the dense CLIP logits; the transferred DeiT-B rates through the official
          code equal LEAF's DiffRate baseline (same kept counts) on a val subset
  search  (GPU) --budgets 8.7 10.0 ... --gpu N: one search per budget -> ROOT/<budget>/{log.jsonl,rates.json}
  rates   (CPU) collect the searched kept counts -> ROOT/rates.json (input of clip_comparison.py, arm 'DiffRate (searched)')
"""
import argparse
import io
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import tarfile
import time

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
ROOT = REPO / 'results/diffrate_clip_search'
DIFFRATE_URL = 'https://github.com/OpenGVLab/DiffRate.git'
DIFFRATE_COMMIT = '91ae3e376e9c3f9f1e69f580fd902b61de5dfe0e'
DIFFRATE_REPO = Path(os.environ.get('DIFFRATE_REPO', Path.home() / '.cache/leaf/DiffRate'))
TRAIN = REPO / 'data/imagenet_tar'
BB = 'clip_b16'
BUDGETS = ('8.7', '10.0', '10.4', '11.5', '12.5')        # the DeiT-B targets, whose rates are transferred today
HP = dict(epochs=3, arch_lr=.01, arch_min_lr=.001, global_batch=1024, granularity=4, lamb=5., drop_path=.1,
          aa='rand-m9-mstd0.5-inc1', color_jitter=.4, reprob=.25, mixup=.8, cutmix=1., smoothing=.1, repeats=3,
          images=1281167)


def save(p, obj):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp'); tmp.write_text(json.dumps(obj, indent=1) + '\n'); tmp.replace(p)


def fetch():
    if not DIFFRATE_REPO.exists():
        DIFFRATE_REPO.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'clone', '-q', DIFFRATE_URL, str(DIFFRATE_REPO)], check=True)
    subprocess.run(['git', '-C', str(DIFFRATE_REPO), 'checkout', '-q', DIFFRATE_COMMIT], check=True)
    head = subprocess.check_output(['git', '-C', str(DIFFRATE_REPO), 'rev-parse', 'HEAD'], text=True).strip()
    if head != DIFFRATE_COMMIT: raise SystemExit(f'DiffRate at {head}, expected {DIFFRATE_COMMIT}')
    print('DiffRate', head, 'in', DIFFRATE_REPO)


def diffrate_modules():
    head = subprocess.check_output(['git', '-C', str(DIFFRATE_REPO), 'rev-parse', 'HEAD'], text=True).strip()
    if head != DIFFRATE_COMMIT: raise SystemExit(f'run the fetch stage first ({DIFFRATE_REPO} at {head})')
    import types
    # the package __init__ files also import the CAFormer patch, which needs timm 0.4: register empty packages and
    # import only the modules the ViT patch uses
    for name, sub in (('DiffRate', 'DiffRate'), ('DiffRate.patch', 'DiffRate/patch')):
        if name not in sys.modules:
            pkg = types.ModuleType(name); pkg.__path__ = [str(DIFFRATE_REPO / sub)]; sys.modules[name] = pkg
    import DiffRate.patch.deit as deit
    return deit


def build(drop_path=0., device='cuda'):
    """CLIP ViT-B/16 with the zero-shot head, patched by the official DiffRate code; only the rates are trainable."""
    import torch.nn as nn
    from timm.layers import DropPath
    from leaf.backbones import load
    deit = diffrate_modules()
    vit, spec = load(BB, device=device)
    for i, blk in enumerate(vit.blocks):
        if not all(isinstance(m, nn.Identity) for m in (blk.ls1, blk.ls2, blk.attn.q_norm, blk.attn.k_norm, getattr(blk.attn, 'norm', nn.Identity()))):
            raise RuntimeError('the DiffRate adapter assumes no LayerScale and no q/k norm')
        rate = drop_path * i / (len(vit.blocks) - 1)          # timm's linear drop-path rule, as in create_model
        blk.drop_path1 = blk.drop_path2 = DropPath(rate) if rate > 0 else nn.Identity()
    deit.apply_patch(vit, prune_granularity=HP['granularity'], merge_granularity=HP['granularity'])
    for blk in vit.blocks: blk.drop_path = blk.drop_path1    # DiffRateBlock uses timm 0.4's single drop_path
    vit.num_classes = vit.head.classifier.shape[0]           # classifier FLOPs in DiffRate's count (1000 classes)
    vit.to(device)
    for n, p in vit.named_parameters(): p.requires_grad_('ddp' in n and n.endswith('selected_probability'))
    return vit, spec


# ----------------------------------------------------------------------------------------------- data
def shard_stream(paths, seed):
    """(jpeg bytes, label) of the training shards in a shuffled shard order with a shuffle buffer."""
    rng = random.Random(seed); paths = list(paths); rng.shuffle(paths); buf = []
    for path in paths:
        with tarfile.open(path) as tar:
            item = {}
            for m in tar:
                if not m.isfile(): continue
                stem, ext = m.name.rsplit('.', 1)
                if ext in ('jpg', 'cls'): item.setdefault(stem, {})[ext] = tar.extractfile(m).read()
                if stem in item and len(item[stem]) == 2:
                    d = item.pop(stem); buf.append((d['jpg'], int(d['cls'])))
                    if len(buf) >= 4096: yield buf.pop(rng.randrange(len(buf)))
    rng.shuffle(buf); yield from buf


import torch.utils.data as tud


class Train(tud.IterableDataset):
    """Every image decoded once and augmented `repeats` times (repeated augmentation), the copies spread by a shuffle
    buffer; the shards are split over the DataLoader workers."""

    def __init__(self, seed, transform, epoch):
        super().__init__(); self.seed, self.transform, self.epoch = seed, transform, epoch

    def __iter__(self):
        from PIL import Image
        info = tud.get_worker_info(); wid, nw = (info.id, info.num_workers) if info else (0, 1)
        paths = sorted(TRAIN.glob('imagenet1k-train-*.tar'))
        rng = random.Random(self.seed * 1000 + self.epoch * 100 + wid); out = []
        for jpg, label in shard_stream(paths[wid::nw], self.seed * 1000 + self.epoch * 100 + wid):
            img = Image.open(io.BytesIO(jpg)).convert('RGB')
            for _ in range(HP['repeats']): out.append((self.transform(img), label))
            if len(out) >= 96:
                rng.shuffle(out); yield from out; out = []
        yield from out


def train_transform():
    from timm.data import create_transform
    from leaf.backbones import SPECS
    mean, std = SPECS[BB].mean, SPECS[BB].std
    return create_transform(224, is_training=True, color_jitter=HP['color_jitter'], auto_augment=HP['aa'],
                            interpolation='bicubic', re_prob=HP['reprob'], re_mode='pixel', re_count=1, mean=mean, std=std)


# ----------------------------------------------------------------------------------------------- check
def check(n=2000, batch=50):
    """Adapter checks on the cached val images (LEAF's preprocessing): keep-all = dense CLIP; transferred DeiT-B rates
    through the official code = LEAF's DiffRate baseline."""
    import torch
    from leaf.backbones import load
    from leaf.planner.configs import diffrate_schedules
    from leaf.evaluation.data import iterate, load_split
    vit, spec = build()
    vit.eval(); dense, _ = load(BB)
    from leaf.reproduction.runners.alpha_runner import build_any
    from leaf.planner.configs import diffrate_config
    out = {}
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        agree, n_seen, diff = {}, 0, 0.
        ref_models = {b: build_any(dense, spec, diffrate_config(spec, b)) for b in ('11.5', '8.7')}
        images, labels = load_split('val')
        for x, y in iterate(images, labels, batch, 'cuda', spec.mean, spec.std, limit=n):
            y = y.cpu(); n_seen += len(y)
            vit.set_kept_num([197] * 12, [197] * 12)
            a, _ = vit(x); b = dense(x)
            diff = max(diff, float((a.float() - b.float()).abs().max()))
            for bid, ref in ref_models.items():
                p, m = diffrate_schedules(spec)[bid]
                vit.set_kept_num(list(p), list(m)); o, _ = vit(x); r = ref(x)
                agree[bid] = agree.get(bid, 0) + int((o.argmax(-1) == r.argmax(-1)).sum())
                out.setdefault(f'top1_official_{bid}', 0); out[f'top1_official_{bid}'] += int((o.argmax(-1).cpu() == y).sum())
                out.setdefault(f'top1_leaf_{bid}', 0); out[f'top1_leaf_{bid}'] += int((r.argmax(-1).cpu() == y).sum())
    out = {k: 100 * v / n_seen for k, v in out.items()}
    out.update(n=n_seen, keep_all_max_abs_logit_diff=diff, **{f'prediction_agreement_{b}': 100 * v / n_seen for b, v in agree.items()})
    save(ROOT / 'check.json', out); print(json.dumps(out, indent=1))


# ----------------------------------------------------------------------------------------------- search
CKPT_EVERY = 100   # steps between checkpoints; SIGUSR1 saves at the next step and stops


def search(budget, micro=128, workers=12, seed=0, max_steps=None):
    """One DiffRate search for target_flops = budget (GMAC). Resumable: a checkpoint every CKPT_EVERY steps and at the
    end of every epoch; `kill -USR1 <pid>` saves after the current step and exits (exit code 3). A resumed epoch draws
    its remaining batches from a freshly seeded stream."""
    import signal
    import torch
    from timm.data import Mixup
    from timm.loss import SoftTargetCrossEntropy
    from timm.scheduler.cosine_lr import CosineLRScheduler
    dest = ROOT / budget; dest.mkdir(parents=True, exist_ok=True)
    if (dest / 'rates.json').exists(): print('done', budget); return
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    vit, spec = build(drop_path=HP['drop_path'])
    target = float(budget)
    opt = torch.optim.AdamW(vit.arch_parameters(), lr=HP['arch_lr'], weight_decay=0)
    sched = CosineLRScheduler(opt, t_initial=HP['epochs'], lr_min=HP['arch_min_lr'], cycle_decay=.1)   # timm 0.4: decay_rate
    scaler = torch.amp.GradScaler('cuda')
    mix = Mixup(mixup_alpha=HP['mixup'], cutmix_alpha=HP['cutmix'], prob=1., switch_prob=.5, mode='batch',
                label_smoothing=HP['smoothing'], num_classes=1000)
    crit = SoftTargetCrossEntropy()
    accum = HP['global_batch'] // micro
    steps_per_epoch = HP['images'] // HP['global_batch']
    start, first = 0, 0
    ck = dest / 'checkpoint.pt'
    if ck.exists():
        s = torch.load(ck, map_location='cuda'); vit.load_state_dict(s['model'], strict=False); opt.load_state_dict(s['opt'])
        scaler.load_state_dict(s['scaler'])
        start, first = (s['epoch'], s['step'] + 1) if s.get('step') is not None else (s['epoch'] + 1, 0)
        print('resumed', budget, 'epoch', start, 'step', first, flush=True)
    stop = []
    signal.signal(signal.SIGUSR1, lambda *_: stop.append(1))

    def checkpoint(epoch, step):
        torch.save(dict(model={k: v for k, v in vit.state_dict().items() if 'ddp' in k}, opt=opt.state_dict(),
                        scaler=scaler.state_dict(), epoch=epoch, step=step), ck.with_suffix('.tmp'))
        ck.with_suffix('.tmp').replace(ck)
    transform = train_transform()
    log = open(dest / 'log.jsonl', 'a')
    for epoch in range(start, HP['epochs']):
        sched.step(epoch); vit.train()
        s0 = first if epoch == start else 0
        loader = torch.utils.data.DataLoader(Train(seed + 7919 * s0, transform, epoch), batch_size=micro, num_workers=workers, pin_memory=True,
                                             drop_last=True, persistent_workers=False, prefetch_factor=4)
        it = iter(loader); t0 = time.time()
        for step in range(s0, steps_per_epoch):
            if max_steps is not None and step >= s0 + max_steps: break
            opt.zero_grad(set_to_none=True); lc = lf = 0.
            for _ in range(accum):
                x, y = next(it); x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
                x, y = mix(x, y)
                with torch.autocast('cuda', dtype=torch.float16):
                    out, flops = vit(x)
                    loss_cls = crit(out, y); loss_flops = ((flops / 1e9) - target) ** 2
                    loss = HP['lamb'] * loss_flops + loss_cls
                scaler.scale(loss / accum).backward(); lc += loss_cls.item() / accum; lf += loss_flops.item() / accum
            scaler.step(opt); scaler.update()
            if step % 50 == 0 or step == steps_per_epoch - 1:
                p, m = vit.get_kept_num()
                rec = dict(epoch=epoch, step=step, loss_cls=lc, loss_flops=lf, gflops=float(flops) / 1e9, lr=opt.param_groups[0]['lr'],
                           prune=p, merge=m, img_s=(step - s0 + 1) * HP['global_batch'] / (time.time() - t0), time=time.time())
                log.write(json.dumps(rec) + '\n'); log.flush()
                print(f"{budget} e{epoch} s{step}/{steps_per_epoch} cls {lc:.3f} flops {float(flops) / 1e9:.2f} ({lf:.3f}) {rec['img_s']:.0f} img/s", flush=True)
            if stop or (step + 1) % CKPT_EVERY == 0:
                checkpoint(epoch, step if step < steps_per_epoch - 1 else None)
                if stop: print('paused', budget, 'epoch', epoch, 'step', step, flush=True); sys.exit(3)
        del it, loader
        if max_steps is not None: return
        checkpoint(epoch, None)
    vit.eval(); _ = vit.calculate_flop_inference()
    p, m = vit.get_kept_num()
    save(dest / 'rates.json', dict(budget=budget, prune_kept_num=p, merge_kept_num=m, gflops=float(vit.calculate_flop_inference()) / 1e9,
                                   hp=HP, commit=DIFFRATE_COMMIT, time=time.time()))
    print('searched', budget, p, m, flush=True)


def rates():
    out = {}
    for b in BUDGETS:
        p = ROOT / b / 'rates.json'
        if p.exists():
            r = json.loads(p.read_text()); out[b] = dict(prune_kept_num=json.dumps(r['prune_kept_num']), merge_kept_num=json.dumps(r['merge_kept_num']), gflops=r['gflops'])
    save(ROOT / 'rates.json', dict(source=f'DiffRate search on {BB} ({DIFFRATE_URL} @ {DIFFRATE_COMMIT})', hp=HP, rates={'CLIP-B16-searched': out}))
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=('fetch', 'check', 'search', 'rates'))
    ap.add_argument('--budgets', nargs='*', default=list(BUDGETS))
    ap.add_argument('--micro', type=int, default=128)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--max-steps', type=int, default=None, help='search: stop after this many steps (smoke test, nothing saved)')
    a = ap.parse_args()
    if a.stage == 'fetch': fetch()
    elif a.stage == 'check': check()
    elif a.stage == 'search':
        for b in a.budgets: search(b, a.micro, a.workers, max_steps=a.max_steps)
    else: rates()


if __name__ == '__main__':
    main()
