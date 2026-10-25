"""LEAF inside the vision encoder of a small VLM with a class token: LLaVA-Phi-3-mini (xtuner/llava-phi-3-mini-hf;
CLIP ViT-L/14 at 336 px, 577 tokens, Phi-3-mini 3.8B).

LLaVA reads the patch tokens after block 22 (vision_feature_layer = -2) and drops the class token, so the reduced
encoder is the timm ViT truncated to 23 blocks, without its final norm, and every surviving patch token goes through the
projector into the LLM: fewer tokens out of the encoder means a shorter LLM prefill. The class token still runs through
every block, as in the MAE backbones, but the readout never uses it.

Answers are scored from the next-token logits after the prompt, without generation: ScienceQA-IMG by the option
letter with the largest logit among the offered options, POPE by 'Yes' against 'No'. The distortion of a schedule is
the KL divergence of that next-token distribution (full vocabulary) to the dense model's, as the classifiers' KL.

Stages (python leaf/reproduction/experiments/vlm_llava.py <stage>; data in data/vlm, results in ROOT):
  prepare (CPU)  decode ScienceQA-IMG validation/test and POPE images (shortest side 336 bicubic, centre crop 336, as
                 the LLaVA processor) into uint8 caches, with prompts and answers -> data/vlm/cache
  check   (GPU)  the truncated timm tower equals the HF vision tower (hidden state -2); dense accuracy through the
                 HF model and through the tower path agree -> ROOT/check.json
  groups  (GPU)  head groups of the forecast D on 256 calibration images (leaf.forecast.head_groups) -> ROOT/groups.json
  teacher (GPU)  dense next-token log-probabilities on the calibration items (CALIB of sqa_val) -> ROOT/teacher.npy
  probe   (GPU)  single-station probes (PROBE_BLOCKS x DELTAS x THETAS x ALPHAS): mean KL on the calibration items
  plan    (CPU)  for each output count T in COUNTS: the DP schedule (separable probe surrogate, any blocks, exactly T
                 patch tokens out) and the baselines at T, sealed -> ROOT/plans.json
  eval    (GPU)  ScienceQA-IMG test and POPE accuracy of every sealed arm -> ROOT/runs/<split>/<hash>.json|npz
Baselines at the same output count: 'cls_out' keeps the T patches with the highest class-token attention of the last
block (the [CLS]-attention selection of FasterVLM/VisionZip's dominant tokens), 'cls_out_merge' merges the rest into
their most similar kept tokens instead of dropping them (one station, theta = 1), and 'uniform' removes the same number
of tokens in every block (ToMe-like schedule with the LEAF station).
GPU stages yield to the pending DiffRate CLIP retime (results/logs/diffrate_clip_more.log) once the other GPU is idle,
so that latency is measured with both GPUs idle. Every stage stops when less than MIN_FREE_GB of disk would remain.
"""
import argparse
import io
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

MODEL = 'xtuner/llava-phi-3-mini-hf'
DATA = REPO / 'data/vlm'
CACHE = DATA / 'cache'
ROOT = REPO / 'results/vlm_llava'
SIDE = 336
MEAN, STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
FEATURE_BLOCKS = 23                       # vision_feature_layer = -2 of 24 blocks
LETTERS = 'ABCDE'
MIN_FREE_GB = 3.0
N_TOKENS = 577
CALIB = 256                               # calibration items: the first CALIB of sqa_val
PROBE_BLOCKS = (2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 21, 22)
DELTAS = (72, 144, 288, 384, 448, 512, 544, 570)
THETAS, ALPHAS = (0., .5, 1.), (0., .75)
COUNTS = (64, 128, 192)                   # patch tokens into the LLM
STEP = 8                                  # token-count grid of the DP
DR_LOG = REPO / 'results/logs/diffrate_clip_more.log'


def free_gb(): return shutil.disk_usage(REPO).free / 2 ** 30


def guard(extra_gb=0.):
    if free_gb() - extra_gb < MIN_FREE_GB:
        raise SystemExit(f'stop: {free_gb():.1f} GB free, {extra_gb:.1f} GB needed, minimum {MIN_FREE_GB} GB')


def load(p): return json.loads(Path(p).read_text())


def save(p, obj):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp'); tmp.write_text(json.dumps(obj, indent=1) + '\n'); tmp.replace(p)


# ----------------------------------------------------------------------------------------------- data
def square(img):
    from PIL import Image
    img = img.convert('RGB'); w, h = img.size; s = SIDE / min(w, h)
    img = img.resize((max(SIDE, round(w * s)), max(SIDE, round(h * s))), Image.BICUBIC)
    w, h = img.size; l, t = (w - SIDE) // 2, (h - SIDE) // 2
    return np.asarray(img.crop((l, t, l + SIDE, t + SIDE)), dtype=np.uint8)


def sqa_prompt(row):
    opts = '\n'.join(f'{LETTERS[i]}. {c}' for i, c in enumerate(row['choices']))
    ctx = f"Context: {row['hint']}\n" if row['hint'] else ''
    return f"{ctx}{row['question']}\n{opts}\nAnswer with the option's letter from the given choices directly."


def pope_prompt(row): return f"{row['question']}\nAnswer the question using a single word or phrase."


def prepare():
    import pyarrow.parquet as pq
    from PIL import Image
    jobs = {'sqa_val': (sorted((DATA / 'scienceqa_img/data').glob('validation-*.parquet')), 'sqa'),
            'sqa_test': (sorted((DATA / 'scienceqa_img/data').glob('test-*.parquet')), 'sqa'),
            'pope': (sorted((DATA / 'pope/data').glob('test-*.parquet')), 'pope')}
    for name, (files, kind) in jobs.items():
        if (CACHE / f'{name}.json').exists(): continue
        rows = [r for f in files for r in pq.read_table(f).to_pylist()]
        keys, images, items = {}, [], []
        for r in rows:                     # POPE repeats each image for many questions: store each image once
            key = r.get('image_source') or len(images)
            if key not in keys:
                keys[key] = len(images); images.append(square(Image.open(io.BytesIO(r['image']['bytes']))))
            if kind == 'sqa':
                items.append(dict(image=keys[key], prompt=sqa_prompt(r), answer=LETTERS[r['answer']], n_options=len(r['choices'])))
            else:
                items.append(dict(image=keys[key], prompt=pope_prompt(r), answer=r['answer'].strip().lower(), category=r['category']))
        arr = np.stack(images); guard(arr.nbytes / 2 ** 30)
        CACHE.mkdir(parents=True, exist_ok=True); np.save(CACHE / f'{name}_{SIDE}_u8.npy', arr)
        save(CACHE / f'{name}.json', dict(name=name, n_images=len(images), items=items))
        print(name, len(items), 'items', len(images), 'images', f'{arr.nbytes / 2 ** 30:.2f} GB', flush=True)


def split(name):
    meta = load(CACHE / f'{name}.json')
    return np.load(CACHE / f'{name}_{SIDE}_u8.npy', mmap_mode='r'), meta['items']


# ----------------------------------------------------------------------------------------------- model
class Llava:
    """The HF LLaVA model (BF16) with prompt encoding, the projector and the LLM step on given vision tokens."""

    def __init__(self, device='cuda'):
        import torch
        from transformers import AutoProcessor, LlavaForConditionalGeneration
        self.torch, self.device = torch, device
        self.processor = AutoProcessor.from_pretrained(MODEL)
        self.model = LlavaForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16).to(device).eval()
        self.tok = self.processor.tokenizer
        core = self.model.model
        self.vision, self.projector, self.lm = core.vision_tower, core.multi_modal_projector, core.language_model
        self.embed = self.lm.get_input_embeddings()
        self.image_token = self.model.config.image_token_index
        letters = {c: self.tok.convert_tokens_to_ids(self.tok.tokenize(c))[-1] for c in LETTERS}
        self.letter_ids = [letters[c] for c in LETTERS]
        self.yes_ids = sorted({self.tok.convert_tokens_to_ids(self.tok.tokenize(w))[-1] for w in ('Yes', 'yes', ' Yes', ' yes')})
        self.no_ids = sorted({self.tok.convert_tokens_to_ids(self.tok.tokenize(w))[-1] for w in ('No', 'no', ' No', ' no')})

    def ids(self, prompt):
        """(prefix ids, suffix ids) around the image of the chat prompt."""
        text = self.tok.apply_chat_template([{'role': 'user', 'content': '<image>\n' + prompt}], tokenize=False, add_generation_prompt=True)
        ids = self.tok(text, add_special_tokens=False)['input_ids']
        k = ids.index(self.image_token)
        return ids[:k], ids[k + 1:]

    def pixels(self, u8):
        torch = self.torch
        x = torch.from_numpy(np.array(u8, copy=True)).to(self.device).permute(0, 3, 1, 2).float()
        mean = torch.tensor(MEAN, device=self.device).view(1, 3, 1, 1) * 255
        std = torch.tensor(STD, device=self.device).view(1, 3, 1, 1) * 255
        return (x - mean) / std

    def hf_tokens(self, pixels):
        """The HF vision tower's hidden state -2 (all 577 tokens) -- the reference of the timm tower."""
        out = self.vision(pixels.to(self.torch.bfloat16), output_hidden_states=True)
        return out.hidden_states[-2]

    def answer_logits(self, tokens, prompts):
        """Next-token logits [B, V] after each prompt with the given vision tokens [B, n, C] (class token first;
        LLaVA's 'default' strategy drops it). Prompts are left-padded."""
        torch = self.torch
        feats = self.projector(tokens[:, 1:].to(torch.bfloat16))
        seqs = []
        for f, p in zip(feats, prompts):
            pre, suf = self.ids(p)
            e = lambda ids: self.embed(torch.tensor(ids, device=self.device))
            seqs.append(torch.cat([e(pre), f, e(suf)]))
        L = max(len(s) for s in seqs); pad = self.embed(torch.tensor([self.tok.pad_token_id or 0], device=self.device))[0]
        x = torch.stack([torch.cat([pad.expand(L - len(s), -1), s]) for s in seqs])
        mask = torch.stack([torch.cat([torch.zeros(L - len(s), device=self.device), torch.ones(len(s), device=self.device)]) for s in seqs]).long()
        pos = (mask.cumsum(-1) - 1).clamp_min(0)
        h = self.lm(inputs_embeds=x, attention_mask=mask, position_ids=pos).last_hidden_state[:, -1]
        return self.model.lm_head(h).float()

    def predict(self, logits, items):
        out = []
        for z, it in zip(logits, items):
            if 'n_options' in it:
                out.append(LETTERS[int(z[self.letter_ids[:it['n_options']]].argmax())])
            else:
                out.append('yes' if z[self.yes_ids].max() > z[self.no_ids].max() else 'no')
        return out


def timm_tower(llava):
    """timm ViT-L/14 (QuickGELU, 336) with LLaVA's vision weights, truncated to FEATURE_BLOCKS blocks, no final norm."""
    import timm
    import torch
    vit = timm.create_model('vit_large_patch14_clip_quickgelu_336', pretrained=False, num_classes=0)
    sd = {k.replace('vision_model.', ''): v.float() for k, v in llava.vision.state_dict().items()}
    out = {'cls_token': sd['embeddings.class_embedding'].view(1, 1, -1),
           'pos_embed': sd['embeddings.position_embedding.weight'].unsqueeze(0),
           'patch_embed.proj.weight': sd['embeddings.patch_embedding.weight'],
           'norm_pre.weight': sd['pre_layrnorm.weight'], 'norm_pre.bias': sd['pre_layrnorm.bias']}
    for i in range(len(vit.blocks)):
        p = f'encoder.layers.{i}.'; q = f'blocks.{i}.'
        out[q + 'attn.qkv.weight'] = torch.cat([sd[p + f'self_attn.{n}_proj.weight'] for n in 'qkv'])
        out[q + 'attn.qkv.bias'] = torch.cat([sd[p + f'self_attn.{n}_proj.bias'] for n in 'qkv'])
        for a, b in (('attn.proj', 'self_attn.out_proj'), ('norm1', 'layer_norm1'), ('norm2', 'layer_norm2'),
                     ('mlp.fc1', 'mlp.fc1'), ('mlp.fc2', 'mlp.fc2')):
            out[q + a + '.weight'] = sd[p + b + '.weight']; out[q + a + '.bias'] = sd[p + b + '.bias']
    missing, unexpected = vit.load_state_dict(out, strict=False)
    assert not unexpected and set(missing) <= {'norm.weight', 'norm.bias'}, (missing, unexpected)
    vit.blocks = vit.blocks[:FEATURE_BLOCKS]; vit.norm = torch.nn.Identity()
    for p in vit.parameters(): p.requires_grad_(False)
    return vit.eval()


def tower_tokens(vit, pixels):
    """Dense timm tower: all tokens after block FEATURE_BLOCKS - 1."""
    x = vit.norm_pre(vit.patch_drop(vit._pos_embed(vit.patch_embed(pixels))))
    for blk in vit.blocks: x = blk(x)
    return x


def spec():
    """A leaf.backbones.Spec for the truncated tower (fields this version of Spec does not know are left out)."""
    import dataclasses
    from leaf.backbones import Spec
    kw = dict(name='llava_phi3_clip', timm_id='vit_large_patch14_clip_quickgelu_336', depth=FEATURE_BLOCKS, dim=1024,
              heads=16, mean=MEAN, std=STD, pool='token', n_tokens=N_TOKENS, patch=14, img_size=SIDE, source=MODEL)
    names = {f.name for f in dataclasses.fields(Spec)}
    return Spec(**{k: v for k, v in kw.items() if k in names})


YIELD_EXIT = 75


def retime_pending():
    log = DR_LOG.read_text() if DR_LOG.exists() else ''
    return 'latency waits until both GPUs are idle' in log and not any(
        s in log.split('latency waits')[-1] for s in ('retime done', 'ALL DONE', 'FAILED', 'stopped'))


def yield_gpu():
    """Exit (code YIELD_EXIT) when the DiffRate CLIP retime waits for idle GPUs and the other GPU is idle: the retime
    needs no process on either GPU, so this process must release its CUDA context; the pipeline restarts the stage
    after the retime (finished configs are cached)."""
    import subprocess
    if not retime_pending(): return
    mine = subprocess.check_output(['nvidia-smi', '--query-gpu=pci.bus_id', '--format=csv,noheader', '-i', '0'], text=True).strip()
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_bus_id', '--format=csv,noheader'], text=True).strip().splitlines()
    if any(r and r.split(', ')[1] != mine for r in apps): return
    print('releasing the GPUs to the DiffRate CLIP retime', flush=True); raise SystemExit(YIELD_EXIT)


class Runner:
    """Reduced tower + LLM on cached items; per-config results cached under ROOT/runs/<split>/<hash>."""

    def __init__(self):
        import torch
        from leaf.latency.timing import inference_copy
        self.torch = torch; self.llava = Llava(); self.spec = spec()
        self.vit = timm_tower(self.llava).to('cuda'); self.cast = inference_copy(self.vit)
        self.teacher = None

    def model(self, cfg):
        from leaf.reproduction.runners.alpha_runner import build_any
        if cfg['kind'] == 'dense': return None
        m = build_any(self.vit, self.spec, cfg); m.vit = self.cast
        for mod in m.modules():
            if mod is not self.cast and hasattr(mod, 'prepare_inference'): mod.prepare_inference(self.torch.bfloat16)
        m.head = lambda x, size: x          # every surviving token goes to the projector
        return m.eval()

    def tokens(self, model, pixels):
        return tower_tokens(self.cast, pixels) if model is None else model(pixels)

    def run(self, cfg, name, n=None, batch=8):
        from leaf.evaluation.runner import config_hash
        torch = self.torch
        p = ROOT / 'runs' / name / config_hash(cfg)
        if p.with_suffix('.json').exists(): return dict(np.load(p.with_suffix('.npz')))
        images, items = split(name); items = items[:n] if n else items
        model = self.model(cfg); kl, correct, out_tokens = [], [], None; t0 = time.time()
        if name == 'sqa_val' and n == CALIB and self.teacher is None: self.teacher = torch.from_numpy(np.load(ROOT / 'teacher.npy')).cuda()
        for i in range(0, len(items), batch):
            its = items[i:i + batch]; x = self.llava.pixels(np.stack([images[it['image']] for it in its]))
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                tok = self.tokens(model, x); out_tokens = tok.shape[1] - 1
                z = self.llava.answer_logits(tok, [it['prompt'] for it in its])
            lp = z.log_softmax(-1)
            if name == 'sqa_val' and n == CALIB:
                t = self.teacher[i:i + len(its)].float()
                kl += (t.exp() * (t - lp)).sum(-1).tolist()
            correct += [a == it['answer'] for a, it in zip(self.llava.predict(z, its), its)]
        del model; torch.cuda.empty_cache()           # the LLM leaves little room on a 16 GB card
        arr = dict(correct=np.array(correct), kl=np.array(kl, dtype=np.float32))
        p.parent.mkdir(parents=True, exist_ok=True); np.savez(p.with_suffix('.npz'), **arr)
        save(p.with_suffix('.json'), dict(config=cfg, split=name, n=len(items), patch_tokens_out=out_tokens, acc=100 * float(np.mean(correct)),
                                          kl=float(np.mean(kl)) if kl else None, seconds=time.time() - t0))
        return arr


def config(sites, counts, thetas, alphas, d):
    """'pda' config of a station layout (counts include the class token) with the M3 merge."""
    from leaf.planner.alpha import to_alpha_config
    from leaf.planner.configs import config_from_state
    from leaf.planner.plan import operator_config
    st = dict(sites=list(sites), counts=list(counts), thetas=list(thetas), ks=list(alphas))
    return to_alpha_config(operator_config(config_from_state(st, spec(), d), 'M3', True))


def forecast_config():
    from leaf.planner.configs import ForecastConfig
    return ForecastConfig(ROOT / 'groups.json', G=16, rank=32)


def probe_jobs():
    for l in PROBE_BLOCKS:
        for delta in DELTAS:
            for th in THETAS:
                for a in ALPHAS:
                    if l == FEATURE_BLOCKS - 1 and a > 0: continue
                    yield l, N_TOKENS - delta, th, a


# ----------------------------------------------------------------------------------------------- stages
def groups(n=256):
    import functools
    import leaf.forecast.head_groups as grouping
    images, _ = split('sqa_val')
    grouping.load_split = lambda name: (images, np.zeros(len(images), dtype=np.int64))
    llava = Llava(); vit = timm_tower(llava).to('cuda')
    grouping.group_heads(vit, spec(), ROOT / 'groups.json', n=n, split='sqa_val')


def teacher():
    import torch
    r = Runner(); images, items = split('sqa_val'); out = []
    for i in range(0, CALIB, 8):
        its = items[i:i + 8]; x = r.llava.pixels(np.stack([images[it['image']] for it in its]))
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            out.append(r.llava.answer_logits(tower_tokens(r.cast, x), [it['prompt'] for it in its]).log_softmax(-1).half().cpu().numpy())
    np.save(ROOT / 'teacher.npy', np.concatenate(out)); print('teacher', np.concatenate(out).shape, flush=True)


def probe():
    r = Runner(); d = forecast_config(); jobs = list(probe_jobs()); t0 = time.time()
    for i, (l, n, th, a) in enumerate(jobs):
        yield_gpu(); guard()
        arr = r.run(config([l], [n], [th], [a], d), 'sqa_val', n=CALIB)
        if i % 20 == 0: print(f'probe {i + 1}/{len(jobs)} block {l} n {n} theta {th} alpha {a} KL {arr["kl"].mean():.4f} ({time.time() - t0:.0f} s)', flush=True)
    # Floor: the keep-all config through the reduced model's code path. The teacher comes from the plain tower, and
    # the two paths differ by BF16 kernel noise that the LLM turns into a KL offset every probe carries.
    floor = r.run(config([], [], [], [], d), 'sqa_val', n=CALIB)['kl'].mean()
    rows = []
    for l, n, th, a in jobs:
        from leaf.evaluation.runner import config_hash
        js = load((ROOT / 'runs/sqa_val' / config_hash(config([l], [n], [th], [a], d))).with_suffix('.json'))
        rows.append(dict(block=l, count=n, theta=th, alpha=a, kl=js['kl']))
    save(ROOT / 'probes.json', dict(calib=CALIB, floor_kl=float(floor), records=rows)); print('probes', len(rows), 'floor KL', floor, flush=True)


def curves():
    """Probe curves with the keep-all floor subtracted (so that no station costs nothing), clipped at zero."""
    from leaf.planner.surrogate import ProbeCurve
    p = load(ROOT / 'probes.json'); floor = p.get('floor_kl', 0.); rows = {}
    for r in p['records']:
        rows.setdefault((r['theta'], r['alpha']), {}).setdefault(r['block'], {})[N_TOKENS - r['count']] = max(r['kl'] - floor, 0.)
    return {k: ProbeCurve(v, N_TOKENS) for k, v in rows.items()}


def dp(cv, final):
    """Exact minimum of the separable surrogate over schedules on the count grid that end with exactly `final` tokens
    (class token included); stations at any probed block, settings by the per-transition probe arg-min."""
    from leaf.planner.surrogate import grid
    counts = sorted(set(grid(N_TOKENS, STEP, final)) | {final}, reverse=True); counts = [c for c in counts if c >= final]
    idx = {c: i for i, c in enumerate(counts)}; inf = float('inf')

    def best(l, a, b):
        out = None
        for (th, al), c in cv.items():
            if l == FEATURE_BLOCKS - 1 and al > 0 or l not in c.curves: continue
            v = c.transition(l, a, b)
            if out is None or v < out[0] - 1e-12: out = (v, th, al)
        return out
    V = {c: (0. if c == final else inf, None) for c in counts}           # after the last block
    choice = {}
    for l in range(FEATURE_BLOCKS - 1, -1, -1):
        new = {}
        for a in counts:
            opts = [(V[a][0], a, None)]
            if l > 0:
                for b in counts[idx[a] + 1:]:
                    bt = best(l, a, b)
                    if bt is not None and V[b][0] < inf: opts.append((bt[0] + V[b][0], b, bt))
            v, b, bt = min(opts, key=lambda o: o[0]); new[a] = (v, b); choice[(l, a)] = (b, bt)
        V = new
    sites, outs, thetas, alphas, a = [], [], [], [], N_TOKENS
    for l in range(FEATURE_BLOCKS):
        b, bt = choice[(l, a)]
        if b != a: sites.append(l); outs.append(b); thetas.append(bt[1]); alphas.append(bt[2])
        a = b
    return dict(fhat=V[N_TOKENS][0], sites=sites, counts=outs, thetas=thetas, alphas=alphas)


def plan():
    from leaf.evaluation.runner import config_hash
    dest = ROOT / 'plans.json'
    if dest.exists(): return load(dest)
    cv, d = curves(), forecast_config(); jobs = {'dense': dict(config=dict(kind='dense'))}
    last = FEATURE_BLOCKS - 1
    for T in COUNTS:
        f = T + 1
        s = dp(cv, f)
        jobs[f'leaf/T{T}'] = dict(config=config(s['sites'], s['counts'], s['thetas'], s['alphas'], d), state=s)
        jobs[f'cls_out/T{T}'] = dict(config=config([last], [f], [0.], [0.], d))
        jobs[f'cls_out_merge/T{T}'] = dict(config=config([last], [f], [1.], [0.], d))
        r = int(np.ceil((N_TOKENS - f) / (FEATURE_BLOCKS - 1))); n, sites, outs = N_TOKENS, [], []
        for l in range(1, FEATURE_BLOCKS):
            m = max(f, n - r)
            if m < n: sites.append(l); outs.append(m); n = m
        jobs[f'uniform/T{T}'] = dict(config=config(sites, outs, [1.] * len(sites), [0.] * len(sites), d))
    for j in jobs.values(): j['config_sha'] = config_hash(j['config'])
    import hashlib
    out = dict(jobs=jobs, sha=hashlib.sha256(json.dumps(jobs, sort_keys=True).encode()).hexdigest(), time=time.time())
    save(dest, out); save(ROOT / 'schedules_readable.json', {k: stations(j['config']) for k, j in jobs.items()})
    print(json.dumps({k: v.get('state') for k, v in jobs.items() if k.startswith('leaf/')}, indent=1)); return out


def stations(cfg):
    """Readable station list of a 'pda' config: block, tokens after pruning, tokens after merging, forecast weight."""
    if cfg['kind'] == 'dense': return []
    out, n = [], N_TOKENS
    for l, (p, m) in enumerate(zip(cfg['prune_keep'], cfg['merge_keep'])):
        p, m = min(n, p), min(n, p, m)
        if m < n: out.append(dict(block=l, tokens_in=n, after_prune=p, after_merge=m, alpha=cfg.get('alpha', {}).get(str(l), 0.))); n = m
    return out


def evaluate():
    r = Runner(); jobs = load(ROOT / 'plans.json')['jobs']
    for name in ('sqa_test', 'pope'):
        for key, j in jobs.items():
            yield_gpu(); guard()
            arr = r.run(j['config'], name)
            print(name, key, f"{100 * arr['correct'].mean():.2f}", flush=True)


def check(n=256, batch=8):
    import torch
    llava = Llava(); vit = timm_tower(llava).to('cuda')
    out = dict(model=MODEL, time=time.time())
    images, items = split('sqa_val')
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        x = llava.pixels(images[:4])
        a, b = llava.hf_tokens(x).float(), tower_tokens(vit, x).float()
        out['tower_max_abs_diff'] = float((a - b).abs().max()); out['tower_rel_diff'] = float((a - b).norm() / a.norm())
    for name in ('sqa_val', 'pope'):
        images, items = split(name); sel = items[:n]; agree = correct_hf = correct_timm = 0
        for i in range(0, len(sel), batch):
            its = sel[i:i + batch]; x = llava.pixels(np.stack([images[it['image']] for it in its]))
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                za = llava.answer_logits(llava.hf_tokens(x), [it['prompt'] for it in its])
                zb = llava.answer_logits(tower_tokens(vit, x), [it['prompt'] for it in its])
            pa, pb = llava.predict(za, its), llava.predict(zb, its)
            agree += sum(u == v for u, v in zip(pa, pb))
            correct_hf += sum(u == it['answer'] for u, it in zip(pa, its)); correct_timm += sum(v == it['answer'] for v, it in zip(pb, its))
        out[name] = dict(n=len(sel), acc_hf=100 * correct_hf / len(sel), acc_tower=100 * correct_timm / len(sel), agreement=100 * agree / len(sel))
        print(name, out[name], flush=True)
    save(ROOT / 'check.json', out)
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=('prepare', 'check', 'groups', 'teacher', 'probe', 'plan', 'eval'))
    ap.add_argument('--n', type=int, default=256)
    a = ap.parse_args()
    guard(); ROOT.mkdir(parents=True, exist_ok=True)
    {'prepare': prepare, 'check': lambda: check(a.n), 'groups': groups, 'teacher': teacher, 'probe': probe,
     'plan': plan, 'eval': evaluate}[a.stage]()


if __name__ == '__main__':
    main()
