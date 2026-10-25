"""Segmentation backbones (Segmenter on ADE20K): the converted MMSegmentation checkpoint matches an independent
re-implementation of MMSegmentation's forward; a schedule without stations reproduces the dense model; the source
map scatters every input token back to its carrier (or to its feature at the station that pruned it); window grid,
stitching, area statistics and mIoU follow MMSegmentation; GMAC accounting and the planner DP at 1,025 tokens;
with a GPU, the segmented merge kernel used beyond 257 tokens. CPU (FP32), plus the kernel check on a GPU.

The checkpoint parts need the MMSegmentation checkpoint of seg_b_ade (downloaded on first use); skipped when absent
and the download fails."""
from common import check, finish
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from leaf.backbones import SPECS, head_gmac
from leaf.data.ade20k import windows
from leaf.evaluation.segmentation import Stitcher, area_stats, miou, pixel_kl
from leaf.model.dense import SourceMap
from leaf.model.token_ops import compress

SPEC = SPECS['seg_b_ade']


def checkpoint_path():
    from leaf.backbones import _download
    try: return _download(SPEC.weights)
    except Exception as exc:
        print(f'SKIP checkpoint parts ({exc})'); return None


def mmseg_reference(state, x, depth=12, heads=12, n_classes=150):
    """MMSegmentation's EncoderDecoder forward from the raw state dict (nn.MultiheadAttention, mmcv FFN)."""
    s = {k: v.float() for k, v in state.items()}
    c = s['backbone.cls_token'].shape[-1]

    def layer(h, p, eps):
        y = F.layer_norm(h, (c,), s[p + 'ln1.weight'], s[p + 'ln1.bias'], eps)
        y, _ = F.multi_head_attention_forward(
            y.transpose(0, 1), y.transpose(0, 1), y.transpose(0, 1), c, heads, s[p + 'attn.attn.in_proj_weight'],
            s[p + 'attn.attn.in_proj_bias'], None, None, False, 0., s[p + 'attn.attn.out_proj.weight'],
            s[p + 'attn.attn.out_proj.bias'], training=False, need_weights=False)
        h = h + y.transpose(0, 1)
        y = F.layer_norm(h, (c,), s[p + 'ln2.weight'], s[p + 'ln2.bias'], eps)
        y = F.linear(F.gelu(F.linear(y, s[p + 'ffn.layers.0.0.weight'], s[p + 'ffn.layers.0.0.bias'])),
                     s[p + 'ffn.layers.1.weight'], s[p + 'ffn.layers.1.bias'])
        return h + y

    b = x.shape[0]
    h = F.conv2d(x, s['backbone.patch_embed.projection.weight'], s['backbone.patch_embed.projection.bias'], stride=16)
    side = h.shape[-1]
    h = h.flatten(2).transpose(1, 2)
    h = torch.cat((s['backbone.cls_token'].expand(b, -1, -1), h), 1) + s['backbone.pos_embed']
    for i in range(depth): h = layer(h, f'backbone.layers.{i}.', 1e-6)
    h = F.layer_norm(h, (c,), s['backbone.ln1.weight'], s['backbone.ln1.bias'], 1e-6)[:, 1:]
    h = F.linear(h, s['decode_head.dec_proj.weight'], s['decode_head.dec_proj.bias'])
    h = torch.cat((h, s['decode_head.cls_emb'].expand(b, -1, -1)), 1)
    for i in range(2): h = layer(h, f'decode_head.layers.{i}.', 1e-5)
    h = F.layer_norm(h, (c,), s['decode_head.decoder_norm.weight'], s['decode_head.decoder_norm.bias'], 1e-5)
    patches = F.normalize(F.linear(h[:, :-n_classes], s['decode_head.patch_proj.weight']), dim=2)
    classes = F.normalize(F.linear(h[:, -n_classes:], s['decode_head.classes_proj.weight']), dim=2)
    masks = F.layer_norm(patches @ classes.transpose(1, 2), (n_classes,), s['decode_head.mask_norm.weight'],
                         s['decode_head.mask_norm.bias'], 1e-5)
    return masks.permute(0, 2, 1).reshape(b, n_classes, side, side)


def backbone(path):
    from leaf.backbones import load
    from leaf.api import build_model
    vit, spec = load('seg_b_ade', device='cpu')
    check('seg_b_ade registry', (spec.task, spec.pool, spec.n_tokens, spec.img_size, spec.num_classes, spec.mean[0])
          == ('segmentation', 'dense', 1025, 512, 150, .5))
    torch.manual_seed(0); x = torch.randn(2, 3, 512, 512)
    state = torch.load(path, map_location='cpu', weights_only=False)['state_dict']
    with torch.no_grad():
        ours, ref = vit(x), mmseg_reference(state, x)
    err = (ours - ref).abs().max().item()
    check('converted checkpoint = MMSegmentation forward', ours.shape == (2, 150, 32, 32) and err < 1e-3, f'max |diff| {err:.2e}')
    N = spec.n_tokens
    with torch.no_grad():
        same = build_model(vit, spec, dict(kind='pd', prune_keep=[N] * 12, merge_keep=[N] * 12, k={}, groups={}, rank=32))(x)
    check('no station = dense', torch.equal(same, ours))
    p, m = [N] * 12, [N] * 12; p[3], m[3], m[8] = 900, 700, 400
    for kind, extra in (('pd', {}), ('pdm', dict(merge=dict(mode='M3', kernel='triton_norm')))):
        model = build_model(vit, spec, dict(kind=kind, prune_keep=p, merge_keep=m, k={}, groups={}, rank=32, **extra))
        with torch.no_grad(): y = model(x)
        src = model.sources.src
        carriers = [set(r.tolist()) - {-1} for r in src]
        check(f'{kind}: 125 tokens pruned, every final token carries one', y.shape == ours.shape
              and (src < 0).sum(1).tolist() == [125, 125] and all(len(c) == 400 for c in carriers)
              and all(c == set(range(400)) for c in carriers))
    model = build_model(vit, spec, dict(kind='pd', prune_keep=[N] * 11 + [700], merge_keep=[N] * 11 + [600], k={}, groups={}, rank=32))
    with torch.no_grad(): y = model(x)
    kl = pixel_kl(y, ours).mean().item()
    check('last-block station: small per-pixel KL', 0 < kl < .5, f'{kl:.4f}')
    tome = build_model(vit, spec, dict(kind='tome', r_blocks=[0, 0, 200, 0, 0, 150]))
    with torch.no_grad(): y = tome(x)
    check('ToMe dense output (merges at blocks 2 and 5)', y.shape == ours.shape and tome.sources.src.max().item() == 1025 - 351
          and set(tome.sources.src[0].tolist()) == set(range(1025 - 350)))
    try:
        build_model(vit, spec, dict(kind='evit', keep_rate=.7))(x); refused = False
    except NotImplementedError:
        refused = True
    check('baselines without source tracking refuse dense output', refused)


def trace(x0, stations):
    """Reference: Python lists of the input tokens carried by each token, frozen features of pruned ones."""
    b, n0, c = x0.shape
    out = []
    for bi in range(b):
        members = [[i] for i in range(n0)]; frozen = {}
        for x, order, destination, n_merge in stations:
            o = order[bi].tolist(); kept = [members[i] for i in o[:n_merge]]
            for src, dst in zip(o[n_merge:], destination[bi].tolist() if destination is not None else []):
                kept[dst] = kept[dst] + members[src]
            for i in set(range(len(members))) - set(o):
                for t in members[i]: frozen[t] = x[bi, i]
            members = kept
        out.append((members, frozen))
    return out


def source_map():
    torch.manual_seed(1)
    b, n, c = 3, 37, 8
    x = torch.randn(b, n, c); size = torch.ones(b, n, 1)
    sources = SourceMap(x); log = []
    for n_prune, n_merge in ((30, 24), (24, 20), (16, 16), (16, 9)):
        scores = torch.randn(b, x.shape[1])
        y, size, order, destination = compress(x, size, scores, n_prune, n_merge, fuse_merge=False)
        log.append((x, order, destination, y.shape[1])); sources.update(x, order, destination, y.shape[1])
        x = y + .1 * torch.randn_like(y)            # later blocks change the tokens
    final = torch.randn(b, x.shape[1], c)
    out = sources.reconstruct(final)
    ok = True
    for bi, (members, frozen) in enumerate(trace(log[0][0], log)):
        expect = torch.zeros(n, c)
        for j, group in enumerate(members):
            for t in group: expect[t] = final[bi, j]
        for t, v in frozen.items(): expect[t] = v
        ok &= torch.equal(out[bi], expect) and sum(map(len, members)) + len(frozen) == n
        ok &= all(group for group in members) and members[0] == [0]
    check('source map = reference trace (prune, merge, both)', ok)
    norm = torch.nn.LayerNorm(c)
    check('reconstruct applies the norm to carried and frozen features',
          torch.allclose(sources.reconstruct(final, norm), norm(out), atol=1e-6))
    # ToMe: the carrier map from unmerge equals ToMe's own source matrix (merge_source), over several blocks
    from baselines.third_party.tome.merge import bipartite_soft_matching, merge_source, merge_wavg
    x = torch.randn(b, n, c); size = torch.ones(b, n, 1); sources = SourceMap(x); source = None
    for r in (6, 9, 4):
        merge, unmerge = bipartite_soft_matching(torch.randn(b, x.shape[1], 5), r, True, False)
        source = merge_source(merge, x, source)
        y, size = merge_wavg(merge, x, size)
        ids = torch.arange(y.shape[1], dtype=torch.float32).view(1, -1, 1).expand(b, -1, 1)
        sources.remap(x, unmerge(ids)[..., 0].round().long(), prunes=False); x = y
    check('ToMe carrier map (unmerge) = merge_source', torch.equal(sources.src, source.argmax(1))
          and bool((source.sum(1) == 1).all()))


def window_grid():
    # MMSegmentation slide_inference: h_grids = max(H - 512 + 480 - 1, 0) // 480 + 1, last window aligned to the border
    def mmseg(h, w):
        out = []
        for hi in range(max(h - 512 + 479, 0) // 480 + 1):
            for wi in range(max(w - 512 + 479, 0) // 480 + 1):
                y1, x1 = hi * 480, wi * 480
                y2, x2 = min(y1 + 512, h), min(x1 + 512, w)
                y1, x1 = max(y2 - 512, 0), max(x2 - 512, 0)
                out.append((y1, x1, y2 - y1, x2 - x1))
        return out
    sizes = [(512, 512), (512, 683), (683, 512), (512, 1024), (512, 993), (512, 2048), (300, 2048), (512, 992)]
    check('window grid = MMSegmentation slide', all(windows(h, w) == mmseg(h, w) for h, w in sizes),
          str({s: len(windows(*s)) for s in sizes}))


def stitching():
    torch.manual_seed(2); k = 5
    # image 0: 512 x 683 -> 2 windows; image 1: 300 x 400 (padded, one window); original sizes differ from resized
    rows = windows(512, 683) + windows(300, 400)
    meta = dict(image=np.array([0, 0, 1]), y=np.array([r[0] for r in rows]), x=np.array([r[1] for r in rows]),
                h=np.array([r[2] for r in rows]), w=np.array([r[3] for r in rows]),
                resized=np.array([[512, 683], [300, 400]]), original=np.array([[600, 800], [300, 400]]))
    gts = [torch.randint(0, k, (600, 800)), torch.randint(0, k, (300, 400))]
    gts[0][:50] = 255
    meta['gt'] = np.concatenate([g.numpy().astype(np.uint8).ravel() for g in gts])
    meta['gt_offset'] = np.array([0, 600 * 800, 600 * 800 + 300 * 400])
    logits = torch.randn(3, k, 32, 32)
    st = Stitcher(meta, k, 512); st.add(logits[:2], 0); st.add(logits[2:], 2); arr = st.result()
    # reference: MMSegmentation-style accumulation without the streaming class
    preds = []
    for image, size in ((0, (512, 683)), (1, (300, 400))):
        canvas = torch.zeros(k, *size); count = torch.zeros(1, *size)
        for i in np.flatnonzero(meta['image'] == image):
            y, x, h, w = (int(meta[key][i]) for key in ('y', 'x', 'h', 'w'))
            up = F.interpolate(logits[i:i + 1], size=(512, 512), mode='bilinear', align_corners=False)[0]
            canvas[:, y:y + h, x:x + w] += up[:, :h, :w]; count[:, y:y + h, x:x + w] += 1
        preds.append(F.interpolate((canvas / count)[None], size=tuple(meta['original'][image]), mode='bilinear',
                                   align_corners=False)[0].argmax(0))
    ok = True
    for i, (pred, gt) in enumerate(zip(preds, gts)):
        keep = gt != 255; p, g = pred[keep].numpy(), gt[keep].numpy()
        inter = np.array([np.sum((p == c) & (g == c)) for c in range(k)])
        union = np.array([np.sum((p == c) | (g == c)) for c in range(k)])
        ok &= np.array_equal(arr['intersect'][i], inter) and np.array_equal(arr['union'][i], union)
        ok &= np.array_equal(arr['label_area'][i], np.bincount(g, minlength=k))
    check('stitching + area statistics = reference (ignore 255, padded window)', ok)
    ref = 100 * np.nanmean(np.where(arr['union'].sum(0) > 0, arr['intersect'].sum(0) / np.maximum(arr['union'].sum(0), 1), np.nan))
    check('mIoU = sum intersect / sum union, mean over classes', abs(miou(arr) - ref) < 1e-9 and abs(miou(arr, np.array([0, 1])) - ref) < 1e-9)
    absent = dict(intersect=np.array([[1, 0, 2]]), union=np.array([[2, 0, 4]]))
    check('classes without union are skipped', abs(miou(absent) - 50.) < 1e-9)
    t = torch.randn(2, k, 4, 4)
    check('pixel KL: zero for equal logits, positive otherwise', pixel_kl(t, t).abs().max() < 1e-6 and (pixel_kl(t + torch.randn_like(t), t) > 0).all())


def accounting_and_planner():
    from leaf.planner.gmac import dense_cost
    from leaf.planner.surrogate import JointTables, ProbeCurve
    from leaf.planner.lagrange import lagrange_dp
    from leaf.cli import default_deltas
    check('classifier head GMAC unchanged (1000 x d)', abs(head_gmac(SPECS['deit_b']) - 1000 * 768 / 1e9) < 1e-12)
    check('Segmenter-B dense ~129 GMAC (258 GFLOPs)', 128 < dense_cost(SPEC) < 131, f'{dense_cost(SPEC):.2f}')
    deltas = default_deltas(SPEC)
    check('probe grid scales to 1,025 tokens', deltas[-1] == 1022 and deltas[0] == 42, str(deltas))
    # synthetic probe curves: damage grows with removal and is larger early
    rng = np.random.default_rng(0)
    curves = {}
    for th in (0., .5, 1.):
        for a in (0., .5):
            curves[(th, a)] = ProbeCurve({l: {d: (1 + th * .1 + a * .05 + rng.random() * .01) * (d / 1024) ** 2 / (l + 1)
                                              for d in deltas} for l in range(1, 12) if not (l == 11 and a > 0)}, SPEC.n_tokens)
    t0 = time.monotonic()
    jt = JointTables(curves, SPEC.n_tokens, SPEC.depth, step=SPEC.count_step, dim=SPEC.dim)
    raw = lagrange_dp(jt, jt.mac, jt.mac, 1e-3)
    seconds = time.monotonic() - t0
    check('planner tables + DP at 1,025 tokens', raw is not None and len(jt.counts) == 65 and seconds < 60,
          f'{len(jt.counts)} counts, {seconds:.1f} s, stations {list(zip(raw["sites"], raw["counts"])) if raw else None}')


def segment_kernel():
    """Merge kernel beyond 257 tokens (sorted segments) = scatter mean; the paper's token counts keep the tile kernel."""
    from leaf.model.kernels import mean_merge
    torch.manual_seed(3); ok = True
    for b, p, m, c in ((4, 1025, 700, 768), (2, 900, 2, 64), (3, 300, 299, 32)):
        x = torch.randn(b, p, c, device='cuda'); size = torch.rand(b, p, 1, device='cuda') + .5
        dest = torch.randint(1, m, (b, p - m), device='cuda'); mx = torch.rand(b, p - m, device='cuda')
        ref = x[:, :m].scatter_reduce(1, dest[..., None].expand(-1, -1, c), x[:, m:], reduce='mean')
        rs = size[:, :m].scatter_reduce(1, dest[..., None], size[:, m:] * mx[..., None], reduce='sum')
        o, s = mean_merge.merge_inference(x, size, dest, mx, m)
        ok &= torch.allclose(o, ref, atol=1e-5) and torch.allclose(s, rs, rtol=1e-5, atol=1e-4)
    check('segmented merge kernel (> 257 tokens) = scatter mean and similarity-weighted size', ok)
    calls = []
    original = mean_merge.segment_merge
    mean_merge.segment_merge = lambda *a: calls.append(a) or original(*a)
    try:
        for p in (197, 257):
            x = torch.randn(2, p, 64, device='cuda'); size = torch.ones(2, p, 1, device='cuda')
            mean_merge.merge_inference(x, size, torch.randint(1, 100, (2, p - 100), device='cuda'), torch.rand(2, p - 100, device='cuda'), 100)
    finally:
        mean_merge.segment_merge = original
    check('197 and 257 tokens keep the tile kernel (paper bits unchanged)', not calls)


if __name__ == '__main__':
    torch.set_num_threads(min(8, os.cpu_count()))
    source_map(); window_grid(); stitching(); accounting_and_planner()
    if torch.cuda.is_available(): segment_kernel()
    else: print('SKIP segmented merge kernel (no GPU)')
    path = checkpoint_path()
    if path: backbone(path)
    finish()
