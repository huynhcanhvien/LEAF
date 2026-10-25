"""Off-the-shelf baselines on the shared frame (baselines/core.py), algorithms from the vendored official code.

ToMe     (Bolya et al., ICLR'23)   r tokens merged per block by bipartite soft matching on k.mean(1); size-weighted
                                   average; proportional attention (off for MAE, as the ToMe README prescribes).
PiToMe   (Tran et al., NeurIPS'24) energy-score protected merging, ratio of tokens kept per block, margins
                                   .75 - .75*l/L, BSM-PiToMe in the first ceil(L/2) blocks; no proportional attention.
EViT     (Liang et al., ICLR'22)   at blocks (3, 6, 9) keep ceil(rate*(N-1)) patch tokens by head-mean CLS attention,
                                   fuse the rest into one token weighted by that attention; no fine-tuning.
"""
import math

import torch

from .core import TokenViT
from .third_party.evit.helpers import complement_idx
from .third_party.pitome.merge import merge_wavg as pitome_wavg, pitome_vision
from .third_party.tome.merge import bipartite_soft_matching, merge_wavg


def parse_r(num_layers, r):
    """tome/utils.parse_r (constant r or (r, inflection) or a list)."""
    inflect = 0
    if isinstance(r, list):
        return list(r) + [0] * (num_layers - len(r))
    if isinstance(r, tuple): r, inflect = r
    min_val = int(r * (1.0 - inflect)); max_val = 2 * r - min_val
    step = (max_val - min_val) / (num_layers - 1)
    return [int(min_val + step * i) for i in range(num_layers)]


class ToMe(TokenViT):
    needs = ('kmean',)
    tracks = True       # unmerge of the output positions gives the carrier of every input token

    def __init__(self, vit, spec, r):
        super().__init__(vit, spec); self.r = parse_r(self.depth, r if not isinstance(r, list) else list(r))
        self.prop = spec.pool != 'avg'

    def reduce(self, l, x, size, info):
        r = self.r[l]
        if r <= 0: return x, size
        merge, unmerge = bipartite_soft_matching(info['kmean'], r, True, False)
        y, size = merge_wavg(merge, x, size)
        if self.sources is not None:
            b, n = y.shape[:2]
            ids = torch.arange(n, device=y.device, dtype=torch.float32).view(1, n, 1).expand(b, n, 1)
            self.sources.remap(x, unmerge(ids)[..., 0].round().long(), prunes=False)
        return y, size


class PiToMe(TokenViT):
    needs = ('kmean',)
    prop = False

    def __init__(self, vit, spec, ratio):
        super().__init__(vit, spec); self.ratio = float(ratio)
        self.margins = [.75 - .75 * (i / self.depth) for i in range(self.depth)]
        n_bsm = math.ceil(self.depth * .5); self.bsm = [True] * n_bsm + [False] * (self.depth - n_bsm)

    def reduce(self, l, x, size, info):
        if self.ratio >= 1.0: return x, size
        merge = pitome_vision(metric=info['kmean'], ratio=self.ratio, margin=self.margins[l], class_token=True, use_bsm_pitome=self.bsm[l])
        return pitome_wavg(merge, x, size)


class EViT(TokenViT):
    needs = ('cls',)
    prop = False

    def __init__(self, vit, spec, keep_rate, drop_loc=(3, 6, 9)):
        super().__init__(vit, spec); self.keep_rate = float(keep_rate)
        self.drop_loc = tuple(round(l * spec.depth / 12) for l in drop_loc)   # (3, 6, 9) for 12 blocks; scaled for ViT-L

    def reduce(self, l, x, size, info):
        if l not in self.drop_loc or self.keep_rate >= 1: return x, size
        B, N, C = x.shape
        left = math.ceil(self.keep_rate * (N - 1))
        if left == N - 1: return x, size
        cls_attn = info['cls'][:, 1:].to(x.dtype)                         # [B, N-1], head mean (evit Attention.forward)
        _, idx = torch.topk(cls_attn, left, dim=1, largest=True, sorted=True)
        index = idx.unsqueeze(-1).expand(-1, -1, C)
        non_cls = x[:, 1:]
        x_others = torch.gather(non_cls, dim=1, index=index)
        compl = complement_idx(idx, N - 1)
        non_topk = torch.gather(non_cls, dim=1, index=compl.unsqueeze(-1).expand(-1, -1, C))
        non_topk_attn = torch.gather(cls_attn, dim=1, index=compl)
        extra = torch.sum(non_topk * non_topk_attn.unsqueeze(-1), dim=1, keepdim=True)
        x = torch.cat([x[:, 0:1], x_others, extra], dim=1)
        # Size is not used by EViT itself; it only weights the MAE average pool (fused token = its source count).
        s_body = size[:, 1:]
        size = torch.cat([size[:, :1], torch.gather(s_body, 1, idx.unsqueeze(-1)),
                          torch.gather(s_body, 1, compl.unsqueeze(-1)).sum(1, keepdim=True)], 1)
        return x, size


class MCTF(TokenViT):
    """MCTF (Lee et al., CVPR'24), off the shelf. Merging happens BEFORE each activated block's attention:
    attention probabilities of the current tokens -> multi-criteria bipartite fusion (similarity, informativeness
    1/attention, size; bidirectional) -> the fused attention map (one-step-ahead) multiplies the new values."""
    needs = ('attn',)

    def __init__(self, vit, spec, r, activate=None, tau=(1, 20, 40)):
        super().__init__(vit, spec); self.r = r
        self.activate = set(range(1, self.depth) if activate is None else activate)
        self.base = dict(use_mctf=True, num_protected=1, r_eval=r, trace_source=0, prop_attn=1, one_step_ahead=1,
                         tau_sim=tau[0], tau_info=tau[1], tau_size=tau[2], bidirection=1, pooling_type=1)

    def forward(self, img):
        from .third_party.mctf.mctf import mctf
        vit = self.vit
        x = vit.norm_pre(vit.patch_drop(vit._pos_embed(vit.patch_embed(img))))
        info = dict(self.base, student_size=None)
        for l, blk in enumerate(vit.blocks):
            a = blk.attn; size = info['student_size']
            h = blk.norm1(x); B, N, C = h.shape
            if l in self.activate:
                q, k, _ = a.qkv(h).reshape(B, N, 3, a.num_heads, a.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
                logits = (q @ k.transpose(-2, -1)) * a.scale
                if size is not None: logits = logits + size.log()[:, None, None, :, 0]
                attn = logits.softmax(-1)
                x, attn_n = mctf(x, attn, info, 'student_', False)
                h = blk.norm1(x); B, N, C = h.shape
                v = a.qkv(h).reshape(B, N, 3, a.num_heads, a.head_dim).permute(2, 0, 3, 1, 4)[2]
                probs = attn_n if attn_n is not False else attn
            else:
                q, k, v = a.qkv(h).reshape(B, N, 3, a.num_heads, a.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
                logits = (q @ k.transpose(-2, -1)) * a.scale
                if size is not None: logits = logits + size.log()[:, None, None, :, 0]
                probs = logits.softmax(-1)
            out = (probs.to(v.dtype) @ v).transpose(1, 2).reshape(B, N, C)
            x = x + blk.drop_path1(blk.ls1(a.proj_drop(a.proj(out))))
            x = x + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(x))))
        size = info['student_size'] if info['student_size'] is not None else x.new_ones(x.shape[0], x.shape[1], 1)
        x = vit.norm(x)
        if self.spec.pool == 'avg':
            w = size[:, 1:]; x = (x[:, 1:] * w).sum(1) / w.sum(1)
        else:
            x = x[:, 0]
        return vit.head(vit.head_drop(vit.fc_norm(x)))


class PPT(TokenViT):
    """PPT (Wu et al., 2023), off the shelf: at blocks pp_loc, per image, ToMe pooling when the variance of the
    significance scores (CLS attention x value norm) is below the threshold, otherwise prune the r lowest-score
    tokens. Both branches are computed and selected per image (static shapes); proportional attention on."""
    needs = ('kmean', 'clsh')
    prop = True

    def __init__(self, vit, spec, r, threshold=7e-5, pp_loc=(3, 6, 9)):
        super().__init__(vit, spec); self.r, self.threshold = int(r), float(threshold)
        self.pp_loc = tuple(round(l * spec.depth / 12) for l in pp_loc)

    def reduce(self, l, x, size, info):
        from .third_party.ppt.score import PPTScore
        if l not in self.pp_loc: return x, size
        B, N, C = x.shape; r = min(self.r, (N - 1) // 2)
        if r <= 0: return x, size
        rows = info['clsh'].float()                                                       # [B,H,N] CLS rows
        scores = PPTScore.score_assignment_step(rows[:, :, None, :].expand(-1, -1, N, -1), info['v'].float())
        scores = torch.cat([2 * torch.ones(B, 1, device=x.device), scores], -1)            # CLS always kept
        pool = (torch.var(scores[:, 1:], dim=1) < self.threshold)[:, None, None]          # [B,1,1]
        merge, _ = bipartite_soft_matching(info['kmean'], r, True, False)
        xm, sm = merge_wavg(merge, x, size)
        keep = scores.sort(descending=True, dim=-1)[1][:, :-r]
        xp = x.gather(1, keep[..., None].expand(-1, -1, C)); sp = size.gather(1, keep[..., None])
        return torch.where(pool, xm, xp), torch.where(pool, sm, sp)
