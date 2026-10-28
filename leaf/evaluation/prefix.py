"""Shared prefix of single-station probes (opt-in: leaf probe --prefix-cache).

A probe with its station in block l runs blocks 0..l-1 exactly as the dense model does, so the tokens entering
block l are the same for every probe at that block. PrefixCache computes them once per block, with the loop of
ReducedViT.forward (no station), the runner's BF16 copy, batches and autocast, and the probe resumes from them
(forward(tokens, start=l)). Blocks only move forward: block l + 1 is computed from block l. The tokens of one split
stay on the GPU (CLIP ViT-L/14, 5,000 images: 5.3 GB in fp32).
"""
import time

import torch

from leaf.evaluation.data import iterate
from leaf.evaluation.gpu_coordination import gpu_work


class PrefixCache:
    def __init__(self, runner, split):
        self.runner, self.split = runner, split
        self.block, self.tokens, self.seconds = None, None, 0.

    def dense(self):
        spec = self.runner.spec
        n = [spec.n_tokens] * spec.depth
        return self.runner.inference_model(dict(kind='pda', base='pd', prune_keep=n, merge_keep=n))

    @torch.no_grad()
    def at(self, block):
        """This cache, advanced to the tokens entering `block` (one tensor per batch)."""
        if block == self.block: return self
        if block < 1: raise ValueError('a prefix needs at least one block')
        t0 = time.monotonic()
        r = self.runner
        with gpu_work():
            images, labels, _ = r.data(self.split)
            model = self.dense()
            if self.block is None or block < self.block:
                self.tokens = None; torch.cuda.empty_cache()
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    self.tokens = [model(x, stop=block) for x, _ in iterate(images, labels, r.batch, 'cuda', r.spec.mean, r.spec.std)]
            else:
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    for i, x in enumerate(self.tokens): self.tokens[i] = model(x, start=self.block, stop=block)
            del model
        self.block = block
        self.seconds += time.monotonic() - t0
        return self

    def batches(self, labels):
        """(tokens, labels) per batch, as data.iterate splits the split."""
        b = self.runner.batch
        for i, x in enumerate(self.tokens):
            yield x, torch.from_numpy(labels[i * b:(i + 1) * b]).to('cuda', non_blocking=True)
