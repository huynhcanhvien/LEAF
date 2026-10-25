"""Zero-shot head of a CLIP image tower: projection, L2 normalisation, scaled cosine to fixed text embeddings.

With the ImageNet class embeddings (80 OpenAI templates, leaf/data/clip_text.py) the head makes CLIP a 1000-way
classifier, so every method, probe, KL objective and runner works on it unchanged. Retrieval swaps the classifier
for a dataset's caption embeddings (with_classifier); the outputs are then image-caption similarities.
"""
from contextlib import contextmanager

import torch
import torch.nn as nn


class ZeroShotHead(nn.Module):
    def __init__(self, proj, classifier, scale=100.):
        super().__init__()
        self.proj, self.scale = proj, float(scale)
        self.register_buffer('classifier', torch.as_tensor(classifier, dtype=torch.float32), persistent=False)

    def forward(self, x):
        z = self.proj(x).float()
        with torch.autocast(z.device.type, enabled=False):   # FP32 cosines: retrieval ranks thousands of captions
            z = z / z.norm(dim=-1, keepdim=True)
            return self.scale * z @ self.classifier.t()


@contextmanager
def with_classifier(head, classifier):
    """Temporarily replace the text embeddings of a ZeroShotHead (e.g. by caption embeddings for retrieval)."""
    old = head.classifier
    head.classifier = torch.as_tensor(classifier, dtype=torch.float32, device=old.device)
    try:
        yield head
    finally:
        head.classifier = old
