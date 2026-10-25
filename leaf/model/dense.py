"""Source tracking for dense heads: every input token keeps the index of the token that carries it.

A station (token_ops.compress and its variants) returns `order` (input indices of the tokens it keeps, the first
n_merge kept as they are, the next ones merged) and `destination` (the kept token each merged one goes to). After
the station, an original token is carried by
  - the kept token at its position, if its carrier was kept;
  - the destination, if its carrier was merged;
  - nobody, if its carrier was pruned: it then keeps its carrier's feature at that station ('frozen', the early exit
    of pruned tokens) for the rest of the network.
`reconstruct` scatters the final tokens back to the input grid (frozen features where a token left early), so a
dense head sees all original positions, as in ToMe's unmerge. All shapes are fixed per schedule (CUDA graphs).
"""
import torch


class SourceMap:
    def __init__(self, x):
        b, n, _ = x.shape
        self.src = torch.arange(n, device=x.device).expand(b, n).contiguous()     # carrier of each input token, -1: pruned
        self.frozen = None                                                         # [B, N0, C] features of pruned tokens

    def update(self, x, order, destination, n_merge):
        """x: the tokens entering the station [B, n, C]; order [B, n_prune]; destination [B, n_prune - n_merge] or None."""
        b, n, c = x.shape
        if destination is not None and destination.ndim != 2:
            raise NotImplementedError('source tracking needs one destination per merged token')
        new = torch.full((b, n), -1, dtype=torch.long, device=x.device)
        new.scatter_(1, order[:, :n_merge], torch.arange(n_merge, device=x.device).expand(b, n_merge))
        if destination is not None: new.scatter_(1, order[:, n_merge:], destination)
        self.remap(x, new, prunes=order.shape[1] < n)

    def remap(self, x, new, prunes=True):
        """new [B, n]: the output token carrying each input token of a reduction step, -1 if pruned. prunes must be
        known statically (CUDA graphs): False skips the frozen-feature update."""
        c = x.shape[-1]
        alive = self.src >= 0
        carrier = self.src.clamp_min(0)
        nxt = torch.where(alive, new.gather(1, carrier), -1)
        if prunes:                          # some carriers leave here: keep their features
            left = (alive & (nxt < 0))[..., None]
            features = x.gather(1, carrier[..., None].expand(-1, -1, c))
            self.frozen = torch.where(left, features, 0 if self.frozen is None else self.frozen)
        self.src = nxt

    def reconstruct(self, x, norm=None):
        """[B, N0, C]: each input position gets its carrier's final feature, or its frozen feature (both through norm)."""
        c = x.shape[-1]
        if norm is not None: x = norm(x)
        out = x.gather(1, self.src.clamp_min(0)[..., None].expand(-1, -1, c))
        if self.frozen is None: return out
        frozen = self.frozen if norm is None else norm(self.frozen)
        return torch.where((self.src < 0)[..., None], frozen.to(out.dtype), out)
