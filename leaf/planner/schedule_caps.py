# Paper-final schedule cost snapshot, including per-station actual D group counts.
"""Token schedules: per-block (prune, merge) caps <-> stations (block, tokens after, merged share theta).

A genome is dict(p=[...], m=[...]) with one cap per block; `canonical` removes caps that do not cut.
All functions take the backbone's token count `total`, depth and width (defaults: ViT-B/16 at 224).
"""


def canonical(genome, total=197, depth=None):
    """Remove inactive caps; a station is a block whose caps really reduce the token count."""
    depth = len(genome['p']) if depth is None else depth
    if len(genome['p']) != depth or len(genome['m']) != depth:
        raise ValueError('one prune/merge cap is required per block')
    p, m, incoming, sites = [], [], total, []
    for l, (prune, merge) in enumerate(zip(genome['p'], genome['m'])):
        prune = min(incoming, max(2, int(prune)))
        merge = min(prune, max(2, int(merge)))
        if merge < incoming:
            p.append(prune); m.append(merge); sites.append(l); incoming = merge
        else:
            p.append(total); m.append(total)
    return dict(p=p, m=m, sites=sites)


def token_counts(genome, total=197):
    """Tokens entering and leaving every block."""
    g = canonical(genome, total=total)
    before, after, n = [], [], total
    for l, (p, m) in enumerate(zip(g['p'], g['m'])):
        before.append(n)
        if l in g['sites']: n = min(n, p, m)
        after.append(n)
    return before, after


def block_gmacs(n_in, prune, merge, d=768):
    """MACs (billions) of one block with P scoring: qkv + proj (4d²n), SDPA (2n²d), the CLS row (dn),
    merge similarity ((P-M)·M·d) and the MLP on the tokens left after compression (8d²·n_out)."""
    p = min(n_in, prune); m = min(p, merge)
    macs = 4 * n_in * d * d + 2 * n_in * n_in * d + d * n_in
    if m < p: macs += (p - m) * m * d
    return (macs + 8 * d * d * m) / 1e9


def analytic_gmacs(p, m, total=197, d=768, classes=1000, head=None):
    """GMAC/image of the schedule without D (patch embedding + blocks + head; head: its GMAC, default classes x d)."""
    macs, n = ((total - 1) * d * d) / 1e9 + (classes * d / 1e9 if head is None else head), total
    for prune, merge in zip(p, m):
        macs += block_gmacs(n, prune, merge, d)
        n = min(n, prune, merge)
    return macs


def full_cost(p, m, alphas=None, total=197, d=768, groups=16, rank=32, patch=16, head=None):
    """Backbone+D MACs and a separate station scalar/comparison-operation estimate.

    Sorting is not a multiply-accumulate: report its comparison estimate separately.
    The conservative full column charges each scalar/comparison as one MAC-equivalent.
    D's token logits are n*G*d even with rank-r CLS projection (see AttentionForecast).
    """
    import math
    g = canonical(dict(p=list(p), m=list(m)), total); before, after = token_counts(g, total)
    backbone = analytic_gmacs(p, m, total, d, head=head); dmac = 0.; scalar = 0.
    # The historical backbone column uses d*d per patch. The actual RGB embedding
    # uses 3*patch*patch*d (equal only for patch 16 at width 768).
    patch_adjustment = (total - 1) * (3 * patch * patch * d - d * d) / 1e9
    alphas = alphas or {}
    for l in g['sites']:
        n, pp, mm = before[l], g['p'][l], after[l]
        if l < len(p) - 1 and float(alphas.get(str(l), alphas.get(l, 0.))) > 0:
            G = groups.get(str(l), groups.get(l, 0)) if isinstance(groups, dict) else groups
            r = rank or d
            dmac += (d * G * r + G * r * d + n * G * d) / 1e9
            scalar += n * G * 8 + n * d * 5
        scalar += n * (math.ceil(math.log2(max(n, 2))) + 12)
        scalar += (pp - mm) * (d + 2) + mm * d  # weighted merge/reduction
    return dict(backbone=backbone, patch_adjustment_gmac=patch_adjustment,
                backbone_full=backbone + patch_adjustment, d_gmac=dmac, station_gops=scalar / 1e9,
                full=backbone + patch_adjustment + dmac + scalar / 1e9)


def schedule(sites, counts, theta, total=197, depth=12):
    """Genome from station blocks, tokens after each station and merged share theta:
    p = m + round((n_in - m) * theta)."""
    if len(sites) != len(counts): raise ValueError('one count per station required')
    thetas = [theta] * len(sites) if isinstance(theta, (int, float)) else list(theta)
    p, m, n = [total] * depth, [total] * depth, total
    for l, c, t in zip(sites, counts, thetas):
        c = max(3, min(n - 1, int(c)))
        m[l] = c; p[l] = c + round((n - c) * t); n = c
    return canonical(dict(p=p, m=m), total=total, depth=depth)


def station_view(genome, total=197):
    """(sites, counts after each station, merged share per station) of a genome."""
    g = canonical(genome, total=total); before, after = token_counts(g, total)
    sites = g['sites']; counts = [after[l] for l in sites]
    thetas = [(g['p'][l] - g['m'][l]) / max(1, before[l] - g['m'][l]) for l in sites]
    return sites, counts, thetas
