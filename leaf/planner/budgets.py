"""Latency budgets from measured anchors, as in the paper.

An anchor is a reference schedule timed on the target GPU: DiffRate's released schedule of x GMAC (`DRx`), a
GMAC-planned LEAF schedule, or any config. Each anchor gives the budget factor x its measured latency (0.97 in the
paper). Two faster budgets `extra1` and `extra2` extend the range below the fastest anchor by the mean gap between
anchor budgets. A backbone without anchors gets fractions of its dense latency (`dense0.xx`, the AugReg budgets).

`anchor_budgets` is `measured_budgets` of the LEAF-M0 campaign (evaluation/paper_final.py of its code), which
wrote the paper's budgets (schedules/leaf/planning/<backbone>/budgets.json).
"""
import math

FACTOR = .97
DENSE_FRACTIONS = (.5, .56, .62, .68, .74, .8, .86)


def anchor_budgets(anchors, factor=FACTOR, extras=2):
    """anchors: {id: {ms, gmac}} -> [{id, latency_ms, gmac, anchor}], fastest first."""
    if not anchors: raise ValueError('no anchors')
    ordered = sorted(anchors.items(), key=lambda v: (v[1]['ms'], v[0]))
    for _, v in ordered:
        if not all(math.isfinite(v[k]) and v[k] > 0 for k in ('ms', 'gmac')):
            raise ValueError('invalid measured anchor')
    out = [dict(id=n, latency_ms=v['ms'] * factor, gmac=v['gmac'], anchor=n) for n, v in ordered]
    gap = ((out[-1]['latency_ms'] - out[0]['latency_ms']) / (len(out) - 1)
           if len(out) > 1 else .1 * out[0]['latency_ms'])
    for i in range(1, extras + 1):
        ms = out[0]['latency_ms'] - i * gap
        if ms <= 0: raise ValueError('extra budget must be positive')
        out.append(dict(id=f'extra{i}', latency_ms=ms, gmac=None, anchor=None))
    return sorted(out, key=lambda v: (v['latency_ms'], v['id']))


def dense_budgets(dense_ms, fractions=DENSE_FRACTIONS):
    """Fractions of the measured dense latency -> [{id, latency_ms, gmac, anchor}]."""
    if not math.isfinite(dense_ms) or dense_ms <= 0: raise ValueError('measured dense latency required')
    return [dict(id=f'dense{f:g}', latency_ms=dense_ms * f, gmac=None, anchor=None) for f in fractions]
