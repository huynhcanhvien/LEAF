"""Token ancestry of a LEAF schedule (M3 operator) on fixed ImageNet-val images, for the paper figure.

python leaf/reproduction/experiments/token_schedule.py --backbone deit_b --schedule ref/M3/gmac/DR11.5 --gpu 0

Wraps leaf.model.mlerp_vit.compress_norm (the production M3 station) to track which original patches each surviving token
contains, and checks that the traced logits equal the normal forward bit for bit. Four validation images are fixed
for visual clarity before tracing; nothing is selected by prediction or score.
Writes results/figures/tokens_<bb>_<schedule>.{png,pdf} and results/token_schedule/<...>.json.
Use --render-only to redraw the saved lineage on the CPU without running inference.
Rendering follows the region-mean fill and coloured group-boundary convention in
https://github.com/OpenGVLab/DiffRate/blob/master/DiffRate/vis.py (originally ToMe).
Only surviving representative positions are drawn as token tiles. Merged-away
source positions are blank; pruned ancestry is gray. A merged summary token uses
the group's mean image colour and a coloured border, not a decoded feature vector.
"""
import argparse
import colorsys
import json
import os
import shutil
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
import numpy as np
import torch

EXAMPLES = [(281, 'Tabby cat', None), (386, 'African elephant', 3982),
            (817, 'Sports car', 4748), (985, 'Daisy', None)]
# 2026-10-19: images the GMAC-planned ~10-GMAC DeiT-B schedule classifies correctly (stored per-image correctness of the
# main campaign; 3982 is misclassified by it and by the dense model, replaced by 7473, chosen for clarity among correct ones)
EXAMPLE_SETS = dict(default=EXAMPLES, correct10=[(281, 'Tabby cat', 67), (386, 'African elephant', 7473),
                                                 (817, 'Sports car', 4748), (985, 'Daisy', 1131)])
# 2026-10-19: chosen by eye from the --scan contact sheets (119 correctly classified images of 20 classes) for a clear object
# that the schedule keeps largely intact; disclosed as such in the caption
EXAMPLE_SETS['object10'] = [(31, 'Tree frog', 1499), (323, 'Monarch butterfly', 382), (779, 'School bus', 6843), (437, 'Lighthouse', 1309)]
FIG = REPO / 'results/figures'
OUT = REPO / 'results/token_schedule'


@torch.no_grad()
def trace(model, image, n_patches):
    import leaf.model.mlerp_vit as pdm
    original = pdm.compress_norm
    members = [[]] + [[i] for i in range(n_patches)]
    reps = [-1] + list(range(n_patches))
    snaps = []

    def tracked(x, size, scores, n_prune, n_merge, mode, fuse_merge=True):
        nonlocal members, reps
        out, newsize, order, destination = original(x, size, scores, n_prune, n_merge, mode, fuse_merge)
        idx = order[0].cpu().tolist(); n = out.shape[1]
        gathered = [members[i][:] for i in idx]
        new = [m[:] for m in gathered[:n]]; newreps = [reps[i] for i in idx[:n]]
        merged = 0
        if destination is not None:
            for src, dst in enumerate(destination[0].cpu().tolist(), start=n):
                assert dst > 0; new[dst].extend(gathered[src]); merged += len(gathered[src])
        owner = np.full(n_patches, -1)
        for t, g in enumerate(new[1:], start=1): owner[g] = t
        snaps.append(dict(block=model.stations[len(snaps)], patch_tokens=n - 1, owner=owner.tolist(), representatives=newreps[1:],
                          in_ancestry=int((owner >= 0).sum()), merged_sources=len(idx) - n, pruned=x.shape[1] - len(idx)))
        members, reps = new, newreps
        return out, newsize, order, destination

    pdm.compress_norm = tracked
    try:
        with torch.autocast('cuda', dtype=torch.bfloat16): logits = model(image).float()
    finally:
        pdm.compress_norm = original
    with torch.autocast('cuda', dtype=torch.bfloat16): ref = model(image).float()
    assert torch.equal(logits, ref), 'tracing changed the forward'
    return snaps, int(logits.argmax(-1))


def token_layout(snap, grid=14):
    """0 = pruned ancestry, 1 = merged-away source, 2 = live token position."""
    owner = np.asarray(snap['owner']).reshape(grid, grid)
    reps = np.asarray(snap['representatives'])
    assert len(reps) == len(np.unique(reps)) == snap['patch_tokens']
    assert set(np.unique(owner[owner >= 0])) == set(range(1, snap['patch_tokens'] + 1))
    assert np.array_equal(owner.flat[reps], np.arange(1, len(reps) + 1))
    state = np.where(owner >= 0, 1, 0)
    state.flat[reps] = 2
    assert int((state == 2).sum()) == snap['patch_tokens']
    return owner, state


def paint(img, snap, grid=14, patch=16):
    """Exactly one visible tile per live token; no tiles at merged source sites."""
    src = img.astype(float) / 255
    assert src.shape == (grid * patch, grid * patch, 3)
    owner, state = token_layout(snap, grid)
    # Background colours are reserved for removed positions and checked below.
    out = np.full_like(src, .99)
    pruned = np.repeat(np.repeat(state == 0, patch, axis=0), patch, axis=1)
    out[pruned] = .86
    for token, rep in enumerate(snap['representatives'], start=1):
        y, x = divmod(rep, grid)
        sy, sx = slice(y * patch, (y + 1) * patch), slice(x * patch, (x + 1) * patch)
        group = np.repeat(np.repeat(owner == token, patch, axis=0), patch, axis=1)
        colour = colorsys.hsv_to_rgb((rep * .61803398875 + .09) % 1., .65, .82)
        tile = src[sy, sx].copy()
        merged = int(np.count_nonzero(owner == token)) > 1
        if merged:
            tile[:] = src[group].mean(axis=0)
        border = 2 if merged else 1
        edge = colour if merged else (.25, .25, .25)
        tile[:border] = tile[-border:] = edge
        tile[:, :border] = tile[:, -border:] = edge
        out[sy, sx] = tile
    # Verify the actual raster, including every pixel of removed patch sites.
    tiles = out.reshape(grid, patch, grid, patch, 3).transpose(0, 2, 1, 3, 4)
    assert np.all(tiles[state == 0] == .86), 'pruned positions were painted'
    assert np.all(tiles[state == 1] == .99), 'merged-away source positions were painted'
    drawn = ~np.all(tiles == .99, axis=(2, 3, 4)) & ~np.all(tiles == .86, axis=(2, 3, 4))
    assert np.array_equal(drawn, state == 2), 'visible tiles do not match live tokens'
    assert int(drawn.sum()) == snap['patch_tokens']
    return out


def paint_diffrate(img, snap, grid=14, patch=16, scale=2, edge=2, singleton='mean'):
    """DiffRate/ToMe convention (2026-10-19): every token's group of source patches is filled with the group's mean
    image colour and outlined by the eroded-mask edge in a per-token colour; pruned patches are not drawn (black).
    Colours follow the token's representative patch, so a group keeps its colour across stations."""
    from scipy.ndimage import binary_erosion
    src = np.kron(img.astype(float) / 255, np.ones((scale, scale, 1)))
    owner, state = token_layout(snap, grid)
    out = np.zeros_like(src)
    up = patch * scale
    rng = np.random.default_rng(0); palette = rng.uniform(.15, 1., size=(grid * grid, 3))
    drawn = 0
    for token, rep in enumerate(snap['representatives'], start=1):
        mask = np.kron(owner == token, np.ones((up, up), dtype=bool))
        if not mask.any(): continue
        ring = mask & ~binary_erosion(mask, iterations=edge)
        if singleton == 'image' and int(np.count_nonzero(owner == token)) == 1:   # unmerged token: its own pixels, thin dark edge
            out[mask] = src[mask]; out[mask & ~binary_erosion(mask, iterations=1)] = .3
        else:
            out[mask] = src[mask].mean(axis=0); out[ring] = palette[rep]
        drawn += 1
    pruned = np.kron(owner < 0, np.ones((up, up), dtype=bool))
    assert np.all(out[pruned] == 0), 'pruned positions were painted'
    assert drawn == snap['patch_tokens'] == len(np.unique(owner[owner >= 0]))
    return out


def paint_black(img, snap, grid=14, patch=16, scale=2, edge=2):
    """One tile per live token at its representative position; every other position (pruned, or merged into another
    token) is black, so the black area grows at every station and the number of tiles equals the token count (2026-10-19).
    Unmerged tokens show their patch with a thin dark edge; merged tokens show the mean colour of all their source
    patches inside a coloured edge (colour follows the representative patch, stable across stations)."""
    src = np.kron(img.astype(float) / 255, np.ones((scale, scale, 1)))
    owner, state = token_layout(snap, grid)
    out = np.zeros_like(src); up = patch * scale
    rng = np.random.default_rng(0); palette = rng.uniform(.15, 1., size=(grid * grid, 3))
    for token, rep in enumerate(snap['representatives'], start=1):
        y, x = divmod(rep, grid); sy, sx = slice(y * up, (y + 1) * up), slice(x * up, (x + 1) * up)
        group = np.kron(owner == token, np.ones((up, up), dtype=bool))
        merged = int(np.count_nonzero(owner == token)) > 1
        tile = np.repeat(np.repeat(src[group].mean(axis=0)[None, None], up, 0), up, 1) if merged else src[sy, sx].copy()
        w, col = (edge, palette[rep]) if merged else (1, (.3, .3, .3))
        tile[:w] = tile[-w:] = col; tile[:, :w] = tile[:, -w:] = col
        out[sy, sx] = tile
    tiles = out.reshape(grid, up, grid, up, 3).transpose(0, 2, 1, 3, 4)
    assert np.all(tiles[state != 2] == 0), 'a removed position was painted'
    assert int((~np.all(tiles == 0, axis=(2, 3, 4))).sum()) == snap['patch_tokens']
    return out


def validate_lineage(rows, stations):
    """Check partitions by membership, invariant to token sorting/renumbering."""
    diagnostics = []
    for row in rows:
        assert [s['block'] for s in row['snapshots']] == [s['block'] for s in stations]
        previous = None
        for snap, station in zip(row['snapshots'], stations):
            owner = np.asarray(snap['owner'])
            count = np.bincount(owner[owner >= 0], minlength=snap['patch_tokens'] + 1)[1:]
            assert len(count) == snap['patch_tokens'] and np.all(count > 0)
            assert snap['patch_tokens'] + 1 == station['tokens_after']
            if previous is not None:
                assert np.all(owner[previous < 0] < 0), 'a pruned patch reappeared'
                for token in np.unique(previous[previous >= 0]):
                    assert len(np.unique(owner[previous == token])) == 1, 'a previous token was split'
                if station['pruned'] == 0:
                    assert np.array_equal(owner >= 0, previous >= 0), 'merge-only station lost ancestry'
            diagnostics.append(dict(image_index=row['image_index'], block=snap['block'],
                                    token_count=len(count), original_patches=int(count.sum()),
                                    visible_token_tiles=int((token_layout(snap)[1] == 2).sum()),
                                    blank_merged_positions=int((token_layout(snap)[1] == 1).sum()),
                                    gray_pruned_positions=int((token_layout(snap)[1] == 0).sum()),
                                    merged_groups=int((count > 1).sum()),
                                    coassigned_patch_pairs=int((count * (count - 1) // 2).sum())))
            previous = owner
    return diagnostics


def render(images, rows, stem, title, include_final=False, style='tiles'):
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/efficient-vlm-matplotlib')
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42})
    displayed = [row['snapshots'] if include_final else row['snapshots'][:-1] for row in rows]
    cols = 1 + len(displayed[0])
    blocks = [s['block'] for s in displayed[0]]
    assert blocks and all([s['block'] for s in snaps] == blocks for snaps in displayed)
    fig, axes = plt.subplots(len(rows), cols, figsize=(cols * 2.45, len(rows) * 2.45), squeeze=False)
    fig.subplots_adjust(left=.08, right=.995, bottom=.13, top=.925, wspace=.055, hspace=.19)
    for r, row in enumerate(rows):
        axes[r, 0].imshow(images[row['image_index']], interpolation='nearest')
        axes[r, 0].set_ylabel(row['class_name'], fontsize=14, labelpad=12)
        if r == 0: axes[r, 0].set_title('Input\n196 patches', fontsize=14, linespacing=1.5, pad=10)
        for c, s in enumerate(displayed[r], start=1):
            pic = paint_black(images[row['image_index']], s) if style == 'tiles_black' else paint_diffrate(images[row['image_index']], s) if style == 'diffrate' else paint_diffrate(images[row['image_index']], s, singleton='image') if style == 'diffrate_image' else paint(images[row['image_index']], s)
            ax = axes[r, c]; ax.imshow(pic, interpolation='nearest')
            owner = np.asarray(s['owner'])
            count = np.bincount(owner[owner >= 0])[1:]
            if style not in ('diffrate', 'diffrate_image'): ax.set_xlabel(f"{s['patch_tokens']} visible tokens", fontsize=12, labelpad=4)
            if r == 0:
                ax.set_title(f"Block {s['block']}\n{s['patch_tokens']} patch tokens",
                             fontsize=14, linespacing=1.5, pad=10)
        for ax in axes[r]:
            ax.set_xticks([]); ax.set_yticks([]); [sp.set_visible(False) for sp in ax.spines.values()]
    if style == 'tiles_black':
        fig.legend(handles=[Patch(facecolor='#a98d6c', edgecolor='.3', label='Unmerged token'),
                            Patch(facecolor='#a98d6c', edgecolor='#287eae', linewidth=2, label='Merged token (mean colour of its source patches)'),
                            Patch(facecolor='black', edgecolor='black', label='Removed (pruned or merged away)')],
                   loc='lower center', bbox_to_anchor=(.54, .005), ncol=3, frameon=False, fontsize=14)
    elif style == 'diffrate_image':
        fig.legend(handles=[Patch(facecolor='#a98d6c', edgecolor='.3', label='Unmerged token (its patch)'),
                            Patch(facecolor='#a98d6c', edgecolor='#287eae', linewidth=2, label='Merged token (its source patches, mean colour)'),
                            Patch(facecolor='black', edgecolor='black', label='Pruned')],
                   loc='lower center', bbox_to_anchor=(.54, .005), ncol=3, frameon=False, fontsize=14)
    elif style == 'diffrate':
        fig.legend(handles=[Patch(facecolor='#a98d6c', edgecolor='#287eae', linewidth=2, label='One token: its source patches, mean colour, outlined'),
                            Patch(facecolor='black', edgecolor='black', label='Pruned')],
                   loc='lower center', bbox_to_anchor=(.54, .005), ncol=2, frameon=False, fontsize=14)
    else:
      fig.legend(handles=[Patch(facecolor='#a98d6c', edgecolor='#287eae', linewidth=2, label='Merged summary token'),
                        Patch(facecolor='#a98d6c', edgecolor='.25', label='Unmerged token'),
                        Patch(facecolor='.99', edgecolor='.8', label='Blank: merged-away source'),
                        Patch(facecolor='.86', edgecolor='.7', label='Gray: pruned')],
               loc='lower center', bbox_to_anchor=(.54, .005), ncol=2, frameon=False, fontsize=14)
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'): fig.savefig(f'{stem}.{ext}', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    for ext in ('png', 'pdf'):
        shutil.copy2(f'{stem}.{ext}', REPO / 'results/paper_build/figures' / f'{stem.name}.{ext}')


def get_schedule(backbone, name):
    exported = json.loads((REPO / f'schedules/kappa_form/main_comparison/{backbone}.json').read_text())['schedules']
    if name in exported:
        return exported[name]
    job = json.loads((REPO / f'results/main_comparison/{backbone}/plans.json').read_text())['jobs'][name]
    stations = [dict(block=r['block'], kappa=r['kappa'], merged=r['merged'], pruned=r['pruned'],
                     theta=r['theta'], tokens_after=r['merge_keep'])
                for r in job['details']['token_path'] if r['removed'] > 0]
    return dict(config=job['config'], config_sha=job['config_sha'], gmac=job['gmac'], stations=stations)


def scan(a):
    """Trace the first `--per-class` correctly classified val images (stored correctness of the main campaign) of each class in
    `--scan` and draw a contact sheet (input | last displayed station, tiles_black) to choose figure examples (2026-10-19)."""
    os.environ['CUDA_VISIBLE_DEVICES'] = a.gpu
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/efficient-vlm-matplotlib')
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from leaf.evaluation.data import iterate, load_split
    from leaf.backbones import load
    from leaf.reproduction.experiments.low_budget_and_factorial import build
    from leaf.reproduction.experiments.campaign_reports import correct, MAIN_COMPARISON
    from leaf.evaluation.runner import require_idle_gpu
    import leaf.evaluation.determinism
    require_idle_gpu(); torch.set_num_threads(4)
    sched = get_schedule(a.backbone, a.schedule)
    vit, spec = load(a.backbone); model = build(vit, spec, sched['config']).eval()
    images, labels = load_split('val')
    ok = correct(MAIN_COMPARISON / a.backbone, sched['config_sha'], 'val')
    rows = []
    for cls in [int(c) for c in a.scan.split(',')]:
        for idx in [int(i) for i in np.flatnonzero((labels == cls) & ok)][:a.per_class]:
            x, _ = next(iterate(images[idx:idx + 1], labels[idx:idx + 1], 1, 'cuda', spec.mean, spec.std))
            snaps, pred = trace(model, x, spec.n_tokens - 1)
            if pred != cls: print('skip (singleton trace disagrees with batched evaluation):', idx, pred, cls); continue
            rows.append(dict(image_index=idx, label=cls, prediction=pred, snapshots=snaps))
    tag = f"{a.backbone}_{a.schedule.replace('/', '_')}"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'scan_{tag}.json').write_text(json.dumps(dict(schedule=a.schedule, config_sha=sched['config_sha'], rows=rows)))
    per_sheet = 24
    for k in range(0, len(rows), per_sheet):
        chunk = rows[k:k + per_sheet]; cols = 6
        fig, axes = plt.subplots((len(chunk) + cols - 1) // cols, 2 * cols, figsize=(2 * cols * 1.6, 1.75 * ((len(chunk) + cols - 1) // cols)), squeeze=False)
        for ax in axes.flat: ax.axis('off')
        for i, r in enumerate(chunk):
            a0, a1 = axes[i // cols, 2 * (i % cols)], axes[i // cols, 2 * (i % cols) + 1]
            a0.imshow(images[r['image_index']]); a0.set_title(f"{r['image_index']} (c{r['label']})", fontsize=7)
            a1.imshow(paint_black(images[r['image_index']], r['snapshots'][-2]))
        fig.tight_layout(); fig.savefig(OUT / f'scan_{tag}_{k // per_sheet}.png', dpi=90); plt.close(fig)
    print('scanned', len(rows), 'images; sheets in', OUT)


def main(a):
    if a.scan: return scan(a)
    tag = f"{a.backbone}_{a.schedule.replace('/', '_')}"
    if a.render_only:
        from leaf.evaluation.data import load_split
        saved = OUT / f'{tag}.json'
        if not saved.exists(): saved = REPO / f'results/paper_build/data/tokens_{tag}.json'
        record = json.loads(saved.read_text())
        schedule = get_schedule(a.backbone, a.schedule)
        assert record['config_sha'] == schedule['config_sha']
        assert all([s['block'] for s in r['snapshots']] == [s['block'] for s in record['stations']]
                   for r in record['examples']), 'every station must be shown'
        images, labels = load_split('val')
        assert all(int(labels[r['image_index']]) == r['label'] for r in record['examples'])
        audit = validate_lineage(record['examples'], record['stations'])
        (REPO / f'results/paper_build/data/tokens_{tag}_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
        render(images, record['examples'], FIG / f'tokens_{tag}', '', a.include_final_station, a.style)
        print('redrew verified cached lineage on CPU; final station displayed:', a.include_final_station)
        return
    os.environ['CUDA_VISIBLE_DEVICES'] = a.gpu
    from leaf.evaluation.data import iterate, load_split
    from leaf.backbones import load, SPECS
    from leaf.reproduction.experiments.low_budget_and_factorial import build
    from leaf.evaluation.runner import require_idle_gpu
    import leaf.evaluation.determinism
    require_idle_gpu()
    torch.set_num_threads(4)
    spec = SPECS[a.backbone]
    sched = get_schedule(a.backbone, a.schedule)
    vit, spec = load(a.backbone); model = build(vit, spec, sched['config']).eval()
    images, labels = load_split('val')
    # Lineage tracking is defined for one image at a time. Profile that exact
    # task, including its untraced reference pass, before collecting examples.
    first = int(np.flatnonzero(labels == EXAMPLE_SETS[a.examples][0][0])[0])
    x, _ = next(iterate(images[first:first + 1], labels[first:first + 1], 1, 'cuda', spec.mean, spec.std))
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    trace(model, x, spec.n_tokens - 1)
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    profile = dict(batch_size=1, seconds=elapsed, images_per_second=1 / elapsed,
                   peak_gib=torch.cuda.max_memory_allocated() / 2**30,
                   reason='singleton lineage trace plus bitwise comparison with singleton production forward')
    assert profile['peak_gib'] < 14, profile
    print('trace profile:', profile, flush=True)
    rows = []
    for cls, name, fixed_index in EXAMPLE_SETS[a.examples]:
        idx = int(np.flatnonzero(labels == cls)[0]) if fixed_index is None else fixed_index
        assert int(labels[idx]) == cls
        x, _ = next(iterate(images[idx:idx + 1], labels[idx:idx + 1], 1, 'cuda', spec.mean, spec.std))
        snaps, pred = trace(model, x, spec.n_tokens - 1)
        rows.append(dict(image_index=idx, class_name=name, label=int(labels[idx]), prediction=pred, snapshots=snaps))
        if a.require_correct: assert pred == int(labels[idx]), f'{name} ({idx}) misclassified by the traced schedule'
    tag = f"{a.backbone}_{a.schedule.replace('/', '_')}"
    assert all([s['block'] for s in row['snapshots']] == model.stations for row in rows)
    validate_lineage(rows, sched['stations'])
    render(images, rows, FIG / f'tokens_{tag}', '', a.include_final_station, a.style)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'{tag}.json').write_text(json.dumps(dict(backbone=a.backbone, schedule=a.schedule, config_sha=sched['config_sha'],
                                                     gmac=sched['gmac'], profile=profile,
                                                     stations=sched['stations'], examples=rows), indent=1))
    shutil.copy2(OUT / f'{tag}.json', REPO / f'results/paper_build/data/tokens_{tag}.json')
    print('wrote', FIG / f'tokens_{tag}.png', [(r['class_name'], r['label'], r['prediction']) for r in rows])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--backbone', default='deit_b'); ap.add_argument('--schedule', default='ref/M3/gmac/DR11.5')
    ap.add_argument('--gpu', default='0', choices=['0', '1'])
    ap.add_argument('--render-only', action='store_true', help='redraw the cached lineage on CPU')
    ap.add_argument('--include-final-station', action='store_true', help='also display the final station (always executed)')
    ap.add_argument('--examples', default='default', choices=sorted(EXAMPLE_SETS), help='fixed image set (2026-10-19)')
    ap.add_argument('--style', default='tiles', choices=['tiles', 'tiles_black', 'diffrate', 'diffrate_image'], help='tiles: one tile per live token; diffrate: DiffRate/ToMe group fill; diffrate_image: same, unmerged tokens keep their pixels (2026-10-19)')
    ap.add_argument('--scan', default='', help='comma-separated classes: trace correct images and draw contact sheets (2026-10-19)')
    ap.add_argument('--per-class', type=int, default=6)
    ap.add_argument('--require-correct', action='store_true', help='fail if the traced schedule misclassifies an example')
    main(ap.parse_args())
