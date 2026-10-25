"""Write the alpha form of every schedule set that the campaigns wrote with the forecast weight k (CPU only).

The campaigns planned and evaluated with the score z(log P) + k z(D). Their schedule exports are kept unchanged in
schedules/kappa_form/<set>/ (the experiment code reads them there), and the sealed plans in
results/main_comparison/<bb>/plans.json. This script reads them and writes:
  schedules/<set>/<bb>.json                   same document with every 'pd'/'pdm'/'extra_pdm' config turned into
  schedules/main_comparison_plans/<bb>.json   'pda' (alpha = k/(1+k), leaf/model/alpha_score.py), station rows
                                              'kappa' -> 'alpha', 'config_sha' recomputed and the old one kept as
                                              'source_config_sha';
  schedules/kappa_form/conversion_manifest.json   source, sha256 and counts per file.
Usage: python leaf/reproduction/experiments/alpha_kappa_conversion.py [--dry-run]
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from leaf.model.alpha_score import BASES, config_to_alpha, kappa_to_alpha

KAPPA = REPO / 'schedules/kappa_form'
MANIFEST = KAPPA / 'conversion_manifest.json'
SOURCES = ('schedules/kappa_form/*/*.json', 'results/main_comparison/*/plans.json')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources():
    return [p for pattern in SOURCES for p in sorted(REPO.glob(pattern))]


def target(src):
    """schedules/kappa_form/<set>/<bb>.json -> schedules/<set>/<bb>.json; results/main_comparison/<bb>/plans.json ->
    schedules/main_comparison_plans/<bb>.json."""
    if KAPPA in src.parents: return REPO / 'schedules' / src.relative_to(KAPPA)
    return REPO / 'schedules/main_comparison_plans' / f'{src.parent.name}.json'


def convert(obj, stats):
    """Recursive copy with alpha configs; stats counts converted configs/stations and unconverted kinds."""
    from leaf.evaluation.runner import config_hash
    if isinstance(obj, list): return [convert(v, stats) for v in obj]
    if not isinstance(obj, dict): return obj
    if obj.get('kind') in BASES and 'k' in obj and 'prune_keep' in obj:
        stats['configs'] += 1
        return config_to_alpha(obj)
    if 'kind' in obj and 'prune_keep' in obj: stats['kind:' + str(obj['kind'])] += 1
    out = {}
    for k, v in obj.items():
        if k == 'kappa' and isinstance(v, (int, float)):
            out['alpha'] = kappa_to_alpha(v); stats['stations'] += 1
        else:
            out[k] = convert(v, stats)
    if isinstance(obj.get('config'), dict) and out['config'] != obj['config'] and 'config_sha' in obj:
        out['source_config_sha'] = obj['config_sha']; out['config_sha'] = config_hash(out['config'])
    return out


def has_k(obj):
    if isinstance(obj, list): return any(has_k(v) for v in obj)
    if not isinstance(obj, dict): return False
    if obj.get('kind') in BASES and 'k' in obj: return True
    return 'kappa' in obj or any(has_k(v) for v in obj.values())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--dry-run', action='store_true'); a = ap.parse_args()
    manifest = dict(rule='alpha = k/(1+k); score (1-alpha) z(log P) + alpha z(D) gives the legacy token order',
                    files=[])
    for src in sources():
        doc = json.loads(src.read_text())
        if not has_k(doc): continue
        rel = src.relative_to(REPO)
        stats = Counter(); new = convert(doc, stats)
        out = target(src)
        row = dict(source=str(rel), sha256=sha256(src), alpha=str(out.relative_to(REPO)), **dict(stats))
        manifest['files'].append(row)
        print(rel, dict(stats))
        if a.dry_run: continue
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(new, indent=1) + '\n')
    total = Counter()
    for r in manifest['files']:
        total.update({k: v for k, v in r.items() if isinstance(v, int)})
    manifest['totals'] = dict(files=len(manifest['files']), **dict(total))
    print(manifest['totals'])
    if not a.dry_run:
        MANIFEST.write_text(json.dumps(manifest, indent=1) + '\n')


if __name__ == '__main__':
    main()
