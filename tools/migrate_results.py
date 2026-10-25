"""Copy the raw campaign results of the paper from the research repository (efficient-vlm) into results/.

    python tools/migrate_results.py copy        # rsync every source into results/<experiment>/, then MANIFEST.json
    python tools/migrate_results.py relocate    # stored paths of the old machine -> repo-relative paths (JSON only)
    python tools/migrate_results.py rename-keys # arm / group / label keys and file names (tools/rename_map.py)
    python tools/migrate_results.py reseal      # recompute seals and recorded file hashes of the rewritten JSON
    python tools/migrate_results.py manifest    # (re)write MANIFEST.json: source sha256 and current sha256 per file
    python tools/migrate_results.py schedules   # the paper's schedule sets -> schedules/<name>/ (paths and keys renamed)
    python tools/migrate_results.py verify      # the files present match the sha256 recorded in MANIFEST.json

copy keeps the contents byte for byte and changes only folder names (no campaign codes, no date level); empty
task-claim files (claims/, *.lock) are not copied. relocate and the key renames (tools/rename_map.py) then
rewrite JSON text; MANIFEST.json keeps the sha256 of every source file next to the current one.
The research repository is only read.
"""
import argparse
import hashlib
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rename_map import apply_keys, original_name, path_replacements, renamed_file  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / 'results'
OLD = Path('/home/huynhvien/projects/efficient-vlm')
OLD_RESULTS = OLD / 'artifacts/results'
M0_WORKTREE = Path('/home/huynhvien/.codex-vien/worktrees/5d6e/efficient-vlm/artifacts/results')

# new folder under results/  <-  source folder (old campaign / date)
SOURCES = {
    # needed to rebuild the paper's tables and figures on the CPU
    'main_comparison':            OLD_RESULTS / 'q1_main_v2/2026-10-18',
    'main_comparison_analysis':   OLD_RESULTS / 'q1_main_r1/2026-10-20',
    'ablation_and_alpha_grid':    OLD_RESULTS / 'ablation_v3/2026-10-19',
    'alpha_kappa_equivalence':    OLD_RESULTS / 'alpha_equivalence/2026-10-19',
    'latency_pricing':            OLD_RESULTS / 'q1_latency_pricing/2026-10-18',
    'equal_gmac':                 OLD_RESULTS / 'q1_equal_gmac/2026-10-18',
    'imagenet_real':              OLD_RESULTS / 'q1_real/2026-10-18',
    'final_station_and_ci':       OLD_RESULTS / 'q1_review_cpu/2026-10-19',
    'forecast_approximation':     OLD_RESULTS / 'q1_forecast_approx/2026-10-19',
    'token_schedule':             OLD_RESULTS / 'q1_tokens/2026-10-18',
    'search_cost':                OLD_RESULTS / 'q1_logs/search_cost_throughput.json',
    # needed to re-run the campaigns (probes, teachers, SALT fits, earlier campaigns the main one builds on)
    'gmac_planning':              OLD_RESULTS / 'dp_m3_baselines/2026-10-16',
    'low_budget_and_factorial':   OLD_RESULTS / 'latency_d_m3_low_budget/2026-10-18',
    'horizon_norm_seed_ablation': OLD_RESULTS / 'latency_d_m3_extra_ablation/2026-10-18',
    'salt_fit':                   OLD_RESULTS / 't_proxy_comparison/2026-10-17',
    'salt_validation':            OLD_RESULTS / 't_proxy_validation/2026-10-17',
    'salt_m3_pilot':              OLD_RESULTS / 't_m3_pilot/2026-10-17',
    'merge_variants':             OLD_RESULTS / 'merge_variants/2026-10-16',
    # the earlier variant LEAF-M0 (its own campaign, outside the research repository)
    'leaf_m0':                    M0_WORKTREE / 'paper_final/2026-10-14',
    'leaf_m0_forecast_quality/deit_s': M0_WORKTREE / 'baselines_ablation_detectron/2026-10-15/deit_s/bs64/forecast_quality',
    'leaf_m0_forecast_quality/deit_b': M0_WORKTREE / 'baselines_ablation_detectron/2026-10-15/deit_b/bs32/forecast_quality',
    'leaf_m0_forecast_quality/mae_b':  M0_WORKTREE / 'baselines_ablation_detectron/2026-10-15/mae_b/bs32/forecast_quality',
}
EXCLUDE = ('claims/', '*.lock', '__pycache__/')


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''): h.update(block)
    return h.hexdigest()


def copy():
    for new, src in SOURCES.items():
        dst = RESULTS / new
        src = src.resolve()                                     # follows the dp_m3_baselines symlink
        if src.is_file():
            dst.mkdir(parents=True, exist_ok=True); target = dst / 'throughput.json' if new == 'search_cost' else dst / src.name
            subprocess.run(['rsync', '-a', str(src), str(target)], check=True)
        else:
            dst.mkdir(parents=True, exist_ok=True)
            cmd = ['rsync', '-a', '--delete'] + [f'--exclude={e}' for e in EXCLUDE] + [f'{src}/', f'{dst}/']
            subprocess.run(cmd, check=True)
        print(f'{new:34s} <- {src}', flush=True)
    manifest()


def _old_rel(rel):
    top, _, rest = rel.partition('/')
    return '/'.join([top] + [original_name(x) for x in rest.split('/')]) if rest else rel


def manifest():
    """MANIFEST.json: for every copied file its source path, the sha256 of the source and the current sha256."""
    old = {}
    if (RESULTS / 'MANIFEST.json').exists():
        old = json.loads((RESULTS / 'MANIFEST.json').read_text())['files']
    rows, sources = {}, {}
    for new, src in SOURCES.items():
        src = src.resolve(); sources[new] = str(src)
        base = RESULTS / new
        files = [base / ('throughput.json' if new == 'search_cost' else src.name)] if src.is_file() else \
            sorted(p for p in base.rglob('*') if p.is_file() and not p.is_symlink())
        for p in files:
            rel = p.relative_to(RESULTS).as_posix()
            sub = Path(*[original_name(part) for part in p.relative_to(base).parts]) if not src.is_file() else None
            origin = src if src.is_file() else src / sub
            rows[rel] = dict(source=str(origin))
    with ThreadPoolExecutor(16) as pool:
        for rel, digest in zip(rows, pool.map(lambda r: sha256(RESULTS / r), list(rows))):
            prev = old.get(rel) or old.get(_old_rel(rel), {})
            origin = Path(rows[rel]['source'])
            rows[rel]['source_sha256'] = prev.get('source_sha256') or (sha256(origin) if origin.exists() else digest)
            rows[rel]['sha256'] = digest
    doc = dict(note='raw results copied from the research repository; folder names changed, stored paths and keys '
                    'renamed (tools/rename_map.py); source_sha256 = the original file, sha256 = the file now',
               sources=sources, files=rows)
    (RESULTS / 'MANIFEST.json').write_text(json.dumps(doc, indent=1) + '\n')
    summary = {}
    for rel, row in rows.items():
        top = rel.split('/')[0]; e = summary.setdefault(top, dict(files=0, h=hashlib.sha256()))
        e['files'] += 1; e['h'].update(f"{rel} {row['sha256']}\n".encode())
    (RESULTS / 'SUMMARY.json').write_text(json.dumps(
        {k: dict(files=v['files'], digest=v['h'].hexdigest(), source=sources.get(k, '')) for k, v in sorted(summary.items())},
        indent=1) + '\n')
    print(f'MANIFEST.json: {len(rows)} files')


def relocate():
    """Rewrite paths of the old machine stored in JSON results into repo-relative paths (tools/rename_map.py)."""
    pairs = path_replacements(); changed = 0
    files = [p for p in RESULTS.rglob('*.json') if p.name not in ('MANIFEST.json', 'SUMMARY.json')]
    def one(p):
        text = p.read_text(errors='surrogateescape'); new = text
        for old, rep in pairs:
            if old in new: new = new.replace(old, rep)
        if new != text: p.write_text(new, errors='surrogateescape'); return 1
        return 0
    with ThreadPoolExecutor(16) as pool: changed = sum(pool.map(one, files))
    print(f'relocate: {changed} of {len(files)} JSON files rewritten')


KEEP_KEYS = ('leaf_m0', 'leaf_m0_forecast_quality')    # the LEAF-M0 campaign keeps its names (its code is kept verbatim)


def rename_keys():
    """Rewrite the internal keys in JSON results and rename the files / folders named after them."""
    tops = [d for d in RESULTS.iterdir() if d.is_dir() and d.name not in KEEP_KEYS]
    files = [p for d in tops for p in d.rglob('*.json')]
    def one(p):
        text = p.read_text(errors='surrogateescape'); new = apply_keys(text)
        if new != text: p.write_text(new, errors='surrogateescape'); return 1
        return 0
    with ThreadPoolExecutor(16) as pool: changed = sum(pool.map(one, files))
    moved = 0
    for d in tops:
        for p in sorted(d.rglob('*'), key=lambda q: -len(q.parts)):      # deepest first
            new = renamed_file(p.name)
            if new != p.name:
                target = p.with_name(new)
                if target.exists(): raise SystemExit(f'rename target exists: {target}')
                p.rename(target); moved += 1
    print(f'rename-keys: {changed} JSON files rewritten, {moved} files / folders renamed')


HEX64 = re.compile(r'\b[0-9a-f]{64}\b')


def _sha_obj(obj): return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()
def _sha_configs(jobs): return hashlib.sha256(json.dumps({n: v['config'] for n, v in jobs.items()}, sort_keys=True).encode()).hexdigest()


SCHEMES = {   # name -> (applies, value of the seal) for the seal formats used by the campaigns
    'doc_sha256': (lambda d: isinstance(d.get('sha256'), str), lambda d: _sha_obj({k: v for k, v in d.items() if k != 'sha256'}), 'sha256'),
    'jobs_sha': (lambda d: isinstance(d.get('sha'), str) and isinstance(d.get('jobs'), dict), lambda d: _sha_obj(d['jobs']), 'sha'),
    'jobs_configs': (lambda d: isinstance(d.get('sha256'), str) and isinstance(d.get('jobs'), dict), lambda d: _sha_configs(d['jobs']), 'sha256'),
    'rule_sha256': (lambda d: isinstance(d.get('sha256'), str) and isinstance(d.get('rule'), dict), lambda d: _sha_obj(d['rule']), 'sha256'),
}


def _scheme(original):
    try: d = json.loads(original)
    except ValueError: return None
    if not isinstance(d, dict): return None
    for name, (applies, value, field) in SCHEMES.items():
        try:
            if applies(d) and value(d) == d[field]: return name
        except (KeyError, TypeError, AttributeError): pass
    return None


def reseal():
    """After relocate / rename-keys: (1) recorded sha256 of files that changed are replaced by their new sha256
    everywhere; (2) seals inside rewritten documents are recomputed with the scheme that validates the original
    file; repeated until nothing changes. The original values stay available through MANIFEST source_sha256."""
    doc = json.loads((RESULTS / 'MANIFEST.json').read_text())['files']
    src = {rel: row['source'] for rel, row in doc.items()}
    current = {rel: row['sha256'] for rel, row in doc.items()}
    original_sha = {rel: row['source_sha256'] for rel, row in doc.items()}
    rels = [rel for rel in doc if rel.endswith('.json')]
    schemes = {}
    for rel in rels:
        if current[rel] != original_sha[rel]:
            try: schemes[rel] = _scheme(Path(src[rel]).read_text(errors='surrogateescape'))
            except OSError: schemes[rel] = None
    print('sealed documents among the rewritten files:', {k: sum(v == k for v in schemes.values()) for k in SCHEMES})
    remap = {original_sha[r]: current[r] for r in doc if current[r] != original_sha[r]}
    for it in range(10):
        changed = 0
        def one(rel):
            p = RESULTS / rel; text = p.read_text(errors='surrogateescape'); new = text
            if remap and HEX64.search(new): new = HEX64.sub(lambda m: remap.get(m.group(0), m.group(0)), new)
            name = schemes.get(rel)
            if name:
                try:
                    d = json.loads(new); applies, value, field = SCHEMES[name]
                    if d.get(field) != value(d):
                        d[field] = value(d); new = json.dumps(d, indent=1) + '\n'
                except ValueError: pass
            if new != text: p.write_text(new, errors='surrogateescape'); return rel
            return None
        with ThreadPoolExecutor(16) as pool: touched = [r for r in pool.map(one, rels) if r]
        for rel in touched:
            new_sha = sha256(RESULTS / rel)
            remap[current[rel]] = new_sha
            if original_sha[rel] in remap: remap[original_sha[rel]] = new_sha
            if rel not in schemes:
                try: schemes[rel] = _scheme(Path(src[rel]).read_text(errors='surrogateescape'))
                except OSError: schemes[rel] = None
            current[rel] = new_sha
        print(f'reseal pass {it + 1}: {len(touched)} files rewritten', flush=True)
        if not touched: break


SCHEDULE_SETS = {   # schedules/<new>  <-  efficient-vlm schedules/<old>  (the LEAF release, schedules/leaf, is written separately)
    'main_comparison': 'q1_main_v2/2026-10-18',
    'gmac_planned': 'dp_m3_baselines',
    'latency_pricing': 'q1_latency_pricing/2026-10-18',
    'low_budget_and_factorial': 'latency_d_m3_low_budget/2026-10-18',
    'horizon_norm_seed_ablation': 'latency_d_m3_extra_ablation/2026-10-18',
    'merge_variants': 'merge_variants',
    'learned_theta_alpha': 'ste_theta_alpha',
    'learned_theta_alpha_m3': 'ste_theta_alpha_m3',
    'learned_theta_alpha_followup': 'ste_evaluation_followup',
    'leaf_m0': 'paper_final',
    'alpha_form': 'alpha',
}


def schedules():
    """Copy the schedule sets of the paper's campaigns into schedules/ with the path and key renames."""
    dst_root = REPO / 'schedules'
    for new, old in SCHEDULE_SETS.items():
        src = OLD / 'schedules' / old
        dst = dst_root / new
        subprocess.run(['rsync', '-a', '--delete', '--exclude=__pycache__', f'{src}/', f'{dst}/'], check=True)
        n = 0
        for f in dst.rglob('*.json'):
            text = f.read_text(); t = text
            for a, b in path_replacements(): t = t.replace(a, b)
            if new != 'leaf_m0': t = apply_keys(t)
            if t != text: f.write_text(t); n += 1
        for f in sorted(dst.rglob('*'), key=lambda q: -len(q.parts)):
            if new != 'leaf_m0' and renamed_file(f.name) != f.name: f.rename(f.with_name(renamed_file(f.name)))
        print(f'schedules/{new:30s} <- schedules/{old} ({n} files rewritten)')


def verify():
    """Check the files present under results/ against MANIFEST.json. Files of a bundle that was not downloaded are
    reported as missing; only a file whose content differs is an error."""
    doc = json.loads((RESULTS / 'MANIFEST.json').read_text()); missing = bad = 0
    for rel, row in doc['files'].items():
        p = RESULTS / rel
        if not p.exists(): missing += 1; continue
        if sha256(p) != row['sha256']: bad += 1; print('MISMATCH', rel)
    n = len(doc['files'])
    print(f'{n - missing - bad}/{n} files match, {missing} not present (bundle not downloaded), {bad} differ')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('action', choices=('copy', 'relocate', 'rename-keys', 'reseal', 'manifest', 'schedules', 'verify'))
    {'copy': copy, 'relocate': relocate, 'rename-keys': rename_keys, 'reseal': reseal, 'manifest': manifest,
     'schedules': schedules, 'verify': verify}[ap.parse_args().action]()
