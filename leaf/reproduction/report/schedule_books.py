"""Schedule book: every schedule the paper evaluated on ImageNet, one file per backbone (CPU, reads results/).

    leaf report assets            -> assets/schedules/<backbone>.json and assets/figures/*.png

A book has the format of a released schedule file, so every entry runs with
`leaf evaluate|benchmark --schedule assets/schedules/<bb>.json --key <key>`, which print the paper's number next to the new one.
Each entry holds:
- `config`: the runtime config in the alpha form (`pda`), or the baseline's own config;
- `measured`: the paper's numbers;
  - `val` and `v2`: top-1 (%) on ImageNet val (50,000) and V2 (10,000), BF16, batch 32;
  - `ms_bs32` and `sd`: latency at batch 32, from the timing session of the entry's campaign;
  - `gmac`;
- `paper`: the tables and figures that use the entry;
- `campaign`, plus `source_key` and `source_config_sha` (the campaign's own name and config hash).

Sections (key prefixes):
- `leaf/`: the paper's method, as released in schedules/leaf;
- `main/`: DiffRate, the re-run baselines and LEAF-M0 of the main comparison;
- `main_ablation/`, `calibration_seed/`, `leaf_alpha4/`: the other arms of the main campaign;
- `ablation/`: the ablation and setting-grid campaign;
- `low_budget/`, `horizon/`, `latency_pricing/`, `merge_variants/`, `gmac_planned/`, `leaf_m0/`: the other campaigns.

A schedule evaluated in several campaigns appears once per campaign, with that campaign's latency session.
"""
import glob
import json
from pathlib import Path
import re
import subprocess

from leaf.backbones import SPECS
from leaf.reproduction.experiments.campaign_reports import BBS as PAPER_BBS

REPO = Path(__file__).resolve().parents[3]
RESULTS = REPO / 'results'
SCHEDULES = REPO / 'schedules'
ASSETS = REPO / 'assets'
SPLIT_N = dict(val=50000, v2=10000)

PAPER = {   # section -> tables (paper/tables/<name>.tex) and figures (paper/figures/<name>.pdf) that use it
    'leaf/latency': ['results_all_budgets', 'latency_paired_comparison', 'published_methods', 'ablation', 'ablation_structure',
                     'grid_sensitivity', 'fig:accuracy_latency_imagenet_val', 'fig:accuracy_latency_imagenet_v2'],
    'leaf/gmac': ['equal_gmac_pairs', 'equal_gmac_methods', 'equal_gmac_vs_diffrate', 'fig:equal_gmac_latency_residual_vs_stations'],
    'main': ['results_all_budgets', 'latency_paired_comparison', 'equal_gmac_pairs', 'equal_gmac_methods', 'equal_gmac_vs_diffrate',
             'fig:accuracy_latency_imagenet_val', 'fig:accuracy_latency_imagenet_v2'],
    'main_ablation': ['ablation'],
    'calibration_seed': ['ablation'],
    'leaf_alpha4': ['ablation', 'leaf_vs_m0'],
    'ablation': ['ablation_structure', 'grid_sensitivity'],
    'low_budget': ['leaf_vs_m0', 'negative_variants'],
    'horizon': ['ablation', 'negative_variants'],
    'latency_pricing': ['latency_pricing', 'fig:salt_vs_gmac_planning'],
    'merge_variants': ['negative_variants'],
    'gmac_planned': ['equal_gmac_pairs', 'fig:token_schedule_deit_b_10gmac'],
    'leaf_m0': ['leaf_vs_m0', 'm0_vs_diffrate', 'fig:m0_latency_residual_vs_stations'],
}
CAMPAIGN = {'main': 'main_comparison', 'main_ablation': 'main_comparison', 'calibration_seed': 'main_comparison',
            'leaf_alpha4': 'main_comparison', 'ablation': 'ablation_and_alpha_grid', 'low_budget': 'low_budget_and_factorial',
            'horizon': 'horizon_norm_seed_ablation', 'latency_pricing': 'latency_pricing', 'merge_variants': 'merge_variants',
            'gmac_planned': 'gmac_planning', 'leaf_m0': 'leaf_m0'}


def load(p): return json.loads(Path(p).read_text())


def slug(text): return re.sub(r'[^a-z0-9]+', '_', text.lower()).strip('_')


class Lookup:
    """Measured top-1 by config hash in every campaign's per-split run files, and bs32 latency by hash in one
    campaign's timing session."""

    def __init__(self, bb):
        self.bb = bb; self.top1 = {}
        for split, n in SPLIT_N.items():
            for p in sorted(glob.glob(str(RESULTS / f'*/{bb}/{split}/runs/*.json')) + glob.glob(str(RESULTS / f'*/{bb}/bs32/{split}/runs/*.json'))):
                sha = Path(p).stem
                if (split, sha) in self.top1: continue
                row = load(p)
                if row.get('n') == n and row.get('batch_size', 32) == 32 and 'top1' in row: self.top1[(split, sha)] = row['top1']

    def accuracy(self, *shas):
        out = {}
        for split in SPLIT_N:
            v = next((self.top1[(split, s)] for s in shas if s and (split, s) in self.top1), None)
            if v is not None: out[split] = round(float(v), 3)
        return out

    def latency(self, campaign, *ids, folder='latency'):
        """{ms_bs32, sd, rounds} (and ms_by_batch when other batch sizes were timed) from the timing rounds
        <folder>/*<id>#<round>.json of a campaign; ids are config hashes or (GMAC planning) arm names."""
        import numpy as np
        for i in ids:
            if not i: continue
            by_bs = {}
            base = RESULTS / f'{campaign}/{self.bb}/{folder}'
            files = glob.glob(str(base / f'{glob.escape(i)}#*.json')) + glob.glob(str(base / f'{self.bb}-{glob.escape(i)}#*.json'))
            for p in sorted(files):
                row = load(p)
                ms = row['measurements'] if 'measurements' in row else {str(row.get('measurement', {}).get('bs', 32)): row.get('measurement')}
                for bs, m in ms.items():
                    if isinstance(m, dict) and 'ms_mean' in m: by_bs.setdefault(str(bs), []).append(m['ms_mean'])
            if not by_bs: continue
            out = {}
            if '32' in by_bs:
                v = by_bs['32']
                out.update(ms_bs32=round(float(np.mean(v)), 3), sd=round(float(np.std(v, ddof=1)), 3) if len(v) > 1 else 0., rounds=len(v))
            if set(by_bs) != {'32'}: out['ms_by_batch'] = {bs: round(float(np.mean(v)), 3) for bs, v in sorted(by_bs.items(), key=lambda x: int(x[0]))}
            return out
        return {}


def entry(section, config, measured, source_key, source_sha=None, extra=None):
    out = dict(config=config, measured=measured, paper=PAPER[section.split('/')[0] if section not in PAPER else section],
               campaign=CAMPAIGN.get(section.split('/')[0], 'ablation_and_alpha_grid' if section.startswith('leaf') else None),
               source_key=source_key)
    if source_sha: out['source_config_sha'] = source_sha
    if extra: out.update(extra)
    return out


def book(bb):
    from leaf.api import stations
    from leaf.evaluation.runner import config_hash
    from leaf.planner.alpha import config_gmac
    spec = SPECS[bb]; look = Lookup(bb); out = {}

    def add(key, value):
        if key in out:                       # the same arm listed in two grids of a campaign: one entry, both names
            if config_hash(out[key]['config']) != config_hash(value['config']): raise KeyError(key)
            names = out[key]['source_key']
            out[key]['source_key'] = (names if isinstance(names, list) else [names]) + [value['source_key']]
            return
        if value['config'].get('kind') in ('pd', 'pdm', 'pda'):
            value.setdefault('stations', stations(value['config'], spec))
        out[key] = value

    def gmac(cfg):
        try: return round(config_gmac(cfg, spec), 4) if cfg['kind'] in ('pd', 'pdm', 'pda') else None
        except Exception: return None

    # leaf/: the release (schedules/leaf)
    release = load(SCHEDULES / f'leaf/{bb}.json')
    for k, e in release['schedules'].items():
        add(f'leaf/{k}', dict(entry('leaf/' + k.split('/')[0], e['config'], e['measured'], k, extra=dict(budget=e['budget'], planner=e['planner'])),
                              campaign='ablation_and_alpha_grid' if k.startswith('latency/') else 'main_comparison'))
    # main/: every family of the main comparison except the method itself (latency-paired session of the report)
    plans = load(SCHEDULES / f'main_comparison_plans/{bb}.json')['jobs']
    for arm in load(RESULTS / 'main_comparison_analysis/report_leaf.json')['arms'][bb]:
        if arm['family'] == 'LEAF': continue
        job = plans[arm['name']]
        parts = arm['name'].split('/')
        tail = '/'.join(parts[2:]) if arm['family'] == 'LEAF-M0' else parts[-1]
        add(f"main/{slug(arm['family'])}/{tail}", entry('main', job['config'],
            dict(val=round(arm['val'], 3), v2=round(arm['v2'], 3), ms_bs32=round(arm['ms'], 3), sd=round(arm['sd'], 3), gmac=arm['gmac']), arm['name'],
            job.get('source_config_sha'), dict(family=arm['family'])))
    dense = plans.get('ref/dense') or plans.get('ref/history/dense')
    # other arms of the main campaign
    for name, job in plans.items():
        if 'config' not in job: continue
        if name.startswith('ablation/'): key, section = 'main_ablation/' + name[len('ablation/'):], 'main_ablation'
        elif name.startswith('seed'): key, section = 'calibration_seed/' + name[len('seed'):], 'calibration_seed'
        elif name.startswith('leaf_alpha4/'): key, section = name, 'leaf_alpha4'
        else: continue
        src = job.get('source_config_sha', job.get('config_sha'))
        m = dict(look.accuracy(src, config_hash(job['config'])), **look.latency('main_comparison', src), gmac=gmac(job['config']))
        add(key, entry(section, job['config'], m, name, src))
    # ablation/: the ablation and setting-grid campaign (alpha form natively)
    seen = {config_hash(e['config']) for k, e in out.items() if k.startswith('leaf/latency/')}
    for f in sorted((RESULTS / f'ablation_and_alpha_grid/{bb}/plans').glob('*.json')):
        if f.stem in ('leaf', 'reference'): continue
        for name, job in load(f)['jobs'].items():
            if 'config' not in job: continue
            sha = config_hash(job['config'])
            m = look.accuracy(sha)
            if not m: continue                                   # planned but not evaluated (e.g. duplicates of another arm)
            m.update(look.latency('ablation_and_alpha_grid', sha), gmac=gmac(job['config']))
            add(f'ablation/{name}', entry('ablation', job['config'], m, name, None,
                                          dict(same_as_leaf=sha in seen) if sha in seen else None))
    # the other campaigns, from their alpha-form exports
    for section, folder, field in (('low_budget', 'low_budget_and_factorial', 'jobs'), ('horizon', 'horizon_norm_seed_ablation', 'jobs'),
                                   ('latency_pricing', 'latency_pricing', 'schedules'), ('merge_variants', 'merge_variants', 'schedules'),
                                   ('gmac_planned', 'gmac_planned', 'schedules'), ('leaf_m0', 'leaf_m0', 'schedules')):
        path = SCHEDULES / f'{folder}/{bb}.json'
        if not path.exists(): continue
        doc = load(path); jobs = doc.get(field) or doc.get('jobs') or doc.get('schedules')
        kdoc = load(SCHEDULES / f'kappa_form/{folder}/{bb}.json')       # the campaign's own configs name its result files
        kjobs = kdoc.get(field) or kdoc.get('jobs') or kdoc.get('schedules')
        campaign = CAMPAIGN[section]
        for name, job in jobs.items():
            cfg = job.get('config')
            if not cfg or cfg.get('kind') == 'dense': continue
            src = job.get('source_config_sha') or config_hash(kjobs[name]['config'])
            if cfg.get('base') == 'extra_pdm' and cfg.get('ablation', {}).get('variant') in ('shuffle', 'shuffle_size', 'identity'):
                cfg = kjobs[name]['config']           # D placebo variants exist only with the k score (leaf/model/alpha_score.py)
            m = look.accuracy(src, config_hash(cfg))
            ex = job.get('measured') or {}                                   # the campaign's own summary, as exported
            for split in SPLIT_N:
                if split not in m and ex.get(f'top1_{split}') is not None: m[split] = round(float(ex[f'top1_{split}']), 3)
            if not m: continue
            if section == 'latency_pricing':                                  # timed per device regime
                for dev, folder in (('gpu', 'final_cuda'), ('cpu', 'final_cpu')):
                    lat = look.latency(campaign, src, config_hash(cfg), folder=folder)
                    if lat: m[f'latency_{dev}'] = lat
            else:
                m.update(look.latency(campaign, src, config_hash(cfg), name.replace('/', '|')))
            if 'ms_bs32' not in m and (ex.get('latency_ms') or {}).get('32') is not None:
                m['ms_bs32'] = ex['latency_ms']['32']
            m['gmac'] = gmac(cfg)
            if ex: m['export'] = ex
            add(f'{section}/{name}', entry(section, cfg, m, name, src))
    return dict(backbone=bb, timm_id=release.get('timm_id'), dense=release.get('dense'),
                note=__doc__.split('\n\n')[1].strip(), sections=PAPER, schedules=out)


def autocrop(png, margin=12):
    """Trim the white page around a rendered figure."""
    import numpy as np
    from PIL import Image
    im = Image.open(png).convert('RGB'); a = np.asarray(im)
    ys, xs = np.where((a < 250).any(axis=2))
    box = (max(xs.min() - margin, 0), max(ys.min() - margin, 0), min(xs.max() + margin, a.shape[1]), min(ys.max() + margin, a.shape[0]))
    im.crop(box).save(png)


def figures():
    """PNG renderings of the paper's figures for the README (GitHub shows PNG, not PDF): the six generated figures
    (pdftocairo) and the TikZ method overview (compiled alone with the paper's preamble; needs tectonic)."""
    import shutil
    import tempfile
    out = ASSETS / 'figures'; out.mkdir(parents=True, exist_ok=True)
    for pdf in sorted((REPO / 'paper/figures').glob('*.pdf')):
        subprocess.run(['pdftocairo', '-png', '-singlefile', '-r', '160', str(pdf), str(out / pdf.stem)], check=True)
        autocrop(out / f'{pdf.stem}.png'); print('wrote', out / f'{pdf.stem}.png')
    if not shutil.which('tectonic'):
        print('tectonic not found: method_overview.png not rendered'); return
    main_tex = (REPO / 'paper/main.tex').read_text()
    preamble = main_tex[:main_tex.index('\\begin{document}')]
    # the drawing alone: the caption is written in the README
    fig = (REPO / 'paper/figures/method_overview.tex').read_text()
    drawing = fig[fig.index('\\centering'):fig.index('\\caption{')]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / 'overview.tex').write_text(preamble + '\\pagestyle{empty}\n\\begin{document}\n' + drawing + '\\end{document}\n')
        subprocess.run(['tectonic', 'overview.tex'], cwd=tmp, check=True, capture_output=True)
        subprocess.run(['pdftocairo', '-png', '-singlefile', '-r', '200', str(tmp / 'overview.pdf'), str(out / 'method_overview')], check=True)
    autocrop(out / 'method_overview.png'); print('wrote', out / 'method_overview.png')


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog='leaf report assets', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--backbones', nargs='+', default=list(PAPER_BBS)); ap.add_argument('--no-figures', action='store_true')
    a = ap.parse_args(argv)
    for bb in a.backbones:
        doc = book(bb)
        path = ASSETS / f'schedules/{bb}.json'; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(doc, indent=1) + '\n')
        counts = {}
        for k in doc['schedules']: counts[k.split('/')[0]] = counts.get(k.split('/')[0], 0) + 1
        print(f'wrote {path}: {len(doc["schedules"])} schedules {counts}', flush=True)
    if not a.no_figures: figures()


if __name__ == '__main__':
    main()
