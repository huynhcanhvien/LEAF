"""CPU-only spectral audit of deployed forecast groups; no images or GPU needed.

Run: HF_HUB_OFFLINE=1 python leaf/reproduction/experiments/forecast_rank.py --backbones deit_b
Uses the same FP32 folding/group averaging and four-thread FP64 CPU SVD as inference.
"""
import argparse
import hashlib
import json
import statistics
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import torch
from leaf.backbones import load
from leaf.forecast.attention_forecast import _group_matrices, _CACHE, SVD_THREADS

ROOT = Path(__file__).resolve().parents[3]


def write_table(result, path):
    names = dict(deit_t='DeiT-T', deit_s='DeiT-S', deit_b='DeiT-B',
                 augreg_s='AugReg-S', augreg_b='AugReg-B', mae_b='ViT-B (MAE)', mae_l='ViT-L (MAE)')
    lines = [r'\begin{table}[t]\centering\small',
        r'\caption{Rank-32 approximation of the forecast group matrices. All nonempty groups at all eligible station blocks ($1\le\ell\le B-2$), with at most 16 groups per block; each matrix has equal weight. Energy is the fraction of squared singular values retained. Error is the relative Frobenius reconstruction error, measured before the BF16 inference cast. No images are used.}',
        r'\label{tab:forecast-rank}',
        r'\begin{tabular}{l r rr r}\toprule',
        r'Backbone & Matrices & \multicolumn{2}{c}{Energy (\%)} & Relative error\\',
        r' & & Median & Min--max & Median\\\midrule']
    for name, entry in result['backbones'].items():
        entry['energy_median'] = statistics.median(r['energy_rank32'] for r in entry['rows'])
        entry['error_median'] = statistics.median(r['relative_frobenius_error'] for r in entry['rows'])
        lines.append(f"{names[name]} & {entry['n_matrices']} & {100*entry['energy_median']:.1f} & "
                     f"{100*entry['energy_min']:.1f}--{100*entry['energy_max']:.1f} & {entry['error_median']:.3f}" + r'\\')
    lines += [r'\bottomrule\end{tabular}', r'\end{table}']
    path.write_text('\n'.join(lines) + '\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backbones', nargs='+', default=['deit_b'])
    parser.add_argument('--output', type=Path, default=ROOT / 'results/forecast_rank/audit.json')
    args = parser.parse_args()
    torch.set_num_threads(SVD_THREADS)
    design = ROOT / 'results/main_comparison/design.json'
    configs = json.loads(design.read_text())['backbones']
    result = dict(rank=32, requested_groups=16, device='cpu', torch_version=torch.__version__,
                  folding_dtype='float32', svd_dtype='float64', threads=SVD_THREADS,
                  design_sha256=digest(design), implementation_sha256=digest(ROOT / 'leaf/forecast/attention_forecast.py'),
                  definition='energy=sum(s[:32]^2)/sum(s^2); relative Frobenius error=sqrt(1-energy)',
                  backbones={})
    for name in args.backbones:
        vit, spec = load(name, device='cpu')
        group_path = Path(configs[name]['root']) / 'groups.json'
        groups = json.loads(group_path.read_text())['groups']
        rows = []
        for block in range(1, spec.depth - 1):
            labels = groups[str(block)]['16']
            U, count, (left, singular, right), _, _ = _group_matrices(list(vit.blocks)[block + 1:], labels)
            # Singular values returned by the production builder are stored as FP32;
            # accumulate their squared energy in FP64.
            squared = singular.double().square()
            energy = squared[:, :32].sum(-1) / squared.sum(-1)
            errors = (1 - energy).clamp_min(0).sqrt()
            reconstructed = (left[:, :, :32].double() * singular[:, None, :32].double()) @ right[:, :32].double()
            measured = torch.linalg.matrix_norm(U.double() - reconstructed) / torch.linalg.matrix_norm(U.double())
            torch.testing.assert_close(measured, errors, atol=1e-6, rtol=1e-6)
            for g in range(len(count)):
                rows.append(dict(block=block, group=g, heads=int(count[g]),
                                 energy_rank32=float(energy[g]), relative_frobenius_error=float(errors[g]),
                                 leading_singular_values=singular[g, :32].tolist(),
                                 squared_frobenius_norm=float(squared[g].sum())))
            print(f'{name} block {block}: energy {energy.min():.4f}..{energy.max():.4f}', flush=True)
            _CACHE.clear()
        e = torch.tensor([r['energy_rank32'] for r in rows], dtype=torch.float64)
        err = torch.tensor([r['relative_frobenius_error'] for r in rows], dtype=torch.float64)
        result['backbones'][name] = dict(groups_path=str(group_path), groups_sha256=digest(group_path),
            n_matrices=len(rows), energy_min=float(e.min()), energy_median=float(e.median()),
            energy_mean=float(e.mean()), energy_max=float(e.max()),
            error_min=float(err.min()), error_median=float(err.median()), error_max=float(err.max()), rows=rows)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_table(result, ROOT / 'results/paper_build/tables/forecast_rank.tex')
        args.output.write_text(json.dumps(result, indent=2) + '\n')
        print(name, {k: v for k, v in result['backbones'][name].items() if k != 'rows'}, flush=True)
        del vit


if __name__ == '__main__':
    main()
