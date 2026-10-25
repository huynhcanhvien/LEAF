"""Which SALT formula extrapolates best below the timing range? Every formula of leaf/latency/salt.py is fitted on the
frozen batch-32 workloads of each backbone (grouped 3-fold CV picks the ridge) and predicts the 21 low-budget schedules
(results/low_budget_and_factorial, measured in the low-budget session). CPU only; writes
results/salt_low_coverage/formula_extrapolation.json. Run: python scripts/salt_formula_extrapolation.py"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from leaf.reproduction.experiments import salt_low_coverage as S, ablation_and_alpha_grid as A
from leaf.latency.salt import features, fit_salt, FORMULAS, predict, matrix
from leaf.evaluation.runner import config_hash
BBS = ('deit_t', 'deit_s', 'deit_b', 'augreg_s', 'augreg_b', 'mae_b', 'mae_l')
out = {f: dict(err_pct=[], cv_mape=[]) for f in FORMULAS}; rows = []
for bb in BBS:
    spec = A.spec_of(bb); data = []
    for row in S.frozen_rows(bb):
        cfg = S.kv(row['config']); g = row.get('group') or config_hash({k: v for k, v in cfg.items() if k not in ('kind', 'merge')})
        data.append(dict(config=cfg, ms=row['ms'], group=g, features=features(cfg, spec)))
    low = [s for s in S.low_schedules(bb) if s['measured_ms']]
    lowrows = [dict(features=features(S.kv(s['config']), spec)) for s in low]; y = np.array([s['measured_ms'] for s in low])
    for f in FORMULAS:
        m = fit_salt(data, f); p = predict(m, matrix(lowrows, f)); e = 100 * (p / y - 1)
        out[f]['err_pct'] += [float(v) for v in e]; out[f]['cv_mape'].append(min(t['mape'] for t in m['tuning']))
        for s, v in zip(low, e): rows.append(dict(backbone=bb, name=s['name'], formula=f, err_pct=float(v)))
summary = {f: dict(low_budget_signed_pct=float(np.mean(d['err_pct'])), low_budget_mape_pct=float(np.mean(np.abs(d['err_pct']))),
                   low_budget_max_abs_pct=float(np.max(np.abs(d['err_pct']))), in_range_cv_mape_pct=float(np.mean(d['cv_mape'])),
                   in_range_cv_mape_range=[float(min(d['cv_mape'])), float(max(d['cv_mape']))]) for f, d in out.items()}
Path('results/salt_low_coverage/formula_extrapolation.json').write_text(json.dumps(dict(summary=summary, rows=rows, time=time.time()), indent=1))
for f, s in summary.items(): print(f"{f:15s} low-budget signed {s['low_budget_signed_pct']:+5.2f} MAPE {s['low_budget_mape_pct']:4.2f} max {s['low_budget_max_abs_pct']:5.2f} | in-range CV {s['in_range_cv_mape_pct']:.2f}")
