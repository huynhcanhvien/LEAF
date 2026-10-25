"""Paper tables from the analysed campaign results (CPU only):  results/paper_build/tables/<name>.tex

Used by leaf.reproduction.report.build (python -m leaf report tables), which also adds the rows of the paper's method, the ablation-campaign
tables, the forecast tables and the LEAF-M0 table, and copies everything to paper/tables/.
Inputs: results/main_comparison (plans, latency, per-image arrays, report.json, ablation_report.json), results/latency_pricing,
results/equal_gmac, results/imagenet_real, results/search_cost.
"""
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
import numpy as np

OUT = REPO / 'results/paper_build/tables'
MAIN_COMPARISON = REPO / 'results/main_comparison'
PRICING = REPO / 'results/latency_pricing'
SEG = REPO / 'results/segmentation'
NAMES = dict(deit_t='DeiT-T', deit_s='DeiT-S', deit_b='DeiT-B', augreg_s='ViT-S (AugReg)', augreg_b='ViT-B (AugReg)',
             mae_b='ViT-B (MAE)', mae_l='ViT-L (MAE)')
BBS = tuple(NAMES)


def load(p): return json.loads(Path(p).read_text())


def write(name, text):
    # 2026-10-18: metric arrows in the header rows and line-break protection (tools/typeset.py)
    sys.path.insert(0, str(Path(__file__).resolve().parent)); from typesetting import metric_arrows, fix_breaks
    text = fix_breaks(metric_arrows(text))
    OUT.mkdir(parents=True, exist_ok=True); (OUT / name).write_text(text); print('wrote', OUT / name)


def fmt(x, d=2): return f'{x:.{d}f}'


def sig(delta, p): return (f'\\underline{{{delta:+.2f}}}' if delta > 0 else f'\\textbf{{{delta:+.2f}}}') if p < .05 else f'{delta:+.2f}'


# ---------------------------------------------------------------- main: LEAF vs DiffRate at every anchor
def anchors(by_gmac=False):
    # by_gmac (2026-10-19): rows are identified by GMAC and latency only, no DiffRate budget names
    from leaf.reproduction.experiments.campaign_reports import round_means, correct
    from leaf.evaluation.data import mcnemar, holm
    rows = []
    for bb in BBS:
        jobs = load(MAIN_COMPARISON / bb / 'plans.json')['jobs']; ms = {k.split('-', 1)[1]: v for k, v in round_means(MAIN_COMPARISON / bb / 'latency', 32).items()}
        dense = jobs['ref/dense']['config_sha']
        for name, row in jobs.items():
            if not name.startswith('leaf_alpha4/') or 'config' not in row: continue
            bid = name.split('/')[1]; dr = jobs.get(f'ref/history/baseline/{bid}')
            r = dict(bb=bb, budget=bid, budget_ms=row.get('budget_ms'), ours=row['config_sha'], ms=ms[row['config_sha']], gmac=row['gmac'], K=row['details']['K'])
            for s in ('val', 'v2'): r[s] = float(correct(MAIN_COMPARISON / bb, row['config_sha'], s).mean() * 100)
            if dr and dr['config_sha'] in ms:
                r['dr_ms'] = ms[dr['config_sha']]; r['dr_gmac'] = dr.get('gmac')
                for s in ('val', 'v2'):
                    a, b = correct(MAIN_COMPARISON / bb, row['config_sha'], s), correct(MAIN_COMPARISON / bb, dr['config_sha'], s)
                    p, _, _ = mcnemar(a, b); r[f'dr_{s}'] = float(b.mean() * 100); r[f'p_{s}'] = dict(p=p)
                t = max(.05, 2 * np.sqrt((r['ms'][1] ** 2 + r['dr_ms'][1] ** 2) / 2))
                r['faster'] = r['dr_ms'][0] - r['ms'][0] >= t
            r['dense'] = dict(ms=ms[dense][0], val=float(correct(MAIN_COMPARISON / bb, dense, 'val').mean() * 100), v2=float(correct(MAIN_COMPARISON / bb, dense, 'v2').mean() * 100))
            rows.append(r)
    for s in ('val', 'v2'): holm([r[f'p_{s}'] for r in rows if f'p_{s}' in r])
    order = lambda r: (BBS.index(r['bb']), r['ms'][0])
    lines = [r'\begin{longtable}{ll rrr rrrrr}',
             r'\caption{\ours\ (station-rich \cost, P+D, M3) at every budget of the main campaign, all arms timed in one session per backbone (bs32 ms, mean of three interleaved rounds). DiffRate columns: the released schedule of the same anchor. $^{\dagger}$: \ours\ dominates the anchor (faster by at least $\tau$ and higher top-1, Holm-adjusted McNemar $p<0.05$); $^{\ddagger}$: significantly less accurate. $K$: stations.}\label{tab:all-budgets}\\',
             r'\toprule', r'& & \multicolumn{3}{c}{DiffRate (re-run)} & \multicolumn{5}{c}{\ours} \\ \cmidrule(lr){3-5}\cmidrule(lr){6-10}',
             r'Model & Budget & ms & Val & V2 & $K$ & GMAC & ms & Val & V2 \\ \midrule\endfirsthead',
             r'\multicolumn{10}{c}{\tablename\ \thetable\ (continued)}\\\toprule Model & Budget & ms & Val & V2 & $K$ & GMAC & ms & Val & V2 \\ \midrule\endhead']
    cur = None
    for r in sorted(rows, key=order):
        if r['bb'] != cur:
            cur = r['bb']; d = r['dense']
            lines.append(f"\\multicolumn{{2}}{{l}}{{{NAMES[cur]} (dense)}} & {fmt(d['ms'])} & {fmt(d['val'])} & {fmt(d['v2'])} &&&&& \\\\")
        mark = {}
        for s in ('val', 'v2'):
            mark[s] = ''
            if f'p_{s}' in r and r[f'p_{s}']['holm'] < .05:
                mark[s] = r'$^{\dagger}$' if (r[s] > r[f'dr_{s}'] and r['faster']) else (r'$^{\ddagger}$' if r[s] < r[f'dr_{s}'] else '')
        drc = f"{fmt(r['dr_ms'][0])} & {fmt(r['dr_val'])} & {fmt(r['dr_v2'])}" if 'dr_ms' in r else '& &'
        lines.append(f"& {r['budget']} & {drc} & {r['K']} & {fmt(r['gmac'])} & {fmt(r['ms'][0])} & {fmt(r['val'])}{mark['val']} & {fmt(r['v2'])}{mark['v2']} \\\\")
    if by_gmac: lines = anchors_gmac_lines(rows, order)
    lines += [r'\bottomrule', r'\end{longtable}']
    write('results_all_budgets.tex', '\n'.join(lines) + '\n')
    dom = sum(1 for r in rows if 'p_val' in r and r['p_val']['holm'] < .05 and r['val'] > r['dr_val'] and r['faster'])
    worse = {s: sum(1 for r in rows if f'p_{s}' in r and r[f'p_{s}']['holm'] < .05 and r[s] < r[f'dr_{s}']) for s in ('val', 'v2')}
    return dict(n_anchor=sum('p_val' in r for r in rows), dominate_val=dom, worse=worse, rows=rows)


def all_budgets_group(bb, d, first=False):
    # group header of tab:all-budgets: backbone name and its dense model; '\\*' keeps it on the page of its first row
    nm = NAMES[bb].split(' ', 1); nm = f"\\mbox{{{nm[0]}}}" + (f" {nm[1]}" if len(nm) > 1 else '')
    return ([] if first else [r'\midrule']) + [
        f"\\multicolumn{{10}}{{l}}{{\\emph{{{nm}}}\\quad dense: {fmt(d['ms'])}\\,ms, Val {fmt(d['val'])}, V2 {fmt(d['v2'])}}} \\\\*"]


def anchors_gmac_lines(rows, order):
    """tab:all-budgets body without budget names: DiffRate block gains GMAC; rows sorted by latency."""
    # 2026-10-22: one header row per backbone (with its dense model) instead of a Model column; '--' where no DiffRate schedule exists
    lines = [r'\begin{longtable}{rrrr r rrrrr}',
             r'\caption{\ours\ (station-rich \cost, P+D, M3) at every budget of the main campaign, all arms timed in one session per backbone (bs32 ms, mean of three interleaved rounds). $L$: latency budget of the \ours\ schedule. Where a baseline releases fixed schedules (DiffRate), budgets were placed just below ($0.97\times$) their latencies measured in an earlier session, so that every released schedule has a latency-matched counterpart, and each \ours\ schedule is shown next to that schedule; two faster budgets per backbone extend the range. The AugReg backbones have no released schedules and use an even grid of $0.50$--$0.86\times$ dense latency. --: no released DiffRate schedule at this budget. $^{\dagger}$: \ours\ dominates that DiffRate schedule (faster by at least $\tau$ and higher top-1, Holm-adjusted McNemar $p<0.05$); $^{\ddagger}$: significantly less accurate. $K$: stations.}\label{tab:all-budgets}\\',
             r'\toprule', r'\multicolumn{4}{c}{DiffRate (re-run)} & \multicolumn{6}{c}{\ours} \\ \cmidrule(lr){1-4}\cmidrule(lr){5-10}',
             r'GMAC & ms & Val & V2 & $L$ (ms) & $K$ & GMAC & ms & Val & V2 \\ \midrule\endfirsthead',
             r'\multicolumn{10}{c}{\tablename\ \thetable\ (continued)}\\\toprule GMAC & ms & Val & V2 & $L$ (ms) & $K$ & GMAC & ms & Val & V2 \\ \midrule\endhead']
    cur = None
    for r in sorted(rows, key=order):
        if r['bb'] != cur:
            lines += all_budgets_group(r['bb'], r['dense'], first=cur is None); cur = r['bb']
        mark = {}
        for s in ('val', 'v2'):
            mark[s] = ''
            if f'p_{s}' in r and r[f'p_{s}']['holm'] < .05:
                mark[s] = r'$^{\dagger}$' if (r[s] > r[f'dr_{s}'] and r['faster']) else (r'$^{\ddagger}$' if r[s] < r[f'dr_{s}'] else '')
        drg = fmt(r['dr_gmac']) if r.get('dr_gmac') is not None else '--'
        drc = f"{drg} & {fmt(r['dr_ms'][0])} & {fmt(r['dr_val'])} & {fmt(r['dr_v2'])}" if 'dr_ms' in r else '-- & -- & -- & --'
        lines.append(f"{drc} & {fmt(r['budget_ms'])} & {r['K']} &{fmt(r['gmac'])} & {fmt(r['ms'][0])} & {fmt(r['val'])}{mark['val']} & {fmt(r['v2'])}{mark['v2']} \\\\")
    return lines


def pairs():
    rep = load(MAIN_COMPARISON / 'report.json'); summary = {}
    for t in rep['tests']:
        s = summary.setdefault(t['family'], dict(n=0, vw=0, vl=0, ww=0, wl=0, dv=0., dw=0., bbs=set()))
        s['n'] += 1; s['dv'] += t['val']['delta']; s['dw'] += t['v2']['delta']; s['bbs'].add(t['backbone'])
        if t['val']['holm'] < .05: s['vw' if t['val']['delta'] > 0 else 'vl'] += 1
        if t['v2']['holm'] < .05: s['ww' if t['v2']['delta'] > 0 else 'wl'] += 1
    order = ['DiffRate', 'LEAF-M0', 'PPT', 'ToFu', 'RAPID', 'ToMe', 'ToMe+M3', 'One-More-Token', 'EViT', 'PiToMe']
    label = {'LEAF-M0': r'\ours-M0 (original)'}
    lines = [r'\begin{table}[t]\centering\small',
             r'\caption{Latency-paired comparisons of \ours\ with every re-run family (79 pairs, seven backbones). A pair matches the nearest point of the family within $\tau=\max(0.05\,\mathrm{ms},2\,\mathrm{SD})$. W/L: significant wins/losses after Holm over all pairs of a split. Means are descriptive.}\label{tab:latency-pairs}',
             r'\begin{tabular}{l rr rr rr}\toprule Family & Pairs & Backbones & Val W/L & $\bar\Delta$Val & V2 W/L & $\bar\Delta$V2 \\ \midrule']
    for f in order:
        if f not in summary: continue
        s = summary[f]
        lines.append(f"{label.get(f, f)} & {s['n']} & {len(s['bbs'])} & {s['vw']}/{s['vl']} & {s['dv']/s['n']:+.2f} & {s['ww']}/{s['wl']} & {s['dw']/s['n']:+.2f} \\\\")
    lines += [r'\bottomrule\end{tabular}\end{table}']
    write('latency_paired_comparison.tex', '\n'.join(lines) + '\n')
    return {k: {kk: (sorted(vv) if isinstance(vv, set) else vv) for kk, vv in v.items()} for k, v in summary.items()}


def pricing():
    cmp = load(PRICING / 'compare.json'); regs = {}
    for t in cmp['tests']:
        if not t.get('matched'): continue
        r = regs.setdefault(t['regime'], dict(n=0, vw=0, vl=0, ww=0, wl=0, d=[]))
        r['n'] += 1; r['d'].append(t['val']['delta'])
        if t['val']['holm'] < .05: r['vw' if t['val']['delta'] > 0 else 'vl'] += 1
        if t['v2']['holm'] < .05: r['ww' if t['v2']['delta'] > 0 else 'wl'] += 1
    gate = load(PRICING / 'analysis_summary.json')
    lab = dict(gpu_bs4='GPU, batch 4', gpu_bs8='GPU, batch 8', cpu_bs1='CPU, batch 1', cpu_bs8='CPU, batch 8')
    lines = [r'\begin{table}[t]\centering\small',
             r'\caption{Latency pricing where latency departs from GMAC (DeiT-S/B, ViT-B MAE). Left: cross-validated error of a GMAC-only latency model versus the best \cost\ form on 83--97 historical workloads per backbone. Right: \cost-planned versus GMAC-planned DP with the same probes, score and operator, paired at equal measured latency on the target device; W/L after Holm over all 31 pairs per split.}\label{tab:latency-pricing}',
             r'\resizebox{\linewidth}{!}{\begin{tabular}{l rr rrrr}\toprule & \multicolumn{2}{c}{Latency MAPE (\%)} & \multicolumn{4}{c}{\cost\ vs GMAC planning} \\ \cmidrule(lr){2-3}\cmidrule(lr){4-7}',
             r'Regime & GMAC & \cost & Pairs & Val W/L & $\Delta$Val range & V2 W/L \\ \midrule']
    for reg in ('gpu_bs4', 'gpu_bs8', 'cpu_bs1', 'cpu_bs8'):
        g = [v for k, v in gate.items() if k.endswith('/' + reg)]; r = regs.get(reg)
        gm = f"{min(x['gmac_mape'] for x in g):.1f}--{max(x['gmac_mape'] for x in g):.1f}"
        sm = f"{min(x['best_mape'] for x in g):.1f}--{max(x['best_mape'] for x in g):.1f}"
        rr = f"{r['n']} & {r['vw']}/{r['vl']} & {min(r['d']):+.2f}\\,\\dots\\,{max(r['d']):+.2f} & {r['ww']}/{r['wl']}" if r else '0 & & &'
        lines.append(f'{lab[reg]} & {gm} & {sm} & {rr} \\\\')
    g32 = [v for k, v in gate.items() if k.endswith('/gpu_bs32')]
    lines.append(f"GPU, batch 32 (ref.) & {min(x['gmac_mape'] for x in g32):.1f}--{max(x['gmac_mape'] for x in g32):.1f} & {min(x['best_mape'] for x in g32):.1f}--{max(x['best_mape'] for x in g32):.1f} & \\multicolumn{{4}}{{c}}{{GMAC planning on the envelope ($\\pm$0.12), budget overrun $\\le$0.69\\,ms (earlier campaign)}} \\\\")
    lines += [r'\bottomrule\end{tabular}}\end{table}']
    write('latency_pricing.tex', '\n'.join(lines) + '\n')
    fid = cmp['fidelity']
    return dict(regimes={k: {kk: vv for kk, vv in v.items() if kk != 'd'} for k, v in regs.items()},
                budget_met=sum(f['meets'] for f in fid), n=len(fid))


def seg():
    rep = load(SEG / 'report.json'); arms = rep['arms']; d = arms['dense']
    lines = [r'\begin{table}[t]\centering\small',
             r'\caption{ADE20K semantic segmentation with a frozen DINOv2 ViT-B/14 and its official linear head (no training). Reduced encoders restore their tokens to the full grid before the head (merged tokens copy their destination, pruned tokens keep their state). \ours\ schedules are re-planned at 1,370 tokens; latency is encoder plus restoration, measured exclusively (the constant head is excluded). Single-scale sliding-window mIoU.}\label{tab:segmentation}',
             r'\begin{tabular}{l r rr rr}\toprule Encoder & mIoU & bs1 ms & vs dense & bs8 ms & vs dense \\ \midrule']
    rowsort = sorted(arms.values(), key=lambda r: (0 if r['name'] == 'dense' else 1 if r['name'].startswith('laps') else 2, -r['ms_bs1'] if r['name'].startswith('laps') else r['ms_bs1']))
    for r in rowsort:
        name = 'Dense' if r['name'] == 'dense' else (r'\ours\ $' + r['name'].split('/')[1] + r'\times$ budget' if r['name'].startswith('laps') else 'ToMe $r=' + r['name'].split('r')[-1] + '$')
        lines.append(f"{name} & {fmt(r['miou'])} & {fmt(r['ms_bs1'])} & {r['ms_bs1']/d['ms_bs1']:.2f}$\\times$ & {fmt(r['ms_bs8'])} & {r['ms_bs8']/d['ms_bs8']:.2f}$\\times$ \\\\")
    lines += [r'\bottomrule\end{tabular}\end{table}']
    write('segmentation.tex', '\n'.join(lines) + '\n')
    return rep


def ablation(by_gmac=False, dense=False):
    # by_gmac (2026-10-19): columns are headed by the GMAC of the full method's schedule, not by budget names
    # 2026-10-22: one panel per backbone at \small (the 16-column table had to be shrunk to about 6 pt)
    rows = load(MAIN_COMPARISON / 'ablation_report.json')
    arms = [('prune_only', r'prune only ($\theta=0$)'), ('merge_only', r'merge only ($\theta=1$)'), ('no_forecast', r'$P$ only ($\alpha_t=0$)'),
            ('no_forecast_same_path', r'no $D$ (same path)'), ('m0_operator_same_path', r'M0 merge (same path)'), ('m0_operator', 'M0 merge (re-planned)'),
            ('horizon1_same_path', r'$D$ over 1 future block'), ('horizon3_same_path', r'$D$ over $\le$3 future blocks'),
            ('seed251', 'calibration seed 251'), ('seed351', 'calibration seed 351')]
    bud = {'deit_b': ['extra2', 'extra1', 'DR8.7', 'DR10.0', 'DR10.4', 'DR11.5', 'DR12.5'], 'mae_b': ['extra2', 'extra1', 'DR8.7', 'DR10.0', 'DR10.4', 'DR11.5']}
    lines = [r'\begin{table*}[t]\centering\small',
             r"\caption{Component ablations of \ours\ on the four-value grid $\mathcal A_4$ at the main budgets (variant minus full method, ImageNet val points). Re-planned variants run the same planner at the same latency budget; `same path' keeps the full method's stations and counts. \underline{Underlined}/\textbf{bold}: significantly better/worse (McNemar, Holm over all ablation contrasts per split). `=': identical schedule. $\bar\Delta_{\rm V2}$: mean V2 difference.}\label{tab:ablation}",
             r'\setlength{\tabcolsep}{4.5pt}', r'\begin{tabular}{l rrrrrrr r}\toprule']
    g = {bb: load(MAIN_COMPARISON / bb / 'plans.json')['jobs'] for bb in bud} if by_gmac or dense else {}
    if by_gmac:
        lines[1] = lines[1].replace('at the main budgets (', 'at the main budgets, columns labelled by the GMAC of the full method (')
    width = max(len(v) for v in bud.values())
    for i, (bb, name) in enumerate((('deit_b', r'\mbox{DeiT-B}'), ('mae_b', r'\mbox{ViT-B} (MAE)'))):
        pad = ' & ' * (width - len(bud[bb]))
        heads = [fmt(g[bb][f'leaf_alpha4/{b}']['gmac'], 1) for b in bud[bb]] if by_gmac else bud[bb]
        lines += ([r'\midrule'] if i else []) + [rf'\multicolumn{{{width + 2}}}{{l}}{{\emph{{{name}}}}} \\',
                  ('Variant (GMAC) & ' if by_gmac else 'Variant & ') + ' & '.join(heads) + pad + r' & $\bar\Delta_{\rm V2}\uparrow$ \\ \midrule']
        for arm, lab in arms:
            cells = []
            for b in bud[bb]:
                r = next((r for r in rows if r['bb'] == bb and r['arm'] == arm and r['budget'] == b), None)
                cells.append('--' if r is None else '=' if r['same'] else sig(r['val']['delta'], r['val']['holm']))
            v = [r['v2']['delta'] for r in rows if r['bb'] == bb and r['arm'] == arm and not r['same'] and r['v2']]
            lines.append(lab + ' & ' + ' & '.join(cells) + pad + ' & ' + (f'{np.mean(v):+.2f}' if v else '=') + r' \\')
        if dense:   # 2026-10-19: absolute accuracy of the full method, dense accuracy in the caption
            full = [fmt(correct_(bb, g[bb][f'leaf_alpha4/{b}']['config_sha']).mean() * 100) for b in bud[bb]]
            lines.append(r'\midrule full method (Val, absolute) & ' + ' & '.join(full) + pad + r' &  \\')
    if dense:
        dv = {bb: dense_ref(bb)['val'] for bb in bud}
        lines[1] = lines[1].replace(r'$\bar\Delta_{\rm V2}$: mean V2 difference.}',
                                    rf"$\bar\Delta_{{\rm V2}}$: mean V2 difference. Dense ImageNet val: {dv['deit_b']:.2f} (DeiT-B), {dv['mae_b']:.2f} (ViT-B MAE).}}")
    lines += [r'\bottomrule\end{tabular}\end{table*}']
    write('ablation.tex', '\n'.join(lines) + '\n')
    return rows


def cost():
    th = load(REPO / 'results/search_cost/throughput.json')
    lines = [r'\begin{table}[t]\centering\small',
             r'\caption{Search cost on one RTX 5060 Ti (BF16). \ours\ probes once per backbone (forward only, upper bound at dense cost); every further budget is one DP solve on the CPU, and a new device only needs about 90 timed workloads. DiffRate trains its rate parameters for three ImageNet epochs per budget (forward and backward through the frozen network; lower bound from measured throughput, operator overhead and data loading excluded).}\label{tab:search-cost}',
             r'\begin{tabular}{l rr}\toprule & DeiT-S & DeiT-B \\ \midrule']
    s, b = th['deit_small_patch16_224.fb_in1k'], th['deit_base_patch16_224.fb_in1k']
    fw = 'forward'; fb = [k for k in s if k != 'forward'][0]
    lines.append(f"Throughput, forward (img/s) & {s[fw]:.0f} & {b[fw]:.0f} \\\\")
    lines.append(f"Throughput, forward+backward (img/s) & {s[fb]:.0f} & {b[fb]:.0f} \\\\ \\midrule")
    # probes of a 12-block model on the paper's grids: blocks 1..10 x 9 removals x 3 theta x 7 alpha, block 11 without forecast
    n_probes = 10 * 9 * 3 * 7 + 9 * 3
    lines.append(f"\\ours: {n_probes // 1000}{{,}}{n_probes % 1000:03d} probes $\\times$ 5{{,}}000 images, once (GPU-h) & $\\le${n_probes*5000/s[fw]/3600:.2f} & $\\le${n_probes*5000/b[fw]/3600:.2f} \\\\")
    lines.append(r"\ours: per additional budget & \multicolumn{2}{c}{one DP solve, $\approx$0.15\,s CPU} \\")
    lines.append(f"DiffRate: 3 epochs $\\times$ 1.28M images, per budget (GPU-h) & $\\ge${3*1281167/s[fb]/3600:.2f} & $\\ge${3*1281167/b[fb]/3600:.2f} \\\\")
    lines.append(f"DiffRate: five budgets (GPU-h) & $\\ge${5*3*1281167/s[fb]/3600:.1f} & $\\ge${5*3*1281167/b[fb]/3600:.1f} \\\\")
    lines += [r'\bottomrule\end{tabular}\end{table}']
    write('search_cost.tex', '\n'.join(lines) + '\n')
    return th


def negative():
    # 2026-10-22: a protocol column replaces the column of internal report names, which readers cannot open; the row
    # 'minimum-SALT planning under a KL cap' is dropped because no documented experiment for it was found
    rows = [
        ('Soft relaxation of $(\\theta,\\alpha)$ on fixed topology', 'sigmoid relaxation on the DP\'s stations and counts; Adam on the calibration KL (lr 0.05, 4 epochs); 6 budgets each on DeiT-B and ViT-B (MAE)', 'confirmation KL $+8$--$35\\%$, val lower at 10/12'),
        ('Straight-through refinement of $(\\theta,\\alpha)$', 'straight-through gradients from the DP or a table start, 5,000 training and 5,000 selection images; DeiT-B and ViT-B (MAE); against a coordinate search', 'no accuracy gain after Holm; matched by coordinate search'),
        ('Nine alternative latency formulas', '1--17 additive features; 248 batch-32 configurations of 7 backbones, 5-fold CV and GMAC-band hold-out; 84 planned schedules evaluated', 'better fit but no accuracy gain; some extrapolate badly'),
        ('N4 merge (size-weighted norm)', 'replanned with the same probes on DeiT-B and ViT-B (MAE), latency-paired with M3', 'no gain over M3; MAE-B mean V2 $-0.13$'),
        ('Mean/RMS instead of max norm in M3', 'same schedules; DeiT-B, ViT-B (MAE), ViT-S (AugReg) at two strong budgets', 'up to $-0.62$ vs max at strong compression'),
        ('Very low budgets (0.7--0.9 of fastest history)', 'budgets 10--30\\% below the fastest timed schedule; 21 schedules on 7 backbones', 'only 6/21 meet the budget; large losses'),
    ]
    lines = [r'\begin{table}[t]\centering\small',
             r'\caption{Variants that were tested and did not improve the method. Per-arm results are in the supplementary data.}\label{tab:negative-variants}',
             r'\setlength{\tabcolsep}{3pt}\begin{tabular}{p{0.27\linewidth}p{0.42\linewidth}p{0.24\linewidth}}\toprule Variant & Protocol & Outcome \\ \midrule']
    lines += [f'{a} & {b} & {c} \\\\' for a, b, c in rows]
    lines += [r'\bottomrule\end{tabular}\end{table}']
    write('negative_variants.tex', '\n'.join(lines) + '\n')


# ---------------------------------------------------------------- single-method paper (2026-10-18 revision)
M0_FAMILY = 'LEAF-M0'


def pairs_single():
    """tab:latency-pairs without the earlier variant (its pairs move to tab:leaf-vs-m0); Holm stays over all pairs of a split."""
    rep = load(MAIN_COMPARISON / 'report.json'); summary = {}; n_all = len(rep['tests'])
    for t in rep['tests']:
        if t['family'] == M0_FAMILY: continue
        s = summary.setdefault(t['family'], dict(n=0, vw=0, vl=0, ww=0, wl=0, dv=0., dw=0., bbs=set()))
        s['n'] += 1; s['dv'] += t['val']['delta']; s['dw'] += t['v2']['delta']; s['bbs'].add(t['backbone'])
        if t['val']['holm'] < .05: s['vw' if t['val']['delta'] > 0 else 'vl'] += 1
        if t['v2']['holm'] < .05: s['ww' if t['v2']['delta'] > 0 else 'wl'] += 1
    n = sum(s['n'] for s in summary.values())
    order = ['DiffRate', 'PPT', 'ToFu', 'RAPID', 'ToMe', 'ToMe+M3', 'One-More-Token', 'EViT', 'PiToMe']
    lines = [r'\begin{table}[t]\centering\small',
             rf'\caption{{Latency-paired comparisons of \ours\ with every re-run family ({n} pairs, seven backbones). A pair matches the nearest point of the family within $\tau=\max(0.05\,\mathrm{{ms}},2\,\mathrm{{SD}})$. W/L: significant wins/losses after Holm over all {n_all} pairs of a split (including the 13 pairs with the earlier variant in \cref{{tab:leaf-vs-m0}}). Means are descriptive.}}\label{{tab:latency-pairs}}',
             r'\begin{tabular}{l rr rr rr}\toprule Family & Pairs & Backbones & Val W/L & $\bar\Delta$Val & V2 W/L & $\bar\Delta$V2 \\ \midrule']
    for f in order:
        if f not in summary: continue
        s = summary[f]
        lines.append(f"{f} & {s['n']} & {len(s['bbs'])} & {s['vw']}/{s['vl']} & {s['dv']/s['n']:+.2f} & {s['ww']}/{s['wl']} & {s['dw']/s['n']:+.2f} \\\\")
    lines += [r'\bottomrule\end{tabular}\end{table}']
    write('latency_paired_comparison.tex', '\n'.join(lines) + '\n')
    out = {k: {kk: (sorted(vv) if isinstance(vv, set) else vv) for kk, vv in v.items()} for k, v in summary.items()}
    return dict(families=out, n=n, n_all=n_all)


def m0_variant(by_gmac=False):
    # by_gmac (2026-10-19): identify each pair by the GMAC of \ours's schedule instead of the budget name
    """tab:leaf-vs-m0: \\ours\\ versus the earlier variant (M0 merge, profile SALT) at equal measured latency, one row per pair."""
    rep = load(MAIN_COMPARISON / 'report.json'); rows = [t for t in rep['tests'] if t['family'] == M0_FAMILY]
    lines = [r'\begin{table}[t]\centering\small',
             r'\caption{\ours\ versus its earlier variant \ours-M0 (M0 merge and the profile/GMAC form of \cost; schedules of the earlier sealed campaign), re-timed in the same session and paired at equal measured latency (bs32 ms). Differences are \ours\ minus \ours-M0; \underline{underlined}: significant after Holm over all pairs of the main comparison.}\label{tab:leaf-vs-m0}',
             r'\begin{tabular}{ll rr rr}\toprule Model & Budget & ms & ms (M0) & $\Delta$Val & $\Delta$V2 \\ \midrule']
    if by_gmac:
        lines[-1] = lines[-1].replace('Model & Budget & ms', 'Model & GMAC & ms')
        lines[1] = lines[1].replace('Differences are', 'GMAC: of the \\ours\\ schedule. Differences are')
        jobs = {bb: load(MAIN_COMPARISON / bb / 'plans.json')['jobs'] for bb in BBS}
    for t in sorted(rows, key=lambda t: (BBS.index(t['backbone']), t['ours_ms'])):
        lab = fmt(jobs[t['backbone']][t['ours']]['gmac']) if by_gmac else t['ours'].split('/')[1]
        lines.append(f"{NAMES[t['backbone']]} & {lab} & {fmt(t['ours_ms'])} & {fmt(t['ref_ms'])} & "
                     f"{sig(t['val']['delta'], t['val']['holm'])} & {sig(t['v2']['delta'], t['v2']['holm'])} \\\\")
    lines += [r'\bottomrule\end{tabular}\end{table}']
    write('leaf_vs_m0.tex', '\n'.join(lines) + '\n')
    win = lambda s, sgn: sum(sgn * t[s]['delta'] > 0 and t[s]['holm'] < .05 for t in rows)
    return dict(n=len(rows), val=(win('val', 1), win('val', -1)), v2=(win('v2', 1), win('v2', -1)),
                mean_val=float(np.mean([t['val']['delta'] for t in rows])))


# Published numbers (self-reported; copied from the source papers as in main.tex tab:published). Each group lists the GMAC
# targets at which \ours\ (same planner, GMAC budget) is shown: the nearest GMAC-capped schedule of the main campaign.
PUBLISHED = [
    ('deit_b', [('DiffRate$^\\ast$ (no FT)~\\cite{chen2023diffrate}', '8.7', '78.98', -2.84), ('LTMP$^\\ast$~\\cite{bonnaerens2023ltmp}', '9.00', '78.8', -3.19)], [8.7]),
    ('deit_b', [('ATM~\\cite{lee2025atm}', '9.98', '80.17', -1.62)], [10.0]),
    ('deit_b', [('LTMP$^\\ast$', '11.02', '81.0', -0.99), ('DTEM$^\\ast$ (frozen)~\\cite{lee2024dtem}', '11.1', '81.01', -0.78), ('ATM', '11.21', '81.05', -0.74),
                ('DiffRate$^\\ast$ (no FT)', '11.5', '81.50', -0.32)], [11.5]),
    ('deit_b', [('ATM', '12.41', '81.51', -0.28)], [12.5]),
    ('deit_s', [('MCTF (off-the-shelf, $r{=}16$)~\\cite{lee2024mctf}', '2.60', '79.21', -0.59), ('ATM', '2.67', '79.29', -0.52)], [2.51, 2.71]),
    ('deit_s', [('DTEM$^\\ast$ (frozen)', '2.91', '79.44', -0.39), ('LTMP$^\\ast$', '3.02', '79.6', -0.3), ('Zero-TPrune~\\cite{wang2024zerotprune}', '3.08', '79.4', -0.4),
                ('ATM', '3.13', '79.74', -0.07)], [2.9, 3.1]),
    ('deit_t', [('MCTF (off-the-shelf, $r{=}16$)', '0.71', '70.99', -1.21)], [0.65, 0.73]),
    ('deit_t', [('LTMP$^\\ast$', '0.87', '71.7', -0.5), ('Zero-TPrune', '0.90', '70.4', -0.9), ('ATM', '0.90', '72.04', -0.10)], [0.9]),
    ('mae_b', [('ToMe (r16)~\\cite{kim2024tofu}', '8.78', '78.88', -4.86), ('ToFu (r16)~\\cite{kim2024tofu}', '8.78', '80.70', -3.04)], [8.78]),
    ('mae_b', [('ToMe (r12)', '10.93', '81.82', -1.92), ('ToFu (r12)', '10.93', '82.46', -1.28)], [10.4, 11.5]),
    ('mae_b', [('ATM', '11.5', '82.46', -1.26), ('DiffRate$^\\ast$ (no FT)', '11.5', '82.91', -0.81), ('DTEM$^\\ast$ (frozen)', '11.6', '82.80', -0.92)], [11.5]),
    ('mae_l', [('ToMe (r8)~\\cite{kim2024tofu}', '30.99', '84.22', -1.73), ('ToFu (r8)~\\cite{kim2024tofu}', '30.99', '84.67', -1.28)], [31.0]),
    ('augreg_s', [('ToMe (r13)~\\cite{kim2024tofu}', '2.7', '79.3', None), ('ToFu (r13)~\\cite{kim2024tofu}', '2.7', '79.6', None)], [2.56, 2.84]),
    ('augreg_b', [('ToMe (r14)~\\cite{choi2026rapid}', '9.84$^\\dagger$', '83.41', -1.70), ('ToFu (r14)~\\cite{choi2026rapid}', '9.84$^\\dagger$', '83.72', -1.39),
                  ('RAPID (r14)~\\cite{choi2026rapid}', '9.84$^\\dagger$', '84.17', -0.94)], [9.84]),
    ('augreg_b', [('ToMe (r11)', '11.47$^\\dagger$', '84.13', -0.98), ('ToFu (r11)', '11.47$^\\dagger$', '84.25', -0.86), ('RAPID (r11)', '11.47$^\\dagger$', '84.61', -0.50)], [10.83, 11.94]),
]


def published_single(positive_drop=False, with_dense=False):
    """tab:published with \\ours\\ planned under a GMAC budget (M3, P+D; the ref/M3/gmac arms of the main campaign)."""
    # positive_drop (2026-10-19): print the drop as dense minus top-1 (>= 0, lower is better) instead of the signed change
    ours, dense = {}, {}
    for bb in BBS:
        jobs = load(MAIN_COMPARISON / bb / 'plans.json')['jobs']
        dense[bb] = float(correct_(bb, jobs['ref/dense']['config_sha']).mean() * 100)
        ours[bb] = sorted((row['gmac'], float(correct_(bb, row['config_sha']).mean() * 100), name) for name, row in jobs.items() if name.startswith('ref/M3/gmac/'))
    lines = [r'\begin{table}[t]', r'\centering\footnotesize\renewcommand{\arraystretch}{0.92}',
             r"\caption{Published numbers of training-free or frozen-backbone methods (self-reported, GMAC as stated) beside \ours\ planned under a GMAC budget instead of a latency budget (same probes, score and M3 operator; schedules of the main campaign at the nearest GMAC). $\Delta$: drop from each source's own dense accuracy (``--'': the source reports no dense accuracy). $^\ast$ learns a few parameters (thresholds, embeddings or rates) on a frozen backbone. $^\dagger$ the source reports throughput only; GMAC computed by us from the token counts of ToMe at the same $r$. Bold: smallest drop when \ours\ is linearly interpolated along its GMAC-capped schedules to the GMAC of the best published entry of the group.}",
             r'\label{tab:published}', r'\begin{tabular}{llrrr}', r'\toprule', r'Backbone & Method & GMAC & Top-1 & $\Delta$\\', r'\midrule']
    ev, prev = [], None
    for bb, pubs, targets in PUBLISHED:
        picks = []
        for t in targets:
            g = min(ours[bb], key=lambda x: abs(x[0] - t))
            if g not in picks: picks.append(g)
        mine = [(g, a, a - dense[bb]) for g, a, _ in picks]
        known = [(float(g.split('$')[0]), d) for _, g, _, d in pubs if d is not None]
        if known:   # compare at the GMAC of the best published entry, ours interpolated along its GMAC-capped schedules
            g_ref, d_ref = max(known, key=lambda x: x[1])
            xs, ys = zip(*[(g, a - dense[bb]) for g, a, _ in ours[bb]]); ours_eff = float(np.interp(g_ref, xs, ys))
            best = ours_eff if ours_eff > d_ref else d_ref
        else:
            g_ref = d_ref = ours_eff = best = None
        if prev is not None: lines.append(r'\midrule' if bb != prev else r'\cmidrule{2-5}')
        named = False
        if with_dense and bb != prev:   # 2026-10-19: our dense model, evaluated in the main campaign
            lines.append(f"{NAMES[bb]} & Dense (our evaluation) & {dense_ref(bb)['gmac']:.2f} & {dense[bb]:.2f} & --\\\\"); named = True
        for i, (m, g, a, d) in enumerate(pubs):
            ds = '--' if d is None else (f'$\\mathbf{{{d:+.2f}}}$' if d == best else f'${d:+.2f}$')
            if positive_drop and d is not None: ds = f'$\\mathbf{{{-d:.2f}}}$' if d == best else f'${-d:.2f}$'
            lines.append(f"{NAMES[bb] if (i == 0 and bb != prev and not named) else ''} & {m} & {g} & {a} & {ds}\\\\")
        gs = ' / '.join(f'{g:.2f}' for g, *_ in mine); acc = ' / '.join(f'{a:.2f}' for _, a, _ in mine)
        ds = ' / '.join(f'$\\mathbf{{{d:+.2f}}}$' if best is not None and best == ours_eff else f'${d:+.2f}$' for *_, d in mine)
        if positive_drop: ds = ' / '.join(f'$\\mathbf{{{-d:.2f}}}$' if best is not None and best == ours_eff else f'${-d:.2f}$' for *_, d in mine)
        lines.append(f'& \\textbf{{Ours}} & {gs} & {acc} & {ds}\\\\'); prev = bb
        ev.append(dict(bb=bb, targets=targets, ref_gmac=g_ref, ours_at_ref=ours_eff, ours_wins=None if best is None else best == ours_eff, ours=[dict(gmac=g, top1=a, drop=d) for g, a, d in mine], best_published=max([d for *_, d in pubs if d is not None], default=None)))
    if positive_drop:
        lines[2] = lines[2].replace("$\\Delta$: drop from each source's own dense accuracy", "Drop: each source's own dense accuracy minus its top-1, in points (lower is better)")
        lines[6] = lines[6].replace(r'& $\Delta$\\', r'& Drop$\downarrow$\\')
    if with_dense:
        lines[2] = lines[2].replace(' $^\\ast$ learns', ' Dense (our evaluation): the unreduced checkpoint in our pipeline, against which our drops are computed. $^\\ast$ learns')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    write('published_methods.tex', '\n'.join(lines) + '\n')
    return dict(dense=dense, groups=ev)


REAL = REPO / 'results/imagenet_real/report.json'


def pairs_single_real():
    """tab:latency-pairs with ImageNet-ReaL columns (leaf/reproduction/experiments/imagenet_real.py re-scores the stored val predictions with ReaL labels)."""
    rep = load(MAIN_COMPARISON / 'report.json'); real = {(t['backbone'], t['ours'], t['ref']): t['real'] for t in load(REAL)['tests']}
    summary = {}; n_all = len(rep['tests'])
    for t in rep['tests']:
        if t['family'] == M0_FAMILY: continue
        s = summary.setdefault(t['family'], dict(n=0, vw=0, vl=0, ww=0, wl=0, rw=0, rl=0, dv=0., dw=0., dr=0., bbs=set()))
        r = real[(t['backbone'], t['ours'], t['ref'])]
        s['n'] += 1; s['dv'] += t['val']['delta']; s['dw'] += t['v2']['delta']; s['dr'] += r['delta']; s['bbs'].add(t['backbone'])
        if t['val']['holm'] < .05: s['vw' if t['val']['delta'] > 0 else 'vl'] += 1
        if t['v2']['holm'] < .05: s['ww' if t['v2']['delta'] > 0 else 'wl'] += 1
        if r['holm'] < .05: s['rw' if r['delta'] > 0 else 'rl'] += 1
    n = sum(s['n'] for s in summary.values())
    order = ['DiffRate', 'PPT', 'ToFu', 'RAPID', 'ToMe', 'ToMe+M3', 'One-More-Token', 'EViT', 'PiToMe']
    lines = [r'\begin{table}[t]\centering\small',
             rf'\caption{{Latency-paired comparisons of \ours\ with every re-run family ({n} pairs, seven backbones) on ImageNet val, ImageNet-ReaL labels~\cite{{beyer2020real}} and ImageNet-V2. A pair matches the nearest point of the family within $\tau=\max(0.05\,\mathrm{{ms}},2\,\mathrm{{SD}})$. W/L: significant wins/losses after Holm over all {n_all} pairs of a split (including the 13 pairs with the earlier variant in \cref{{tab:leaf-vs-m0}}). Means are descriptive.}}\label{{tab:latency-pairs}}',
             r'\setlength{\tabcolsep}{4pt}\begin{tabular}{l rr rr rr rr}\toprule & & & \multicolumn{2}{c}{Val} & \multicolumn{2}{c}{ReaL} & \multicolumn{2}{c}{V2} \\ \cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}',
             r'Family & Pairs & Backbones & W/L & $\bar\Delta$ & W/L & $\bar\Delta$ & W/L & $\bar\Delta$ \\ \midrule']
    for f in order:
        if f not in summary: continue
        s = summary[f]
        lines.append(f"{f} & {s['n']} & {len(s['bbs'])} & {s['vw']}/{s['vl']} & {s['dv']/s['n']:+.2f} & {s['rw']}/{s['rl']} & {s['dr']/s['n']:+.2f} & {s['ww']}/{s['wl']} & {s['dw']/s['n']:+.2f} \\\\")
    lines += [r'\bottomrule\end{tabular}\end{table}']
    write('latency_paired_comparison.tex', '\n'.join(lines) + '\n')
    return {k: {kk: (sorted(vv) if isinstance(vv, set) else vv) for kk, vv in v.items()} for k, v in summary.items()}


EQG = REPO / 'results/equal_gmac/report.json'
EQG_FAMS = ['DiffRate', 'RAPID', 'PiToMe', 'ToMe', 'ToMe+M3', 'ToFu', 'PPT', 'EViT', 'One-More-Token', 'MCTF']


def equal_gmac_summary():
    """tab:eqgmac-pairs: per family, LEAF under a GMAC budget versus the family point at the same GMAC (leaf/reproduction/experiments/equal_gmac.py)."""
    rep = load(EQG); s = rep['summary']; n = len(rep['pairs'])
    lines = [r'\begin{table}[t]\centering\small',
             rf"\caption{{Equal-GMAC comparison: \ours\ planned under a GMAC budget (same probes, score and operator) against the point of each family with the nearest GMAC (within 3\%), all timed in the session of \cref{{tab:latency-pairs}} (bs32). $K$: stations (blocks with a reduction step). Faster/slower: latency difference beyond $\tau$. $\bar\Delta$ms: mean relative latency difference (negative: \ours\ faster). W/L: significant accuracy wins/losses (McNemar, Holm over all {n} pairs per split). MCTF's adapter is slower than the dense model.}}\label{{tab:eqgmac-pairs}}",
             r'\setlength{\tabcolsep}{3pt}\small\begin{tabular}{l cc r rr r rr r}\toprule',
             r'& \multicolumn{2}{c}{$K$} & & \multicolumn{3}{c}{Latency} & \multicolumn{3}{c}{Accuracy} \\ \cmidrule(lr){2-3}\cmidrule(lr){5-7}\cmidrule(lr){8-10}',
             r'Family & theirs & ours & Pairs & faster & slower & $\bar\Delta$ms & Val W/L & $\bar\Delta$Val & V2 W/L \\ \midrule']
    for f in EQG_FAMS:
        if f not in s: continue
        r = s[f]; kt = f"{r['ref_K'][0]}" if r['ref_K'][0] == r['ref_K'][1] else f"{r['ref_K'][0]}--{r['ref_K'][1]}"
        lines.append(f"{f} & {kt} & {r['K'][0]}--{r['K'][1]} & {r['n']} & {r['faster']} & {r['slower']} & {100 * r['rel_ms_mean']:+.1f}\\% & "
                     f"{r['val_w']}/{r['val_l']} & {r['dval_mean']:+.2f} & {r['v2_w']}/{r['v2_l']} \\\\")
    lines += [r'\bottomrule\end{tabular}\end{table}']
    write('equal_gmac_pairs.tex', '\n'.join(lines) + '\n')
    fit = rep['station_fit']
    return dict(n=n, summary=s, us_per_station={bb: 1000 * f['ms_per_station'] for bb, f in fit.items()},
                r2={bb: f['r2'] for bb, f in fit.items()})


def equal_gmac_main(groups=(('deit_b', 'DR8.7'), ('deit_b', 'DR11.5'), ('mae_b', 'DR8.7'), ('mae_b', 'DR11.5'))):
    """tab:eqgmac-methods: every family side by side at four GMAC levels (DiffRate anchors), K, GMAC, latency and val."""
    rep = load(EQG); col = {}
    for bb, bid in groups:
        ps = [p for p in rep['pairs'] if p['backbone'] == bb and p['ours'] == f'ref/M3/gmac/{bid}']
        col[(bb, bid)] = ({p['family']: p for p in ps}, ps[0] if ps else None)
    head = ' & '.join(rf"\multicolumn{{4}}{{c}}{{{NAMES[bb]}, $\approx${col[(bb, bid)][1]['gmac']:.1f} GMAC}}" for bb, bid in groups)
    rule = ''.join(rf'\cmidrule(lr){{{2 + 4 * i}-{5 + 4 * i}}}' for i in range(len(groups)))
    lines = [r'\begin{table*}[t]\centering\footnotesize',
             r"\caption{Methods at the same GMAC (nearest point within 3\% of \ours\ planned under a GMAC budget), timed in one session per backbone (bs32 ms, RTX~5060~Ti). $K$: stations. $^\dagger$: significantly less accurate than \ours\ on ImageNet val (McNemar, Holm over all equal-GMAC pairs); --: no point of the family within 3\% (not timed or outside the grid). All equal-GMAC pairs are summarised in \cref{tab:eqgmac-pairs}.}\label{tab:eqgmac-methods}",
             r'\setlength{\tabcolsep}{2.6pt}\resizebox{\textwidth}{!}{\begin{tabular}{l ' + ' '.join(['rrrr'] * len(groups)) + r'}\toprule',
             ' & ' + head + r' \\ ' + rule, 'Method' + r' & $K$ & GMAC & ms & Val' * len(groups) + r' \\ \midrule']
    for f in EQG_FAMS + ['ours']:
        cells = []
        for g in groups:
            fams, ref = col[g]
            if f == 'ours':
                cells.append(f"\\textbf{{{ref['K']}}} & \\textbf{{{ref['gmac']:.2f}}} & \\textbf{{{ref['ms']:.2f}}} & \\textbf{{{ref['val']['ours']:.2f}}}"); continue
            p = fams.get(f)
            if p is None: cells.append('-- & -- & -- & --'); continue
            mark = r'$^\dagger$' if p['val']['holm'] < .05 and p['val']['delta'] > 0 else ''
            cells.append(f"{p['ref_K']} & {p['ref_gmac']:.2f} & {p['ref_ms']:.2f} & {p['val']['ref']:.2f}{mark}")
        if all(c.startswith('--') for c in cells): continue
        lines.append(('\\midrule \\ours\\ (GMAC budget)' if f == 'ours' else f) + ' & ' + ' & '.join(cells) + r' \\')
    lines += [r'\bottomrule\end{tabular}}\end{table*}']
    write('equal_gmac_methods.tex', '\n'.join(lines) + '\n')


def eqg_best(fams, ref):
    """Best printed value per column of one equal-GMAC group: fewest stations, lowest ms, highest val."""
    ps = [p for f, p in fams.items() if f in EQG_FAMS]
    return dict(K=min([ref['K']] + [p['ref_K'] for p in ps]), ms=min([f"{ref['ms']:.2f}"] + [f"{p['ref_ms']:.2f}" for p in ps], key=float),
                val=max([f"{ref['val']['ours']:.2f}"] + [f"{p['val']['ref']:.2f}" for p in ps], key=float))


def eqg_cells(K, gmac, ms, val, best, mark=''):
    """One group's four cells with the column bests in bold (GMAC is the matching variable and is never bold)."""
    b = lambda x, ok: f'\\textbf{{{x}}}' if ok else f'{x}'
    ms, val = f'{ms:.2f}', f'{val:.2f}'
    return f"{b(K, K == best['K'])} & {gmac:.2f} & {b(ms, ms == best['ms'])} & {b(val, val == best['val'])}{mark}"


def equal_gmac_main_stacked(blocks=((('deit_b', 'DR8.7'), ('deit_b', 'DR11.5')), (('mae_b', 'DR8.7'), ('mae_b', 'DR11.5'))), bold='row', with_dense=False):
    # bold='best' (2026-10-19): bold only each column's best value instead of the whole \ours\ row
    """tab:eqgmac-methods as two stacked 9-column blocks (one backbone each), readable in a one-column layout."""
    rep = load(EQG)
    def col(bb, bid):
        ps = [p for p in rep['pairs'] if p['backbone'] == bb and p['ours'] == f'ref/M3/gmac/{bid}']
        return {p['family']: p for p in ps}, ps[0]
    lines = [r'\begin{table*}[t]\centering\small',
             r"\caption{Methods at the same GMAC (nearest point within 3\% of \ours\ planned under a GMAC budget), timed in one session per backbone (bs32 ms, RTX~5060~Ti). $K$: stations. $^\dagger$: significantly less accurate than \ours\ on ImageNet val (McNemar, Holm over all equal-GMAC pairs); --: no point of the family within 3\% (not timed or outside the grid). All equal-GMAC pairs are summarised in \cref{tab:eqgmac-pairs}.}\label{tab:eqgmac-methods}",
             r'\setlength{\tabcolsep}{4pt}']
    for k, groups in enumerate(blocks):
        cols = {g: col(*g) for g in groups}
        head = ' & '.join(rf"\multicolumn{{4}}{{c}}{{{NAMES[bb]}, $\approx${cols[(bb, bid)][1]['gmac']:.1f} GMAC}}" for bb, bid in groups)
        rule = ''.join(rf'\cmidrule(lr){{{2 + 4 * i}-{5 + 4 * i}}}' for i in range(len(groups)))
        if k: lines.append(r'\par\medskip')
        lines += [r'\begin{tabular}{l ' + ' '.join(['rrrr'] * len(groups)) + r'}\toprule', ' & ' + head + r' \\ ' + rule,
                  'Method' + r' & $K$ & GMAC & ms & Val' * len(groups) + r' \\ \midrule']
        if with_dense:   # 2026-10-19: the unreduced model of the same session, not part of the comparison
            d = {bb: dense_ref(bb) for bb, _ in groups}
            lines.append('Dense & ' + ' & '.join(f"0 & {d[bb]['gmac']:.2f} & {d[bb]['ms']:.2f} & {d[bb]['val']:.2f}" for bb, _ in groups) + r' \\ \midrule')
        for f in EQG_FAMS + ['ours']:
            cells = []
            for g in groups:
                fams, ref = cols[g]
                if f == 'ours' and bold == 'best':
                    cells.append(eqg_cells(ref['K'], ref['gmac'], ref['ms'], ref['val']['ours'], eqg_best(fams, ref))); continue
                if f == 'ours':
                    cells.append(f"\\textbf{{{ref['K']}}} & \\textbf{{{ref['gmac']:.2f}}} & \\textbf{{{ref['ms']:.2f}}} & \\textbf{{{ref['val']['ours']:.2f}}}"); continue
                p = fams.get(f)
                if p is None: cells.append('-- & -- & -- & --'); continue
                mark = r'$^\dagger$' if p['val']['holm'] < .05 and p['val']['delta'] > 0 else ''
                if bold == 'best':
                    cells.append(eqg_cells(p['ref_K'], p['ref_gmac'], p['ref_ms'], p['val']['ref'], eqg_best(fams, ref), mark)); continue
                cells.append(f"{p['ref_K']} & {p['ref_gmac']:.2f} & {p['ref_ms']:.2f} & {p['val']['ref']:.2f}{mark}")
            if all(c.startswith('--') for c in cells): continue
            lines.append(('\\midrule \\ours\\ (GMAC budget)' if f == 'ours' else f) + ' & ' + ' & '.join(cells) + r' \\')
        lines.append(r'\bottomrule\end{tabular}')
    lines.append(r'\end{table*}')
    if bold == 'best':
        lines[1] = lines[1].replace('All equal-GMAC pairs', 'Bold: fewest stations, lowest latency and highest accuracy of each group. All equal-GMAC pairs')
    if with_dense:
        lines[1] = lines[1].replace(' All equal-GMAC pairs', ' Dense: the unreduced model, timed in the same session (not bolded). All equal-GMAC pairs')
    write('equal_gmac_methods.tex', '\n'.join(lines) + '\n')


def equal_gmac_dr(with_dense=False):
    """tab:eqgmac-diffrate (appendix): \\ours\\ (GMAC budget) versus DiffRate at every released schedule, same GMAC."""
    rep = load(EQG); ps = sorted([p for p in rep['pairs'] if p['family'] == 'DiffRate'], key=lambda p: (BBS.index(p['backbone']), p['gmac']))
    lines = [r'\begin{table}[h]\centering\small',
             r'\caption{\ours\ under a GMAC budget versus every released DiffRate schedule at the same GMAC, one timing session per backbone (bs32 ms). $\Delta$ms: \ours\ minus DiffRate. \underline{Underlined}/\textbf{bold}: significantly higher/lower accuracy (Holm over all equal-GMAC pairs).}\label{tab:eqgmac-diffrate}',
             r'\setlength{\tabcolsep}{3pt}\begin{tabular}{l rrr rrr r rr}\toprule',
             r'& \multicolumn{3}{c}{DiffRate} & \multicolumn{3}{c}{\ours\ (GMAC budget)} & & & \\ \cmidrule(lr){2-4}\cmidrule(lr){5-7}',
             r'Model & $K$ & GMAC & ms & $K$ & GMAC & ms & $\Delta$ms & $\Delta$Val & $\Delta$V2 \\ \midrule']
    prev = None
    for p in ps:
        if with_dense and p['backbone'] != prev:   # 2026-10-19: dense reference of each backbone as a group header
            d = dense_ref(p['backbone'])
            lines.append(('' if prev is None else r'\midrule ') + rf"\multicolumn{{10}}{{l}}{{{NAMES[p['backbone']]}: dense {d['gmac']:.2f}~GMAC, {d['ms']:.2f}~ms, val {d['val']:.2f}, V2 {d['v2']:.2f}}} \\")
        lines.append(f"{NAMES[p['backbone']] if (p['backbone'] != prev and not with_dense) else ''} & {p['ref_K']} & {p['ref_gmac']:.2f} & {p['ref_ms']:.2f} & {p['K']} & {p['gmac']:.2f} & {p['ms']:.2f} & "
                     f"{p['dms']:+.2f} & {sig(p['val']['delta'], p['val']['holm'])} & {sig(p['v2']['delta'], p['v2']['holm'])} \\\\"); prev = p['backbone']
    lines += [r'\bottomrule\end{tabular}\end{table}']
    write('equal_gmac_vs_diffrate.tex', '\n'.join(lines) + '\n')


def dense_ref(bb):
    """Dense model of a backbone in the main session (2026-10-19): GMAC (config_cost convention), bs32 ms, val and V2 top-1."""
    from leaf.reproduction.experiments.campaign_reports import round_means
    from leaf.planner.gmac import dense_cost
    from leaf.backbones import SPECS
    sha = load(MAIN_COMPARISON / bb / 'plans.json')['jobs']['ref/dense']['config_sha']
    ms = round_means(MAIN_COMPARISON / bb / 'latency', 32)[f'{bb}-{sha}']
    return dict(gmac=float(dense_cost(SPECS[bb])), ms=float(ms[0]), val=float(correct_(bb, sha, 'val').mean() * 100),
                v2=float(correct_(bb, sha, 'v2').mean() * 100))


def correct_(bb, sha, split='val'):
    from leaf.reproduction.experiments.campaign_reports import correct
    return correct(MAIN_COMPARISON / bb, sha, split)


if __name__ == '__main__':
    ev = dict(anchors=anchors(by_gmac=True), pairs=pairs_single(), pairs_real=pairs_single_real(), m0=m0_variant(by_gmac=True), published=published_single(positive_drop=True, with_dense=True), equal_gmac=equal_gmac_summary(), pricing=pricing(), seg=seg(), cost=cost())
    ablation(by_gmac=True, dense=True); negative(); equal_gmac_main_stacked(bold='best', with_dense=True); equal_gmac_dr(with_dense=True)
    from typesetting import arrow_audit; arrow_audit(OUT)   # 2026-10-19: no arrows on matching columns
    (REPO / 'results/paper_build/table_evidence.json').write_text(json.dumps(ev, indent=1, default=float))
    a = ev['anchors']; print('anchors', a['n_anchor'], 'dominate', a['dominate_val'], 'worse', a['worse'])
