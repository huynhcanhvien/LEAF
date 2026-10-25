"""Typesetting passes of the paper tables (leaf.reproduction.report.tables, leaf.reproduction.report.build).

metric_arrows(table_tex): marks metric column headers with $\\uparrow$ (higher is better) or $\\downarrow$ (lower is better).
fix_breaks(tex): keeps numbers with their units, model and method names, and short inline relations on one line.
Both are idempotent, so a table or text can pass through them more than once.
"""
import re

UP, DOWN = r'$\uparrow$', r'$\downarrow$'

# header cells, matched as whole cells (between '&' / line start and '&' / '\\')
HEADER_CELLS = {
    'Val': UP, 'V2': UP, 'Top-1': UP, 'mIoU': UP,
    'ms': DOWN, 'GMAC': DOWN, 'ms (M0)': DOWN, 'bs1 ms': DOWN, 'bs8 ms': DOWN, 'CV SD': DOWN,
    r'$\Delta$Val': UP, r'$\Delta$V2': UP, r'$\bar\Delta$Val': UP, r'$\bar\Delta$V2': UP,
    r'$\Delta$Val range': UP, r'$\Delta$': UP, r'$\bar\Delta$': UP, r'$\Delta$ms': DOWN, r'$\bar\Delta$ms': DOWN,
}
# phrases that may sit inside a longer cell or a row label
PHRASES = {
    r'Latency MAPE (\%)': DOWN, '(img/s)': UP, '(GPU-h)': DOWN,
    r'\multicolumn{2}{c}{$\bar\Delta_{\rm V2}$}': None,     # handled below (multicolumn group head)
}


def _mark_cell(cell):
    core = cell.strip()
    if core in HEADER_CELLS and not core.endswith(('$\\uparrow$', '$\\downarrow$')):
        return cell.replace(core, core + HEADER_CELLS[core])
    return cell


def metric_arrows(tex):
    """Add arrows to the column-header rows of every tabular/longtable in `tex` (rows before the first \\midrule)."""
    out, pos = [], 0
    for m in re.finditer(r'\\begin\{(tabular|longtable)\}', tex):
        if m.start() < pos: continue
        end = tex.find('\\midrule', m.end())
        close = tex.find('\\end{' + m.group(1) + '}', m.end())
        if end < 0 or (0 <= close < end): continue      # no header rule in this table (e.g. the notation table)
        head = tex[m.end():end]
        rows = re.split(r'(\\\\)', head)
        rows = ['&'.join(_mark_cell(c) for c in re.split(r'(?<!\\)&', r)) if r != '\\\\' else r for r in rows]
        head = ''.join(rows)
        out.append(tex[pos:m.end()] + head); pos = end
    tex = ''.join(out) + tex[pos:]
    for k, arrow in PHRASES.items():
        if arrow and k in tex and k + arrow not in tex: tex = tex.replace(k, k + arrow)
    tex = tex.replace(r'\multicolumn{2}{c}{$\bar\Delta_{\rm V2}$}', r'\multicolumn{2}{c}{$\bar\Delta_{\rm V2}\uparrow$}')
    return tex


UNITS = r'(?:GMAC|GPU-hours|GPU-h|points?|stations?|tokens?|blocks?|images|pairs?|epochs?|budgets?|backbones|families|heads|workloads|schedules|rounds)'
RELATION = re.compile(r'=|<|>|\\le\b|\\ge\b|\\leq|\\geq|\\approx|\\ne\b')


def _text_pass(s):
    s = re.sub(r'(?<![\w\\{-])((?:MAE )?(?:DeiT|ViT))-([TSBLHG](?:/\d+)?)\b', r'\\mbox{\1-\2}', s)
    s = re.sub(r'(?<!\\mbox\{)ImageNet-V2', r'\\mbox{ImageNet-V2}', s)   # robust in captions (\\nobreakdash is not)
    s = re.sub(r'Pruning One More Token', r'Pruning~One~More~Token', s)
    s = re.sub(r'(?<!~)One More Token', r'One~More~Token', s)
    s = re.sub(r'(\d(?:[\d.,]*\d)?(?:\\%)?)[ ]+(' + UNITS + r')\b', r'\1~\2', s)
    s = re.sub(r'\b(batch|batch size|bs|block|blocks|Table|Figure|rank|budget)[ ]+(\d)', r'\1~\2', s)
    s = re.sub(r'\bImageNet[ ]+(val|V2)\b', r'ImageNet~\1', s)
    s = re.sub(r'\b(GPU|CPU)[ ]+(batch)\b', r'\1~\2', s)
    s = re.sub(r'\bRTX[ ~]+5060[ ~]+Ti\b', r'RTX~5060~Ti', s)
    return s


def fix_breaks(tex):
    """Apply line-break protection outside math, comments and command arguments that are keys (labels, refs, files)."""
    lines = tex.split('\n'); out = []
    in_display = 0
    for line in lines:
        if re.match(r'\s*%', line):
            out.append(line); continue
        if re.search(r'\\begin\{(equation|align|gather|multline|eqnarray)\*?\}', line): in_display += 1
        if in_display:
            out.append(line)
            if re.search(r'\\end\{(equation|align|gather|multline|eqnarray)\*?\}', line): in_display -= 1
            continue
        # split into text / inline math / protected command arguments
        parts = re.split(r'((?<!\\)\$[^$]*?(?<!\\)\$|\\(?:label|ref|cref|Cref|eqref|cite|citep|citet|includegraphics(?:\[[^\]]*\])?|input|url|href|texttt|email|bibliography)\{[^}]*\})', line)
        new = []
        for i, p in enumerate(parts):
            if i % 2 == 1 and not p.startswith('$'):     # captured key argument (label, ref, cite, file, url)
                new.append(p); continue
            if i % 2 == 1:
                inner = p[1:-1]
                if (RELATION.search(inner) and len(inner) <= 26 and '\\\\' not in inner and '&' not in inner
                        and not (new and new[-1].endswith('\\mbox{'))):
                    p = r'\mbox{' + p + '}'
                new.append(p)
            else:
                new.append(_text_pass(p))
        line = ''.join(new)
        out.append(line)
    return '\n'.join(out)


def plain_headers(tex, cells):
    """Remove the metric arrow from the given header cells (2026-10-19): columns that only identify or match points,
    e.g. GMAC in an equal-GMAC table or ms in an equal-latency pairing, are not outcomes and carry no arrow."""
    out, pos = [], 0
    for m in re.finditer(r'\\begin\{(tabular|longtable)\}', tex):
        if m.start() < pos: continue
        end = tex.find('\\midrule', m.end())
        close = tex.find('\\end{' + m.group(1) + '}', m.end())
        if end < 0 or (0 <= close < end): continue
        head = tex[m.end():end]
        for c in cells:
            for a in (UP, DOWN):
                head = re.sub(r'(^|&)(\s*)' + re.escape(c + a) + r'(\s*)(?=&|\\\\|$)', lambda mm: mm.group(1) + mm.group(2) + c + mm.group(3), head, flags=re.M)
        out.append(tex[pos:m.end()] + head); pos = end
    return ''.join(out) + tex[pos:]


# tables whose matching/identifier columns carry no arrow, and arrows added where a metric header has none (2026-10-19)
PLAIN = {
    'leaf_vs_m0.tex': ['GMAC', 'ms', 'ms (M0)'],
    'equal_gmac_methods.tex': ['GMAC'], 'equal_gmac_vs_diffrate.tex': ['GMAC'], 'published_methods.tex': ['GMAC'],
    'results_all_budgets.tex': ['GMAC'], 'main_dr_m0.tex': ['GMAC'], 'latency_pricing.tex': ['GMAC'],
}
ADD = {
    'forecast_rank.tex': [(r'\multicolumn{2}{c}{Energy (\%)}', r'\multicolumn{2}{c}{Energy (\%)$\uparrow$}'),
                          (r'& Relative error\\', r'& Relative error$\downarrow$\\')],
    'forecast_quality.tex': [('bootstrap intervals are below $\\pm0.02$).', 'bootstrap intervals are below $\\pm0.02$); higher is better.')],
    'results_all_budgets.tex': [(r'\toprule GMAC & ms & Val & V2 & $L$ (ms) & $K$ & GMAC & ms & Val & V2 \\',
                     r'\toprule GMAC & ms$\downarrow$ & Val$\uparrow$ & V2$\uparrow$ & $L$ (ms) & $K$ & GMAC & ms$\downarrow$ & Val$\uparrow$ & V2$\uparrow$ \\')],
}


def arrow_audit(tables_dir):
    """Apply PLAIN and ADD to the generated tables in place (idempotent)."""
    from pathlib import Path
    for name in set(PLAIN) | set(ADD):
        p = Path(tables_dir) / name
        if not p.exists(): continue
        t = p.read_text()
        if name in PLAIN: t = plain_headers(t, PLAIN[name])
        for a, b in ADD.get(name, []):
            if b not in t: t = t.replace(a, b)
        p.write_text(t)
