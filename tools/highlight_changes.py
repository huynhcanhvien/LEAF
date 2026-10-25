"""Mark our changes to the paper in blue, for the mentor.

    python tools/highlight_changes.py              # paper/edited-main.tex + paper/main.tex -> paper/main-highlight.tex

The marks are switched by \\showchangestrue / \\showchangesfalse in the preamble; set it to false before submission.
main.tex itself is not changed; re-run the script after every edit of main.tex. Marks left in the input are removed first.

Lines of main.tex that have no counterpart in the mentor's edited-main.tex are coloured as a whole; in lines that were
edited, only the sentences that are new or changed are coloured. Deleted text (mostly the resolved \\cmt comments) is not
shown. Tables whose content changed are coloured as a whole and listed, with the layout-only changes, in a note after the
title.
"""
import argparse
import difflib
import re
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1] / 'paper'
NEW, OLD = r'\leafnew{}', r'\leafold{}'
PREAMBLE = (r'\newif\ifshowchanges\showchangestrue   % blue marks of the changes after the mentor\'s v3 (tools/highlight_changes.py); false before submission' '\n'
            r'\ifshowchanges\newcommand{\leafnew}{\color{blue}}\newcommand{\leafold}{\color{black}}\else\newcommand{\leafnew}{}\newcommand{\leafold}{}\fi' '\n')
SKIP = re.compile(r'\s*(%|\\label|\\input|\\begin\{(table|figure|longtable)|\\end\{(table|figure|longtable)|$)')
# tables with new content (coloured) and tables whose layout or font size only changed (listed), relative to the mentor's v3
CONTENT = []   # none since the mentor's v4 (2026-10-22); v3 round: efficiency, latency_pricing, search_cost, ...
LAYOUT = []
SENT = re.compile(r'(?<=[.;:])(\s+)(?=[A-Z\\$(])')


def sentences(line):
    """[(sentence, separator)] with the separators kept, so that ''.join restores the line."""
    parts = SENT.split(line)
    return [(parts[i], parts[i + 1] if i + 1 < len(parts) else '') for i in range(0, len(parts), 2)]


def table_row(line):
    return '&' in line and line.rstrip().endswith('\\\\')


def colour(text):
    return text if not text.strip() or SKIP.match(text) or table_row(text) else NEW + text + OLD


def edited(old_lines, new_lines):
    """Colour the sentences of new_lines that do not occur in old_lines."""
    old = [s for l in old_lines for s, _ in sentences(l)]
    out = []
    for line in new_lines:
        if SKIP.match(line) or table_row(line): out.append(line); continue   # colour cannot span a table row end
        sents = sentences(line)
        sm = difflib.SequenceMatcher(None, old, [s for s, _ in sents], autojunk=False)
        fresh = set()
        for tag, _, _, j1, j2 in sm.get_opcodes():
            if tag in ('replace', 'insert'): fresh.update(range(j1, j2))
        out.append(''.join((colour(s) if k in fresh else s) + sep for k, (s, sep) in enumerate(sents)))
    return out


def note():
    t = lambda xs: ', '.join(r'\texttt{' + x.replace('_', r'\_') + '}' for x in xs)
    tables = ((r' Tables with new content are blue: ' + t(CONTENT) + '.' if CONTENT else '') +
              (r' Tables with a new layout or font size only: ' + t(LAYOUT) + '.' if LAYOUT else ''))
    return ('\n' r'\ifshowchanges\begin{center}\fbox{\parbox{0.92\linewidth}{\small\textbf{Reading copy for the mentor.} '
            r'Text in {\color{blue}blue} is new or changed relative to \texttt{edited-main.tex}; deleted text, '
            r'mostly the resolved red comments, is not shown.' + tables + r'}}\end{center}\fi' '\n')


def unmark(text):
    """main.tex without the marks of an earlier run."""
    text = text.replace(PREAMBLE, '').replace(note(), '')
    return text.replace(NEW, '').replace(OLD, '')


def build(old_path, new_path, out_path):
    old, new = old_path.read_text().split('\n'), unmark(new_path.read_text()).split('\n')
    out = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, old, new, autojunk=False).get_opcodes():
        if tag == 'equal': out += new[j1:j2]
        elif tag == 'insert': out += [colour(l) for l in new[j1:j2]]
        elif tag == 'replace': out += edited(old[i1:i2], new[j1:j2])
    text = '\n'.join(out)
    head, body = text.split(r'\begin{document}', 1)   # the colour macros exist only inside the document
    text = head.replace(NEW, '').replace(OLD, '') + r'\begin{document}' + body
    text = text.replace(r'\begin{document}', PREAMBLE + r'\begin{document}', 1).replace(r'\maketitle', r'\maketitle' + note(), 1)
    for name in CONTENT:
        text = re.sub(r'(\{\\small)?\\input\{tables/' + name + r'\.tex\}\}?', lambda m: NEW + m.group(0) + OLD, text)
    out_path.write_text(text)
    print('wrote', out_path)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--old', default=PAPER / 'edited-main.tex', type=Path)
    ap.add_argument('--new', default=PAPER / 'main.tex', type=Path)
    ap.add_argument('--out', default=PAPER / 'main-highlight.tex', type=Path)
    a = ap.parse_args(); build(a.old, a.new, a.out)
