"""The schedule books in assets/schedules (CPU, no results needed):
- every entry has a config, the paper's top-1 and the tables or figures that use it;
- `leaf/` entries equal the released schedules;
- `main/` entries reproduce the published latency-paired report rows (DiffRate values of the README);
- every key loads through leaf.api.load_schedule, as `leaf evaluate|benchmark --schedule` does.

    python tests/test_schedule_books.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import check, finish

from leaf.api import load_schedule
from leaf.backbones import SPECS
from leaf.reproduction.experiments.campaign_reports import BBS as PAPER_BBS


def main():
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for bb in PAPER_BBS:          # the paper's backbones (registered extra ones, e.g. clip_b16, have no book)
        path = f'assets/schedules/{bb}.json'
        doc, _ = load_schedule(path)
        entries = doc['schedules']
        complete = all('config' in e and 'kind' in e['config'] and isinstance(e['measured'].get('val'), float) and e['paper']
                       for e in entries.values())
        check(f'{bb}: {len(entries)} book entries with config, paper top-1 and paper references', complete and len(entries) > 200)
        released = json.load(open(f'schedules/leaf/{bb}.json'))['schedules']
        same = all(entries[f'leaf/{k}']['config'] == v['config'] and entries[f'leaf/{k}']['measured'] == v['measured']
                   for k, v in released.items())
        check(f'{bb}: book leaf/ entries equal schedules/leaf', same, f'{len(released)} entries')
        keys = list(entries)[:: max(1, len(entries) // 25)]
        check(f'{bb}: book keys load through leaf.api.load_schedule', all(load_schedule(path, k)[1] is entries[k] or
                                                                          load_schedule(path, k)[1] == entries[k] for k in keys))
    b = json.load(open('assets/schedules/deit_b.json'))['schedules']
    readme = {'DR8.7': (79.11, 67.44, 18.81), 'DR10.0': (80.66, 69.40, 21.49), 'DR11.5': (81.48, 70.76, 24.26), 'DR12.5': (81.71, 71.01, 26.91)}
    got = {k: tuple(round(b[f'main/diffrate/{k}']['measured'][f], 2) for f in ('val', 'v2', 'ms_bs32')) for k in readme}
    check('deit_b: book DiffRate entries give the README / paper numbers', got == readme, str(got))
    finish()


if __name__ == '__main__':
    main()
