"""Tiny check harness: python tests/<file>.py prints PASS/FAIL lines and exits 1 on any failure."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FAILS = []


def check(name, ok, detail=''):
    print(('PASS ' if ok else 'FAIL ') + name + (f'  {detail}' if detail else ''), flush=True)
    if not ok: FAILS.append(name)


def finish():
    print('ALL PASS' if not FAILS else f'FAILED: {FAILS}')
    sys.exit(1 if FAILS else 0)
