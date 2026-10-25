"""Shared helpers of the Q1 campaigns: exclusive measurement windows and crash-safe task claims.

Timing rule (user, 2026-10-18): while a latency is being measured, only that device works. Every
measurement call runs inside `measure_exclusive()`:
  * one machine-wide lock, so GPU0 timing and CPU timing never overlap;
  * all other GPU processes of these campaigns (any card) are paused with SIGSTOP and resumed with SIGCONT
    afterwards, so the second GPU keeps doing accuracy work between measurements but never during one;
  * a GPU process that does not belong to these campaigns is never signalled: the measurement waits.
"""
import contextlib
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import time

LOCK = Path(__file__).resolve().parents[3] / 'results/logs/timing.lock'
MARKERS = ('main_comparison.py', 'latency_pricing.py', 'clip_comparison.py', 'ablation_and_alpha_grid.py',
           'alpha_kappa_equivalence.py', 'horizon_norm_seed_ablation.py', 'second_device.py', 'station_discount.py',
           'diffrate_clip_search.py', 'separability_check.py', 'final_station_alpha7.py')   # processes of these campaigns


def _cmdline(pid):
    try:
        return Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
    except OSError:
        return ''


def gpu_pids():
    out = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
    return sorted({int(s) for s in out.split() if s.strip()})


def _ours(pid):
    return any(m in _cmdline(pid) for m in MARKERS)


def _gpu_busy(min_util=5):
    out = subprocess.check_output(['nvidia-smi', '--query-gpu=index,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    return {int(i): int(u) for i, u in (r.split(', ') for r in out.strip().splitlines())}


@contextlib.contextmanager
def measure_exclusive(settle=0.5, poll=10, drain=60):
    """Exclusive measurement window; yields the list of paused PIDs (recorded with each measurement)."""
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with open(LOCK, 'w') as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        me = os.getpid()
        while True:
            others = [p for p in gpu_pids() if p != me]
            foreign = [p for p in others if not _ours(p)]
            if not foreign: break
            time.sleep(poll)                     # never signal a process of another project
        paused = []
        if not others:                           # nothing else on any GPU: no signals, no drain polling
            try:
                yield paused
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)
            return
        try:
            for p in others:
                try:
                    os.kill(p, signal.SIGSTOP); paused.append(p)
                except ProcessLookupError:
                    pass
            time.sleep(settle)                   # in-flight kernels of paused processes drain
            # SIGSTOP halts host threads only; already-queued kernels still run. Wait until every other
            # card reads 0% utilisation twice in a row (the counter averages over ~1 s), at most `drain` s.
            own = os.environ.get('CUDA_VISIBLE_DEVICES', '')
            mine = {int(own.split(',')[0])} if own.strip() else set()
            quiet, start = 0, time.time()
            while quiet < 2 and time.time() - start < drain:
                util = _gpu_busy()
                quiet = quiet + 1 if all(u == 0 for i, u in util.items() if i not in mine) else 0
                if quiet < 2: time.sleep(.5)
            yield paused
        finally:
            for p in paused:
                try:
                    os.kill(p, signal.SIGCONT)
                except ProcessLookupError:
                    pass
            fcntl.flock(fh, fcntl.LOCK_UN)


def claim(root, split, tid):
    """Claim a task; a claim whose owner died without completing is taken over (crash/kill safe)."""
    p = Path(root) / 'claims' / split / f'{tid}.lock'; p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f'{p.name}.{os.getpid()}')        # the lock appears with its pid already written: workers started
    tmp.write_text(str(os.getpid()))                     # together never read an empty lock and take it over
    try:
        os.link(tmp, p); return True
    except FileExistsError:
        pass
    finally:
        tmp.unlink(missing_ok=True)
    try:
        owner = int(p.read_text() or 0)
    except (OSError, ValueError):
        owner = 0
    if owner and Path(f'/proc/{owner}').exists() and _ours(owner):
        return False
    if (Path(root) / 'completed' / split / f'{tid}.json').exists():
        return False
    p.write_text(str(os.getpid()))           # stale claim (dead owner or pre-pid claim file)
    return True


def cast_exclusive(model, batch_size, **kw):
    """leaf.latency.timing.measure_latency inside an exclusive window; records the paused PIDs."""
    from leaf.latency.timing import measure_latency
    if os.environ.get('LEAF_EXCLUSIVE_TIMING') != '1':          # 2026-10-18: user deferred strict re-timing; tag the session
        return dict(measure_latency(model, batch_size, **kw), paused_pids=[], exclusive=False)
    with measure_exclusive() as paused:
        out = measure_latency(model, batch_size, **kw)
    return dict(out, paused_pids=paused, exclusive=True)


def cpu_exclusive(model, batch_size, **kw):
    """leaf.latency.cpu_timing.measure_latency_cpu inside an exclusive window (all campaign GPU work paused)."""
    from leaf.latency.cpu_timing import measure_latency_cpu
    if os.environ.get('LEAF_EXCLUSIVE_TIMING') != '1':
        return dict(measure_latency_cpu(model, batch_size, **kw), paused_pids=[], exclusive=False)
    with measure_exclusive() as paused:
        out = measure_latency_cpu(model, batch_size, **kw)
    return dict(out, paused_pids=paused, exclusive=True)


def guard_foreign():
    """Refuse a card shared with a process that is not part of these campaigns (timing rule: never use a card busy
    with another project). Campaign processes may share a card for accuracy work; timing uses measure_exclusive."""
    index = os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0]
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,pci.bus_id', '--format=csv,noheader'], text=True)
    bus = dict(r.split(', ') for r in rows.strip().splitlines())[index]
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_bus_id', '--format=csv,noheader'], text=True)
    foreign = [r for r in apps.strip().splitlines() if r and r.split(', ')[1] == bus
               and int(r.split(', ')[0]) != os.getpid() and not _ours(int(r.split(', ')[0]))]
    if foreign: raise RuntimeError(f'GPU {index} shared with a foreign process: {foreign}')
