"""Progress lines of the command line: `HH:MM:SS leaf <command> | message` on stderr (stdout keeps the results)."""
import sys
import time


def log(command, message):
    print(f'{time.strftime("%H:%M:%S")} leaf {command} | {message}', file=sys.stderr, flush=True)


def duration(seconds):
    seconds = int(round(seconds))
    if seconds < 60: return f'{seconds}s'
    if seconds < 3600: return f'{seconds // 60}m{seconds % 60:02d}s'
    return f'{seconds // 3600}h{seconds % 3600 // 60:02d}m'


class Progress:
    """Counts finished items and logs every `every` items (and the last) with the elapsed time and an ETA."""

    def __init__(self, command, total, what, every=None):
        self.command, self.total, self.what = command, total, what
        self.every = every or max(1, total // 20); self.done = 0; self.t0 = time.monotonic()

    def step(self, n=1):
        self.done += n
        if self.done % self.every and self.done != self.total: return
        el = time.monotonic() - self.t0
        eta = el / self.done * (self.total - self.done) if self.done else 0
        log(self.command, f'{self.done}/{self.total} {self.what}  ({duration(el)} elapsed, ~{duration(eta)} left)')
