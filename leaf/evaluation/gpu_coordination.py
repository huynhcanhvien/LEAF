"""Cooperate across campaign GPUs: concurrent evaluation, exclusive timing."""
from contextlib import contextmanager
from functools import wraps
import os
from pathlib import Path


@contextmanager
def gpu_work(exclusive=False):
    path = os.environ.get('A6_GPU_COORDINATION_LOCK')
    if not path:
        yield
        return
    import fcntl
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def exclusive_timing(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        with gpu_work(exclusive=True):
            return fn(*args, **kwargs)
    return wrapped


def shared_evaluation(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        with gpu_work():
            return fn(*args, **kwargs)
    return wrapped
