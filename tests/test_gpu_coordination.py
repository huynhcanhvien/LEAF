"""CPU-only check: two evaluations overlap and timing waits for both."""
import multiprocessing as mp
import os
from pathlib import Path
import queue
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from leaf.evaluation.gpu_coordination import gpu_work


def reader(events, release):
    with gpu_work():
        events.put('evaluation')
        assert release.wait(10)


def writer(events):
    with gpu_work(exclusive=True):
        events.put('timing')


if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as directory:
        os.environ['A6_GPU_COORDINATION_LOCK'] = str(Path(directory) / 'gpu.lock')
        events, release = mp.Queue(), mp.Event()
        readers = [mp.Process(target=reader, args=(events, release)) for _ in range(2)]
        for process in readers: process.start()
        try:
            assert [events.get(timeout=5) for _ in readers] == ['evaluation', 'evaluation']
            timing = mp.Process(target=writer, args=(events,)); timing.start()
            try:
                events.get(timeout=.3)
                raise AssertionError('timing overlapped active evaluation')
            except queue.Empty:
                pass
            release.set()
            assert events.get(timeout=5) == 'timing'
            timing.join(5); assert timing.exitcode == 0
        finally:
            release.set()
            for process in readers:
                process.join(5)
                assert process.exitcode == 0
    print('ALL PASS: concurrent evaluation and exclusive timing')
