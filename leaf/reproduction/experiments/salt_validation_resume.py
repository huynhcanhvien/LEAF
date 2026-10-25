"""Resume the sealed campaign after unrelated GPU jobs release both cards."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('results/salt_validation')


def save(path, data):
    tmp = path.with_suffix('.resume.tmp')
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def idle():
    apps = subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'
    ], text=True).strip()
    return not apps


def completed(folder):
    path = folder / 'status.json'
    return path.exists() and json.loads(path.read_text()).get('phase') == 'completed'


def run(script, args, folder):
    while not completed(folder):
        while not idle():
            save(ROOT / 'resume_status.json', dict(
                phase='waiting_for_idle_gpus', next_script=script,
                pid=os.getpid(), time=time.time()))
            time.sleep(20)
        with (folder / 'scheduler.log').open('a') as log:
            proc = subprocess.Popen([sys.executable, script, *args],
                stdout=log, stderr=subprocess.STDOUT,
                env=dict(os.environ, PYTHONUNBUFFERED='1', CUDA_VISIBLE_DEVICES='0'))
            (folder / 'scheduler.pid').write_text(str(proc.pid))
            save(ROOT / 'resume_status.json', dict(
                phase='running', script=script, child_pid=proc.pid,
                pid=os.getpid(), time=time.time()))
            code = proc.wait()
        if code:
            # Retry only the known transient busy-GPU failure. Preserve all logs.
            logs = [folder / 'scheduler.log', *folder.glob('*gpu*.log'), folder / 'latency.log']
            recent = '\n'.join(p.read_text()[-3000:] for p in logs if p.exists())
            if 'GPU ' not in recent or 'shared with' not in recent:
                raise RuntimeError(f'{script} failed ({code}); inspect saved logs')
            time.sleep(20)


if __name__ == '__main__':
    lock = (ROOT / 'resume.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    run('leaf/reproduction/experiments/salt_validation.py', ['launch'], ROOT)
    run('leaf/reproduction/experiments/salt_validation_extra_timing.py', ['launch'], ROOT / 'extra')
    with (ROOT / 'finalizer.log').open('a') as log:
        subprocess.run([sys.executable, 'leaf/reproduction/experiments/salt_validation_audit.py'],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    save(ROOT / 'resume_status.json', dict(phase='completed', time=time.time()))
