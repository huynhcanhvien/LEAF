"""Accuracy scheduling amendment: one backbone worker per GPU, timing still last."""
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from leaf.reproduction.experiments import salt_m3_pilot as pilot


def main():
    root=pilot.ROOT
    lock=(root/'scheduler.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    pilot.validate()
    pilot.immutable(root/'accuracy_parallel_amendment.json',dict(
        source_sha=pilot.hash_file(__file__),user_request='Use both GPUs for accuracy',
        assignment={'0':['deit_s'],'1':['augreg_s']},
        mae_b='already complete; exact-config historical accuracy reused',
        accuracy_batch=32,latency='unchanged: bs32, single GPU, after all accuracy'))
    assert (root/'mae_b/accuracy_done.json').exists()
    pilot.save(root/'status.json',dict(phase='accuracy_two_gpus',pid=os.getpid(),time=time.time()))
    workers=[]
    for gpu,bb in [(0,'deit_s'),(1,'augreg_s')]:
        log=(root/f'accuracy_parallel_gpu{gpu}.log').open('a')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONUNBUFFERED='1')
        workers.append(subprocess.Popen([sys.executable,pilot.__file__,'worker','--backbones',bb],
            env=env,stdout=log,stderr=subprocess.STDOUT))
    while any(p.poll() is None for p in workers):
        time.sleep(10);pilot.report()
    assert all(p.returncode==0 for p in workers), 'accuracy worker failed'
    pilot.save(root/'status.json',dict(phase='latency',pid=os.getpid(),time=time.time()))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHONUNBUFFERED='1')
    with (root/'latency.log').open('a') as log:
        proc=subprocess.Popen([sys.executable,pilot.__file__,'timing'],env=env,stdout=log,stderr=subprocess.STDOUT)
        while proc.poll() is None:
            time.sleep(10);pilot.report()
        assert proc.returncode==0, 'latency worker failed'
    pilot.report();pilot.save(root/'status.json',dict(phase='completed',pid=os.getpid(),time=time.time()))


if __name__=='__main__':main()
