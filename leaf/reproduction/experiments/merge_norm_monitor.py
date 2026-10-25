"""CPU-only durable journal for an already-running norm campaign.

Every completed evaluation already has its own NPZ+JSON. This watcher appends
checkpoints without changing the frozen experiment code or the inference process.
Human-facing progress is emitted every 30 minutes and on a phase/state transition.
"""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

DEFAULT='results/merge_variants/stage2/attempt2'


def save(path,obj):
    tmp=path.with_suffix(f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(obj,indent=1)+'\n');tmp.replace(path)


def snapshot(root):
    status=json.loads((root/'status.json').read_text())
    backbones={}
    for b in ('deit_b','mae_b'):
        base=root/b
        counts={split:len(list((base/split/'runs').glob('*.json'))) for split in ('half','confirm','val','v2')}
        modes={}
        for mode in ('M3','N4'):
            path=base/f'probe_progress_{mode}.json'
            if path.exists():modes[mode]=json.loads(path.read_text())
        backbones[b]=dict(completed=counts,probes=modes,
            schedules=len(list((base/'search').glob('*/*.json'))),
            frozen=(base/'stage2/frozen.json').exists(),pairs=(base/'stage2/pairs.json').exists(),
            latency_rows=len(list((base/'stage2/lat').glob('*.json'))))
    return dict(time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status=status,backbones=backbones,
                report='results/reports/merge-variants-stage2-2026-10-16.md')


def run(root):
    lock=(root/'watch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    save(root/'watch_pid.json',dict(pid=os.getpid()))
    last_state=None;last_emit=time.monotonic()
    while True:
        data=snapshot(root)
        # Journal is append-only; latest checkpoint is replaced atomically.
        with (root/'checkpoints.jsonl').open('a') as handle:
            handle.write(json.dumps(data)+'\n');handle.flush();os.fsync(handle.fileno())
        save(root/'checkpoint.json',data)
        state=data['status']['state']
        if state!=last_state or time.monotonic()-last_emit>=1800:
            print(json.dumps(data),flush=True);last_emit=time.monotonic()
        last_state=state
        if state in ('complete','failed','paused'):
            save(root/'watch_terminal.json',data);return
        time.sleep(30)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['launch','run','snapshot'])
    parser.add_argument('--root',default=DEFAULT);args=parser.parse_args();root=Path(args.root)
    if args.action=='snapshot':print(json.dumps(snapshot(root),indent=1));return
    if args.action=='run':run(root);return
    log=(root/'watch.log').open('a')
    child=subprocess.Popen([sys.executable,'-u',__file__,'run','--root',args.root],start_new_session=True,stdout=log,stderr=subprocess.STDOUT)
    log.close();print(f'checkpoint watcher pid={child.pid}')


if __name__=='__main__':main()
