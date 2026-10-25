"""Persist final analysis automatically after the detached GPU scheduler completes."""
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from leaf.evaluation.runner import load_json,save
from leaf.reproduction.experiments import salt_fit as cpu

if __name__=='__main__':
    root=cpu.ROOT/'gpu'
    while True:
        status=load_json(root/'status.json')
        if status['phase']=='completed':break
        pid=int((root/'scheduler.pid').read_text())
        if not Path(f'/proc/{pid}').exists():
            save(cpu.ROOT/'finalizer_status.json',dict(phase='scheduler_stopped_before_completion',time=time.time()))
            raise SystemExit('GPU scheduler stopped before completion; saved results remain available.')
        time.sleep(10)
    subprocess.run([sys.executable,'leaf/reproduction/experiments/salt_formula_summary.py'],check=True)
    save(cpu.ROOT/'finalizer_status.json',dict(phase='completed',time=time.time(),report=str(cpu.REPORT)))
