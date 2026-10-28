"""Bit-exact reproducibility of every model kind (method with/without D, every baseline, I-role variants):
same process twice, a second process, and every visible GPU; and the method's bits are the same with or without
deterministic mode (so runs cached before the switch stay valid). Needs a GPU.
    python tests/test_determinism.py            (checks GPU 0 and 1 when both exist)"""
from common import check, finish
import hashlib, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
STATE = dict(sites=[1, 4, 7, 11], counts=[120, 80, 40, 5], thetas=[0., .5, 1., 0.], alphas=[3 / 4, 1 / 2, 2 / 3, 0.])


def worker(mode):
    """Hashes of logits on 256 calib images for every kind, each evaluated twice in this process."""
    import torch
    if mode == 'native':
        import leaf.evaluation.determinism as D; torch.use_deterministic_algorithms(False)
    from leaf.evaluation.data import iterate
    from leaf.evaluation.runner import Runner
    from leaf.backbones import SPECS
    from leaf.planner.configs import ForecastConfig, config_from_state
    r = Runner('deit_s', root='/tmp/pd_determinism_test', limit=256); r.spec = SPECS['deit_s']
    d = ForecastConfig('schedules/leaf/planning/deit_s/groups.json'); base = config_from_state(STATE, r.spec, d)
    cfgs = {'pd_D': base, 'pd_P': config_from_state(dict(STATE, alphas=[0.] * 4), r.spec, d), 'tome': dict(kind='tome', r=13),
            'pitome': dict(kind='pitome', ratio=.9), 'evit': dict(kind='evit', keep_rate=.7), 'mctf': dict(kind='mctf', r=16),
            'ppt': dict(kind='ppt', r=50, threshold=7e-5)}
    images, labels, _ = r.data('half'); out = {}
    for name, c in cfgs.items():
        hs = []
        for _ in range(2):
            m = r.inference_model(c); ls = []
            with torch.no_grad():
                for x, _y in iterate(images, labels, 128, 'cuda', r.spec.mean, r.spec.std):
                    with torch.autocast('cuda', dtype=torch.bfloat16): ls.append(m(x).float().cpu())
            hs.append(hashlib.sha256(torch.cat(ls).numpy().tobytes()).hexdigest())
        out[name] = hs
    print('RESULT ' + json.dumps(out))


def run(gpu, mode):
    p = subprocess.run([sys.executable, __file__, 'worker', mode], capture_output=True, text=True,
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)), cwd=os.path.dirname(HERE))
    line = [l for l in p.stdout.splitlines() if l.startswith('RESULT ')]
    if not line: raise RuntimeError(p.stderr[-2000:])
    return json.loads(line[0][7:])


def main():
    import torch
    gpus = list(range(min(2, torch.cuda.device_count())))
    a, b = run(gpus[0], 'det'), run(gpus[0], 'det')
    for k in a:
        check(f'{k}: identical twice in one process', a[k][0] == a[k][1])
        check(f'{k}: identical in a second process', a[k][0] == b[k][0])
    if len(gpus) > 1:
        c = run(gpus[1], 'det')
        for k in a: check(f'{k}: identical on GPU {gpus[1]}', a[k][0] == c[k][0])
    n = run(gpus[0], 'native')
    for k in ('pd_D', 'pd_P', 'evit'):
        check(f'{k}: same bits with and without deterministic mode', a[k][0] == n[k][0])


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'worker':
        sys.path.insert(0, os.path.dirname(HERE)); worker(sys.argv[2])
    else:
        main(); finish()
