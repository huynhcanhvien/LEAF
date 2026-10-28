"""Hardware latency model of a schedule (once per backbone, GPU and batch size).

    T_b(x) = c0 + s * sum_l [t_attn(n_in,l) + t_mlp(n_out,l)] + c_st * K + c_D * n_D

t_attn / t_mlp: one encoder block's attention sublayer (with the CLS row and size bias) and MLP, profiled
on a token grid (BF16, CUDA graph, GPU alone) and interpolated. c0, s, c_st, c_D: least squares on measured
schedules. The alternative 'gmac' model (s * GMAC instead of the profile sum) is kept when its cross-validated
residual is lower. n_D defaults to the number of stations before the last block (as the planner used).
"""
import numpy as np

from leaf.planner.schedule_caps import analytic_gmacs, block_gmacs, canonical, token_counts


def grid(total):
    return sorted({total, 3} | set(range(total - 4, 2, -6)), reverse=True)


def profile(vit, spec, batches=(1, 32, 128)):
    """{grid, ms: {'bs/attn': [...], 'bs/mlp': [...]}} for the middle block of the backbone."""
    import torch, torch.nn as nn
    from leaf.latency.timing import measure_latency
    from leaf.model.token_ops import cls_attention
    blk = vit.blocks[spec.depth // 2 - 1]; g = grid(spec.n_tokens); out = {}

    class Sub(nn.Module):
        def __init__(self, kind, n, bs):
            super().__init__(); self.blk, self.kind = blk, kind
            self.register_buffer('tok', torch.randn(bs, n, spec.dim)); self.register_buffer('size', torch.rand(bs, n, 1) + .5)

        def forward(self, _):
            x = self.tok
            if self.kind == 'attn': return x + self.blk.ls1(cls_attention(self.blk, self.blk.norm1(x), self.size)[0])
            return x + self.blk.mlp(self.blk.norm2(x))

    for bs in batches:
        for kind in ('attn', 'mlp'):
            out[f'{bs}/{kind}'] = [measure_latency(Sub(kind, n, bs).cuda().eval(), bs, graph=True, iters=100 if bs <= 32 else 40,
                                                warmup=10)['ms_mean'] for n in g]
    return dict(grid=g, ms=out)


def n_d(cfg, depth):
    """Stations that compute D in a config."""
    return sum(1 for l, a in cfg.get('alpha', {}).items() if int(l) < depth - 1 and float(a) > 0)


def features(row, kind, prof, bs, spec):
    g = canonical(dict(p=list(row['p']), m=list(row['m'])), spec.n_tokens)
    rich = kind.endswith('_station'); base_kind = kind.removesuffix('_station')
    if base_kind == 'gmac': base = analytic_gmacs(g['p'], g['m'], spec.n_tokens, spec.dim)
    before, after = token_counts(g, spec.n_tokens); gr = prof['grid'][::-1]
    ta = lambda n: np.interp(n, gr, prof['ms'][f'{bs}/attn'][::-1]); tm = lambda n: np.interp(n, gr, prof['ms'][f'{bs}/mlp'][::-1])
    if base_kind == 'profile': base = float(sum(ta(a) + tm(b) for a, b in zip(before, after)))
    out = [1., base, len(g['sites']), row['nd']]
    if rich:
        active = row.get('alpha', {})
        out += [sum(before[l] for l in g['sites']) / spec.n_tokens,
                sum(g['p'][l] - g['m'][l] for l in g['sites']) / spec.n_tokens,
                sum(before[l] * row.get('groups', 16) for l in g['sites']
                    if float(active.get(str(l), active.get(l, 0.))) > 0 and l < spec.depth - 1) / (spec.n_tokens * 16)]
    return out


def fit_model(rows, prof, spec, bs, folds=5, seed=0, station_features=False):
    """Least squares for both models; k-fold CV residuals choose the model. rows: [{p, m, nd, ms, group}].
    With a 'group' per row (measurement session), folds are leave-one-group-out."""
    y = np.array([r['ms'] for r in rows]); res = {}
    groups = sorted({r.get('group', 0) for r in rows})
    if len(groups) > 1: fold_of = [groups.index(r.get('group', 0)) for r in rows]; nf = len(groups)
    else: fold_of = list(np.random.default_rng(seed).permutation(len(rows)) % folds); nf = folds
    kinds = ('gmac', 'profile', 'gmac_station', 'profile_station') if station_features else ('gmac', 'profile')
    for kind in kinds:
        X = np.array([features(r, kind, prof, bs, spec) for r in rows])
        c, *_ = np.linalg.lstsq(X, y, rcond=None); e = y - X @ c; cv = []
        for f in range(nf):
            tr = [i for i in range(len(rows)) if fold_of[i] != f]; te = [i for i in range(len(rows)) if fold_of[i] == f]
            if len(tr) < 8 or not te: continue
            ct, *_ = np.linalg.lstsq(X[tr], y[tr], rcond=None); cv += list(y[te] - X[te] @ ct)
        cv = np.array(cv)
        res[kind] = dict(coef=[float(v) for v in c], r2=float(1 - e.var() / y.var()), sd=float(e.std()), max=float(abs(e).max()),
                         cv_sd=float(cv.std()) if len(cv) else None, cv_max=float(abs(cv).max()) if len(cv) else None, n=len(rows))
    res['chosen'] = min(kinds, key=lambda kind: (res[kind]['cv_sd'] if res[kind]['cv_sd'] is not None else float('inf'),
                                              -kinds.index(kind)))
    return res


class ProfileLatencyModel:
    """Latency model of one batch size: T(p, m) and per-block transition costs for the DP."""

    def __init__(self, model, spec, bs=32):
        f = model['fit'][str(bs)]; self.kind = f['chosen']; self.bs = bs; self.spec = spec
        coef = f[self.kind]['coef']
        self.c0, self.s, self.cst, self.cd = coef[:4]; self.extra = coef[4:]; self.prof = model['profile']
        g = self.prof['grid'][::-1]
        self._ta = lambda n: float(np.interp(n, g, self.prof['ms'][f'{bs}/attn'][::-1]))
        self._tm = lambda n: float(np.interp(n, g, self.prof['ms'][f'{bs}/mlp'][::-1]))

    def block(self, n_in, p, m):
        """One block without the station constant (prune keep p, merge keep m)."""
        if self.kind.startswith('gmac'): return self.s * block_gmacs(n_in, p, m, self.spec.dim)
        return self.s * (self._ta(n_in) + self._tm(min(n_in, p, m)))

    def station(self, n_in, p, m, active_d=False, groups=16):
        value = self.cst + self.cd * bool(active_d)
        if self.extra:
            a, b, c = self.extra
            value += (a * n_in + b * (p - m) + c * n_in * groups / 16 * bool(active_d)) / self.spec.n_tokens
        return value

    def __call__(self, p, m, nd=None, alphas=None):
        g = canonical(dict(p=list(p), m=list(m)), self.spec.n_tokens); n, t = self.spec.n_tokens, self.c0
        for l, (pp, mm) in enumerate(zip(g['p'], g['m'])):
            t += self.block(n, pp, mm)
            if self.extra and l in g['sites']:
                active = l < self.spec.depth - 1 and (alphas is None or float(alphas.get(str(l), alphas.get(l, 0.))) > 0)
                t += self.station(n, pp, mm, active)
            n = min(n, pp, mm)
        if self.extra: return t
        K = len(g['sites'])
        active = sum(1 for l in g['sites'] if l < self.spec.depth - 1 and
                     (alphas is None or float(alphas.get(str(l), alphas.get(l, 0.))) > 0))
        return t + self.cst * K + self.cd * (active if nd is None else nd)


def close_loop(state, budget, measure, total=197, max_measurements=6, window=(.08, .02), fixed_counts=None,
               fallback_window=None):
    """Safeguarded secant on the keep-ratio exponent; return only a measured feasible state.

    measure(state) -> ms. No extrapolated/model-only feasibility is accepted.
    Failed fits return state=None and the full measurement trace.
    """
    low, high = budget - window[0], budget - window[1]; target = (low + high) / 2
    if not 0 <= window[1] <= window[0] or budget <= 0: raise ValueError('invalid window/budget')
    ratios = []; n = total
    for c in state['counts']: ratios.append(c / n); n = c
    def build(eta):
        n = total; counts = []
        for i, r in enumerate(ratios):
            c = (fixed_counts[i] if fixed_counts and i in fixed_counts else
                 max(3, min(n - 1, round(n * r ** eta))))
            if c >= n: return None
            counts.append(c); n = c
        return dict(state, counts=counts)
    eta = 1.; trace = []; seen = set(); bracket = [None, None]
    secant_attempts = max_measurements if fallback_window is None else max(1, max_measurements - 2)
    for _ in range(secant_attempts):
        st = build(eta)
        if st is None: break
        key = tuple(st['counts'])
        if key in seen:
            # Integer rounding can repeat a count vector before the measurement
            # budget is exhausted. Try unmeasured vectors inside the bracket.
            if all(x is not None for x in bracket):
                lo, hi = sorted(x[0] for x in bracket)
            else: lo, hi = eta * .8, eta * 1.2
            alternatives = sorted(np.linspace(lo, hi, 257), key=lambda x: abs(x - eta))
            fresh = next(((float(x), build(float(x))) for x in alternatives
                          if build(float(x)) is not None and tuple(build(float(x))['counts']) not in seen), None)
            if fresh is None: break
            eta, st = fresh; key = tuple(st['counts'])
        seen.add(key); ms = float(measure(st)); trace.append(dict(eta=eta, ms=ms, state=st))
        if not np.isfinite(ms): break
        if low <= ms <= high: return dict(state=st, ms=ms, measurements=trace, feasible=True)
        bracket[0 if ms > target else 1] = (eta, ms)
        if all(x is not None for x in bracket):
            a, b = bracket; next_eta = a[0] + (target - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
            lo, hi = sorted((a[0], b[0])); eta = float(np.clip(next_eta, lo + .1 * (hi - lo), hi - .1 * (hi - lo)))
        elif (len(trace) >= 2 and abs(trace[-1]['ms'] - trace[-2]['ms']) > 1e-6 and
              (trace[-1]['ms'] - trace[-2]['ms']) * (trace[-1]['eta'] - trace[-2]['eta']) < 0):
            a, b = trace[-2:]; guess = b['eta'] + (target - b['ms']) * (b['eta'] - a['eta']) / (b['ms'] - a['ms'])
            eta = float(np.clip(guess, eta * .5, eta * 2.))
        else: eta *= 1.02 if ms > target else .98
    if fallback_window is not None:
        if not 0 <= fallback_window[1] <= fallback_window[0]: raise ValueError('invalid fallback window')
        # A scalar exponent can jump across the window when several integer
        # counts change together. Spend remaining attempts on single-token
        # neighbours, preserving strict ordering and any fixed early counts.
        while trace and len(trace) < max_measurements:
            base = min(trace, key=lambda r: abs(r['ms'] - target))
            direction = 1 if base['ms'] < target else -1
            proposal = None
            for i in range(len(base['state']['counts'])):
                if fixed_counts and i in fixed_counts: continue
                counts = list(base['state']['counts']); counts[i] += direction
                if tuple(counts) in seen: continue
                if counts[-1] < 3 or counts[0] >= total or any(a <= b for a,b in zip(counts, counts[1:])): continue
                proposal = dict(base['state'], counts=counts); break
            if proposal is None: break
            seen.add(tuple(proposal['counts'])); ms = float(measure(proposal))
            trace.append(dict(eta=None, discrete_refine=True, ms=ms, state=proposal))
            if low <= ms <= high:
                return dict(state=proposal, ms=ms, measurements=trace, feasible=True, discrete_refine=True)
        feasible = [r for r in trace if budget - fallback_window[0] <= r['ms'] <= budget - fallback_window[1]]
        if feasible:
            best = min(feasible, key=lambda r: abs(r['ms'] - target))
            return dict(state=best['state'], ms=best['ms'], measurements=trace, feasible=True,
                        fallback=True, accepted_window=list(fallback_window))
    return dict(state=None, ms=None, measurements=trace, feasible=False)


from leaf.evaluation.gpu_coordination import exclusive_timing
from leaf.latency.timing import INFERENCE_DTYPE


@exclusive_timing
def measured_latency(runner, cfg, blocks=3, replays=20):
    """One capture, three timed replay blocks; no other process on the selected GPU."""
    import torch
    import os, subprocess
    from leaf.evaluation.runner import require_idle_gpu
    from leaf.evaluation.determinism import native
    require_idle_gpu()
    with native(), torch.no_grad():
        model = runner.inference_model(cfg)
        x = torch.randn(32, 3, 224, 224, device='cuda')
        side = torch.cuda.Stream(); side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side), torch.autocast('cuda', dtype=INFERENCE_DTYPE):
            for _ in range(10): model(x)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph), torch.autocast('cuda', dtype=INFERENCE_DTYPE): model(x)
        for _ in range(10): graph.replay()
        torch.cuda.synchronize(); samples = []
        for _ in range(blocks):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(replays): graph.replay()
            end.record(); end.synchronize(); samples.append(start.elapsed_time(end) / replays)
        try:
            telemetry = subprocess.check_output(['nvidia-smi', '--id=' + os.environ.get('CUDA_VISIBLE_DEVICES', '0'),
                '--query-gpu=clocks.current.sm,clocks.current.memory,power.draw,temperature.gpu',
                '--format=csv,noheader,nounits'], text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError): telemetry = None
        return dict(ms=float(np.median(samples)), samples=samples, telemetry=telemetry,
                    telemetry_fields=['sm_mhz', 'memory_mhz', 'power_w', 'temperature_c'],
                    peak_gib=torch.cuda.max_memory_allocated() / 2 ** 30)
