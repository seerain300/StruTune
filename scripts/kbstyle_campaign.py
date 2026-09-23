#!/usr/bin/env python3
"""Official-style multi-turn campaign for drkernel-8b over all 30 tasks.

Protocol (aligned with KernelGYM drkernel-14b-maxturns3.sh):
  - task order: flashinfer first (fewest workloads first), then SOL L1, L2
  - per task: 8 samples (n=8 batched on turn 1, parallel requests after),
    up to 3 user turns; evaluator feedback becomes the next user message
  - turn-1 eval screens workload #0 only; later turns evaluate ALL workloads
    with cached reference outputs/latencies (cache built once up front);
    final reporting re-runs the official evaluators fresh (100 iters)
  - reward = 0.3*compile + 0.4*correct + 0.3*min(geomean_speedup, 3.0)
  - evaluation dispatched to a monitored GPU pool (default GPUs 0-6),
    incremental per-task results written to results.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

sys_path = str(Path(__file__).resolve().parent)
import sys
sys.path.insert(0, sys_path)
from drkernel_kbstyle_pass1 import (  # noqa: E402
    build_prompt, extract_modelnew, solution_source, sol_document,
)

WS = Path('/home/ziming/whr/run')
SOL = Path('/home/ziming/dataset/SOL-ExecBench')
FIB = Path('/home/ziming/dataset/flashinfer-test')
PLAN = WS / 'task_plan.json'
RUN_ROOT = WS / 'baseline/drtriton/kbstyle_campaign_v2_20260922'
FIB_EVAL = WS / 'evaluators/evaluate.py'
SOL_EVAL = WS / 'evaluators/evaluate_sol.py'
FIB_FB = WS / 'scripts/kbstyle_fib_eval.py'
MTMC_PY = Path('/home/ziming/miniconda3/envs/fib/bin/python')
SOL_PY = SOL / '.venv/bin/python'
SOL_REF_KEY = 'h100-hbm3-80gb-v1'

os.environ['NO_PROXY'] = ','.join(filter(None, [os.environ.get('NO_PROXY', ''), '127.0.0.1', 'localhost']))
os.environ['no_proxy'] = ','.join(filter(None, [os.environ.get('no_proxy', ''), '127.0.0.1', 'localhost']))

N_SAMPLES = 8
MAX_TURNS = 3
W_COMP, W_CORR, W_PERF = 0.3, 0.4, 0.3
SPEEDUP_CAP = 3.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--gpus', default='0,1,2,3,4,5')
    p.add_argument('--max-parallel', type=int, default=4)
    p.add_argument('--task', action='append', default=[])
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--top-p', type=float, default=0.95)
    p.add_argument('--max-tokens', type=int, default=8192)
    p.add_argument('--seed', type=int, default=20260918)
    p.add_argument('--screen-timeout', type=int, default=420)
    p.add_argument('--full-timeout', type=int, default=1200)
    p.add_argument('--precompute-timeout', type=int, default=2700)
    p.add_argument('--final-timeout', type=int, default=2700)
    p.add_argument('--poll-seconds', type=int, default=10)
    p.add_argument('--phase', choices=['scan', 'continue', 'both'], default='both',
                   help='scan: turn-0 only for all tasks (fast rough signal); '
                        'continue: feedback turns + final eval; both: scan then continue')
    p.add_argument('--concurrent-tasks', type=int, default=2,
                   help='tasks processed concurrently in the continue phase')
    p.add_argument('--model', default='drkernel-stop-8b')
    p.add_argument('--server', default='http://127.0.0.1:8001/v1')
    return p.parse_args()


def log(msg):
    print(f'[{time.strftime("%m-%d %H:%M:%S")}] {msg}', flush=True)


# ---------------------------------------------------------------- GPU pool
class GPUPool:
    def __init__(self, gpus, max_parallel, poll):
        self.gpus = list(gpus)
        self.max_parallel = max_parallel
        self.poll = poll
        self.sem = threading.Semaphore(max_parallel)
        self.lock = threading.Lock()
        self.ours: set[int] = set()
        self._monitor()

    def _probe(self, gpu):
        r = subprocess.run(['nvidia-smi', '-i', str(gpu), '--query-compute-apps=pid',
                            '--format=csv,noheader'], capture_output=True, text=True, check=False)
        return bool(r.stdout.strip())

    def _monitor(self):
        free = [g for g in self.gpus if not self._probe(g)]
        log(f'GPU pool {self.gpus} | free now: {free} | our max parallel: {self.max_parallel}')

    def acquire(self):
        self.sem.acquire()
        waited_since = None
        while True:
            with self.lock:
                for g in self.gpus:
                    if g not in self.ours and not self._probe(g):
                        self.ours.add(g)
                        if waited_since is not None:
                            log(f'GPU {g} acquired after {time.time() - waited_since:.0f}s wait')
                        return g
            if waited_since is None:
                waited_since = time.time()
            elif time.time() - waited_since > 120:
                free = [g for g in self.gpus if not self._probe(g)]
                log(f'GPU pool: all busy, waiting (externally free: {free}, ours: {sorted(self.ours)})')
                waited_since = time.time()
            time.sleep(self.poll)

    def release(self, gpu):
        with self.lock:
            self.ours.discard(gpu)
        self.sem.release()


def _gpu_pids(gpu):
    r = subprocess.run(['nvidia-smi', '-i', str(gpu), '--query-compute-apps=pid',
                        '--format=csv,noheader'], capture_output=True, text=True, check=False)
    return {int(t) for t in r.stdout.split() if t.isdigit()}


def _pgid(pid):
    try:
        return os.getpgid(pid)
    except OSError:
        return None


def run_on_gpu(pool: GPUPool, cmd, log_path: Path, timeout: int, poll_s: int = 1):
    """Run cmd on a pooled GPU. While the evaluator runs, the GPU is polled for
    foreign (non-ours) compute processes; if one lands mid-run the attempt is
    killed, we wait for the GPU to become empty again (pool.acquire blocks),
    and the evaluation is re-run from scratch — contended timing never reaches
    the model as feedback. Ours = the evaluator's process group."""
    while True:
        gpu = pool.acquire()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu)}
        started = time.time()
        contended = False
        try:
            with log_path.open('w') as stream:
                proc = subprocess.Popen(cmd, stdout=stream, stderr=subprocess.STDOUT,
                                        env=env, start_new_session=True)
                deadline = started + timeout
                while True:
                    try:
                        rc = proc.wait(timeout=poll_s)
                        timed_out = False
                        break
                    except subprocess.TimeoutExpired:
                        if time.time() >= deadline:
                            os.killpg(proc.pid, signal.SIGKILL)
                            proc.wait()
                            rc, timed_out = 124, True
                            break
                        foreign = {q for q in _gpu_pids(gpu) if _pgid(q) != proc.pid}
                        if foreign:
                            contended = True
                            log(f'  GPU {gpu}: foreign pids {sorted(foreign)[:4]} landed '
                                f'mid-eval; aborting, will wait for the GPU to empty '
                                f'and re-run this evaluation')
                            os.killpg(proc.pid, signal.SIGKILL)
                            proc.wait()
                            rc, timed_out = 125, False
                            break
        finally:
            pool.release(gpu)
        if not contended:
            break
        # contended: loop back; pool.acquire() now blocks until the card is empty
    return {'gpu': gpu, 'exit_code': rc, 'timed_out': timed_out,
            'elapsed_seconds': round(time.time() - started, 2)}


# ---------------------------------------------------------------- tasks
def load_tasks(wanted):
    rows = json.loads(PLAN.read_text())
    if wanted:
        rows = [r for r in rows if r['key'] in set(wanted)]
    fib = sorted([r for r in rows if r['benchmark'] == 'flashinfer'],
                 key=lambda r: r['workloads'])
    sol_l1 = [r for r in rows if r.get('level') == 'L1']
    sol_l2 = [r for r in rows if r.get('level') == 'L2']
    return fib + sol_l1 + sol_l2


# ---------------------------------------------------------------- generation
def compact(text, chars=16000):
    if len(text) <= chars:
        return text
    half = chars // 2
    return text[:half] + '\n\n# ... (middle omitted) ...\n\n' + text[-half:]


def chat(client, args, messages, seed, n=1):
    resp = client.chat.completions.create(
        model=args.model, messages=messages, n=n,
        temperature=args.temperature, top_p=args.top_p,
        max_tokens=args.max_tokens, stop=['<|im_end|>', '<|endoftext|>'], seed=seed,
        extra_body={'stop_token_ids': [151643, 151645]})  # vllm>=0.2x ignores eos override
    out = []
    for choice in resp.choices:
        out.append({'text': choice.message.content or '',
                    'finish_reason': choice.finish_reason})
    usage = {'prompt_tokens': getattr(resp.usage, 'prompt_tokens', None),
             'completion_tokens': getattr(resp.usage, 'completion_tokens', None)}
    return out, usage


def gen_turn(client, args, task, histories, active, task_index, turn):
    """Generate one response per active sample. Returns {sample_idx: {'text',...}}."""
    if turn == 0:
        prompt, _ = build_prompt(task)
        samples, usage = chat(client, args, [{'role': 'user', 'content': prompt}],
                              seed=args.seed + task_index * 100, n=N_SAMPLES)
        return {s: samples[s] for s in active}, usage
    out = {}
    with ThreadPoolExecutor(max_workers=N_SAMPLES) as ex:
        futs = {s: ex.submit(chat, client, args, histories[s],
                             args.seed + task_index * 100 + turn * 10 + s, 1)
                for s in active}
        for s, f in futs.items():
            try:
                res, _ = f.result()
                out[s] = res[0]
            except Exception as e:
                out[s] = {'text': '', 'finish_reason': f'error:{e}'}
    return out, {}


# ---------------------------------------------------------------- evaluation
def eval_candidate(pool, args, task, source, out_dir, mode):
    """mode: 'screen' (workload 0) or 'full' (all workloads, cached refs)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if task['benchmark'] == 'flashinfer':
        cache = RUN_ROOT / 'refcache' / 'fib' / task['name']
        fib_mode = {'screen': 'screen', 'full': 'feedback'}[mode]
        cmd = [str(MTMC_PY), str(FIB_FB), '--mode', fib_mode, '--definition', task['definition'],
               '--workload', task['workload'], '--solution', str(out_dir / 'solution.py'),
               '--cache-dir', str(cache), '--dataset-root', str(FIB),
               '--warmup', '3', '--iters', '10', '--json', str(out_dir / 'evaluation.json')]
        timeout = args.screen_timeout if mode == 'screen' else args.full_timeout
    else:
        cache = RUN_ROOT / 'refcache' / 'sol' / task['name'] / 'reference_cache.json'
        (out_dir / 'solution.json').write_text(json.dumps(sol_document(task, source), indent=2))
        cmd = [str(SOL_PY), str(SOL_EVAL), '--definition', task['definition'],
               '--workload', task['workload'], '--solution', str(out_dir / 'solution.json'),
               '--output', str(out_dir / 'evaluation.json'), '--timeout', '1200', '--rerun',
               '--warmup', '3', '--iterations', '10',
               '--reference-cache', str(cache), '--reference-cache-key', SOL_REF_KEY]
        if mode == 'screen':
            cmd += ['--max-workloads', '1']
        timeout = args.screen_timeout if mode == 'screen' else args.full_timeout
    status = run_on_gpu(pool, cmd, out_dir / 'evaluation.log', timeout)
    result = {}
    try:
        result = json.loads((out_dir / 'evaluation.json').read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    status['result'] = result
    return status


def precompute_refs(pool, args, task, task_dir):
    if task['benchmark'] == 'flashinfer':
        cache = RUN_ROOT / 'refcache' / 'fib' / task['name']
        if (cache / 'manifest.json').exists():
            log(f'  refcache exists: {cache}')
            return
        cmd = [str(MTMC_PY), str(FIB_FB), '--mode', 'precompute', '--definition', task['definition'],
               '--workload', task['workload'], '--cache-dir', str(cache),
               '--dataset-root', str(FIB), '--warmup', '3', '--iters', '10']
    else:
        cache = RUN_ROOT / 'refcache' / 'sol' / task['name'] / 'reference_cache.json'
        if cache.exists():
            log(f'  refcache exists: {cache}')
            return
        ref_dir = task_dir / 'refprecompute'
        ref_dir.mkdir(parents=True, exist_ok=True)
        (ref_dir / 'solution.json').write_text(
            json.dumps({**sol_document(task, task['pytorch']),
                        'description': 'reference itself (ref-latency precompute)'}, indent=2))
        cache.parent.mkdir(parents=True, exist_ok=True)
        cmd = [str(SOL_PY), str(SOL_EVAL), '--definition', task['definition'],
               '--workload', task['workload'], '--solution', str(ref_dir / 'solution.json'),
               '--output', str(ref_dir / 'evaluation.json'), '--timeout', '2400', '--rerun',
               '--warmup', '3', '--iterations', '10',
               '--reference-cache', str(cache), '--reference-cache-key', SOL_REF_KEY]
    status = run_on_gpu(pool, cmd, task_dir / 'precompute.log', args.precompute_timeout)
    log(f'  precompute exit={status["exit_code"]} elapsed={status["elapsed_seconds"]}s')


def final_official(pool, args, task, source, final_dir):
    final_dir.mkdir(parents=True, exist_ok=True)
    if task['benchmark'] == 'flashinfer':
        (final_dir / 'solution.py').write_text(source)
        common = [str(MTMC_PY), str(FIB_EVAL), '--definition', task['definition'],
                  '--workload', task['workload'], '--solution', str(final_dir / 'solution.py'),
                  '--entry', 'run', '--dataset-root', str(FIB), '--device', 'cuda:0',
                  '--trials', '1']
        corr = run_on_gpu(pool, common + ['--correctness-only', '--json',
                                          str(final_dir / 'correctness.json')],
                          final_dir / 'correctness.log', args.final_timeout)
        correctness = json.loads((final_dir / 'correctness.json').read_text()) \
            if (final_dir / 'correctness.json').exists() else {}
        corr['result'] = correctness
        perf = None
        if correctness.get('valid') is True:
            perf_status = run_on_gpu(pool, common + ['--warmup', '3', '--iters', '100', '--json',
                                                     str(final_dir / 'performance.json')],
                                     final_dir / 'performance.log', args.final_timeout)
            perf = json.loads((final_dir / 'performance.json').read_text()) \
                if (final_dir / 'performance.json').exists() else {}
            perf_status['result'] = perf
        return corr, (perf_status if correctness.get('valid') is True else None)
    # SOL
    (final_dir / 'solution.json').write_text(json.dumps(sol_document(task, source), indent=2))
    common = [str(SOL_PY), str(SOL_EVAL), '--definition', task['definition'],
              '--workload', task['workload'], '--solution', str(final_dir / 'solution.json'),
              '--output', str(final_dir / 'evaluation.json'), '--timeout', '2700', '--rerun']
    corr_status = run_on_gpu(pool, common + ['--correctness-only', '--warmup', '0'],
                             final_dir / 'correctness.log', args.final_timeout)
    correctness = json.loads((final_dir / 'evaluation.json').read_text()) \
        if (final_dir / 'evaluation.json').exists() else {}
    corr_status['result'] = correctness
    perf_status = None
    if correctness.get('valid') is True:
        perf_status = run_on_gpu(pool, common + ['--iterations', '100', '--warmup', '3'],
                                 final_dir / 'performance.log', args.final_timeout)
        perf_status['result'] = json.loads((final_dir / 'evaluation.json').read_text()) \
            if (final_dir / 'evaluation.json').exists() else {}
    return corr_status, perf_status


# ---------------------------------------------------------------- rewards
def reward_of(status):
    result = status.get('result') or {}
    per = result.get('per_workload') or []
    total = result.get('total') or len(per) or 1
    passed = result.get('passed') or sum(1 for w in per if w.get('status') == 'PASSED')
    compiled = 0.0 if (not per and status.get('exit_code', 0) != 0) else (
        0.0 if any(w.get('status') in ('RUNTIME_ERROR', 'CACHE_MISS') and w.get('error_log', '').startswith(('CompilationError', 'NameError', 'SyntaxError', 'ImportError')) for w in per)
        else (1.0 if per else 0.0))
    correct = passed / total if total else 0.0
    geo = result.get('geomean_speedup')
    perf = (min(geo, SPEEDUP_CAP) / SPEEDUP_CAP) if (correct >= 1.0 and geo) else 0.0
    return W_COMP * compiled + W_CORR * correct + W_PERF * perf, {
        'compiled': compiled, 'correct_ratio': correct, 'geomean': geo,
        'passed': passed, 'total': total}


# ------------------------------------------------- triton-only compliance
# Method component: static decoy-kernel & torch-fallback detection.
#   decoy   = @triton.jit kernel defined but never referenced outside its own def
#   fallback= host-side torch compute (only allocation/prep calls are allowed)
ALLOWED_TORCH = {'torch.empty', 'torch.empty_like', 'torch.zeros', 'torch.zeros_like',
                 'torch.empty_strided', 'torch.full', 'torch.cuda', 'torch.float32',
                 'torch.float16', 'torch.bfloat16', 'torch.int32', 'torch.int64', 'torch.bool',
                 'torch.arange', 'torch.no_grad', 'torch.enable_grad', 'torch.Tensor',
                 'torch.device', 'torch.is_tensor', 'torch.is_floating_point',
                 'torch.finfo', 'torch.iinfo', 'torch.cuda.is_available'}
ALLOWED_METHODS = {'.contiguous', '.to', '.view', '.reshape', '.transpose', '.permute',
                   '.stride', '.shape', '.numel', '.dtype', '.device', '.expand',
                   '.unsqueeze', '.squeeze', '.clone', '.detach', '.item', '.float',
                   '.half', '.bfloat16', '.int', '.long', '.bool', '.cuda', '.cpu',
                   '.is_cuda', '.data_ptr', '.size', '.dim', '.element_size'}


def _dotted(node):
    import ast
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return '.'.join(reversed(parts))


def compliance_check(source):
    """Returns (ok, issues). issues are human/model readable violation strings."""
    import ast
    issues = []
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return False, [f'code does not parse: {e}']

    fn_aliases = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and _dotted(node.module or '') == 'torch.nn.functional':
            fn_aliases.update(a.asname or a.name for a in node.names)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name == 'torch.nn.functional':
                    fn_aliases.add(a.asname or 'torch.nn.functional')

    jit_kernels = []
    host_funcs = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            deco = any(_dotted(d).endswith('triton.jit') for d in node.decorator_list)
            (jit_kernels if deco else host_funcs).append(node)
    if not jit_kernels:
        issues.append('no @triton.jit kernel is defined — all computation must be in Triton kernels')
    model_new = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'forward']

    # decoy: kernel name referenced anywhere outside its own definition?
    for k in jit_kernels:
        refs = [n for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id == k.name
                and not (n.lineno >= k.lineno and n.end_lineno <= k.end_lineno)]
        if not refs:
            issues.append(f'decoy kernel: @{k.name} is defined but never launched — '
                          f'ModelNew.forward must call {k.name}[grid](...)')

    # host-side torch compute
    def check_host(func):
        for node in ast.walk(func):
            if isinstance(node, ast.Call):
                name = _dotted(node.func)
                if name.startswith('torch.') and name not in ALLOWED_TORCH:
                    issues.append(f'host code uses torch compute: {name}() — move this '
                                  f'computation into a @triton.jit kernel')
                root = name.split('.')[0]
                if root in fn_aliases:
                    issues.append(f'host code uses torch.nn.functional compute: {name}() — '
                                  f'move this computation into a @triton.jit kernel')
                if isinstance(node.func, ast.Attribute) and node.func.attr in (
                        'matmul', 'softmax', 'sum', 'mean', 'max', 'min', 'exp', 'log',
                        'sqrt', 'rsqrt', 'pow', 'addmm', 'bmm', 'cumsum', 'argmax', 'argmin',
                        'logsumexp', 'fft', 'rfft', 'einsum'):
                    issues.append(f'host code uses tensor compute method .{node.func.attr}() — '
                                  f'move this computation into a @triton.jit kernel')
            if isinstance(node, ast.MatMult):
                issues.append('host code uses the @ matmul operator — use a @triton.jit '
                              'matmul kernel instead')
    for f in host_funcs:
        if f.name in ('get_inputs', 'get_init_inputs'):
            continue  # benchmark harness helpers, not the solution compute path
        check_host(f)
    for f in model_new:
        check_host(f)
    return (not issues), issues


def feedback_text(code_ok, status):
    if not code_ok:
        return ('Your previous reply contained no complete codeblock (it may have been cut off). '
                'Return EXACTLY ONE complete ```python codeblock defining class ModelNew, '
                'with no additional prose before or after it.')
    result = status.get('result') or {}
    per = result.get('per_workload') or []
    passed, total = result.get('passed', 0), result.get('total', len(per))
    geo = result.get('geomean_speedup')
    lines = [f'Evaluation of your last submission: {passed}/{total} workloads correct.',
             f'Geomean speedup over correct workloads: {geo if geo is not None else "n/a"}.']
    problems = [w for w in per if w.get('status') not in (None, 'PASSED')][:3]
    for w in problems:
        err = (w.get('error_log') or w.get('status') or '')[:400].replace('\n', ' ')
        lines.append(f'workload {str(w.get("uuid"))[:8]} axes={w.get("axes", {})}: '
                     f'{w.get("status")} {err}')
    if passed == total and total > 0:
        lines.append('All workloads are correct. Keep correctness and optimize the '
                     'kernels further (block sizes, vectorization, fewer passes).')
    else:
        lines.append('Fix correctness on the failing workloads first (check shape handling, '
                     'masks, dtypes), then optimize speed. Return the complete ModelNew codeblock.')
    return '\n'.join(lines)


# ---------------------------------------------------------------- campaign
RESULTS_LOCK = threading.Lock()


def run_task(pool, client, args, task, task_index, phase):
    """phase='scan': turn 0 only (rough signal, fast). 'continue': remaining
    turns + final official evaluation. State persists in state.json per turn."""
    key = task['key']
    task_dir = RUN_ROOT / 'tasks' / key
    task_dir.mkdir(parents=True, exist_ok=True)
    prompt, n_wl = build_prompt(task)
    (task_dir / 'prompt.txt').write_text(prompt)
    log(f'=== {phase} task {task_index + 1}: {key} ({n_wl} workloads) ===')

    precompute_refs(pool, args, task, task_dir)

    state_path = task_dir / 'state.json'
    if state_path.exists():
        st = json.loads(state_path.read_text())
        histories = st['histories']
        done = st['done']
        records = st['records']
        best = tuple(st['best']) if st.get('best') else None
        if best is not None and len(best) == 5:
            best = (best[0], best[1], best[2], best[3], best[4])
        start_turn = st.get('next_turn', 0)
    else:
        histories, done, records, best = [None] * N_SAMPLES, [False] * N_SAMPLES, \
            [{'samples': []} for _ in range(N_SAMPLES)], None
        start_turn = 0

    if phase == 'scan' and start_turn > 0:
        log(f'  scan already past turn 0 (next_turn={start_turn}), skip')
        return
    end_turn = 1 if phase == 'scan' else MAX_TURNS

    for turn in range(start_turn, end_turn):
        active = list(range(N_SAMPLES))
        samples, usage = gen_turn(client, args, task, histories, active, task_index, turn)
        log(f'  turn {turn + 1}: {len(active)} samples, usage={usage}')

        evals = {}
        with ThreadPoolExecutor(max_workers=args.max_parallel) as ex:
            futs = {}
            for s in active:
                code = extract_modelnew(samples[s]['text'])
                sd = task_dir / f's{s}t{turn}'
                sd.mkdir(exist_ok=True)
                (sd / 'response.txt').write_text(samples[s]['text'])
                ok, violations = (False, ['no complete codeblock extracted'])
                if code:
                    (sd / 'modelnew.py').write_text(code)
                    source = solution_source(code)
                    (sd / 'solution.py').write_text(source)
                    ok, violations = compliance_check(code)
                evals[s] = {'code_ok': code is not None, 'compliant': ok,
                            'violations': violations, 'dir': sd,
                            'finish_reason': samples[s]['finish_reason']}
                if code and ok:
                    mode = 'screen' if turn == 0 else 'full'
                    futs[s] = ex.submit(eval_candidate, pool, args, task, source, sd, mode)
                else:
                    futs[s] = None
            for s in active:
                if futs[s] is not None:
                    evals[s]['status'] = futs[s].result()

        correct_count = 0
        for s in active:
            ev = evals[s]
            base = histories[s] or [{'role': 'user', 'content': prompt}]
            if not ev['code_ok']:
                fb = feedback_text(False, None)
            elif not ev['compliant']:
                fb = ('Your submission violates the TRITON-ONLY requirement and was not '
                      'evaluated:\n- ' + '\n- '.join(ev['violations'][:6]) +
                      '\nRewrite ModelNew so that all computation happens in @triton.jit '
                      'kernels launched by forward.')
            else:
                reward, info = reward_of(ev['status'])
                ev.update({'reward': reward, 'info': info})
                fb = feedback_text(True, ev['status'])
                if info['correct_ratio'] >= 1.0:
                    correct_count += 1
                    if best is None or reward > best[0]:
                        best = (reward, info['geomean'], s, turn,
                                (ev['dir'] / 'solution.py').read_text())
            histories[s] = base + [{'role': 'assistant', 'content': compact(samples[s]['text'])},
                                   {'role': 'user', 'content': fb}]
            records[s]['samples'].append({'turn': turn, 'finish_reason': ev['finish_reason'],
                                          'code_ok': ev['code_ok'], 'compliant': ev['compliant'],
                                          'violations': ev['violations'][:6],
                                          'status': ev.get('status', {}).get('result'),
                                          'reward': ev.get('reward'),
                                          'info': ev.get('info')})
        log(f'  turn {turn + 1}: {correct_count}/{len(active)} samples fully correct '
            f'(all turns continue for speedup exploration)')
        state_path.write_text(json.dumps(
            {'histories': histories, 'done': [True] * N_SAMPLES if end_turn == MAX_TURNS else [False] * N_SAMPLES,
             'records': records, 'best': list(best) if best else None,
             'next_turn': turn + 1}, default=str))

    if phase == 'scan':
        recs = [r for s in range(N_SAMPLES) for r in records[s]['samples']]
        any_correct = any((r.get('info') or {}).get('correct_ratio', 0) >= 1.0 for r in recs)
        geos = [(r.get('info') or {}).get('geomean') for r in recs
                if (r.get('info') or {}).get('correct_ratio', 0) >= 1.0]
        row = {'task': key, 'extracted': sum(1 for r in recs if r['code_ok']) / len(recs),
               'compliant': sum(1 for r in recs if r.get('compliant')) / len(recs),
               'any_wl0_correct': any_correct,
               'best_wl0_geomean': max((g for g in geos if g), default=None)}
        with RESULTS_LOCK, open(RUN_ROOT / 'scan_results.jsonl', 'a') as fh:
            fh.write(json.dumps(row, default=str) + '\n')
        log(f'=== scan done: {key} extracted={row["extracted"]:.2f} '
            f'compliant={row["compliant"]:.2f} any_correct={any_correct}')
        return

    # exploration ceiling: best compliant fully-correct sample-turn by feedback geomean
    all_recs = [r for s in range(N_SAMPLES) for r in records[s]['samples']]
    good_geos = [(r.get('info') or {}).get('geomean') for r in all_recs
                 if r.get('compliant') and (r.get('info') or {}).get('correct_ratio', 0) >= 1.0]
    pass1 = sum(1 for s in range(N_SAMPLES)
                if any((r.get('info') or {}).get('correct_ratio', 0) >= 1.0
                       for r in records[s]['samples']))
    summary = {'task': key, 'workloads': n_wl,
               'pass_at_1': pass1 / N_SAMPLES,
               'best_of_8x3_geomean_feedback': max((g for g in good_geos if g), default=None),
               'any_compliant_correct': bool(good_geos),
               'extraction_rate': sum(1 for r in all_recs if r['code_ok']) / len(all_recs),
               'compliance_rate': sum(1 for r in all_recs if r.get('compliant')) / len(all_recs),
               'samples': records}
    if best is not None:
        log(f'  final official eval (best reward={best[0]:.3f} sample={best[2]} turn={best[3]})')
        corr, perf = final_official(pool, args, task, best[4], task_dir / 'final')
        summary['final_correctness'] = {'valid': (corr.get('result') or {}).get('valid'),
                                        'passed': (corr.get('result') or {}).get('passed'),
                                        'total': (corr.get('result') or {}).get('total')}
        if perf is not None:
            summary['final_geomean_speedup'] = (perf.get('result') or {}).get('geomean_speedup')
        (task_dir / 'best_solution.py').write_text(best[4])
    (task_dir / 'summary.json').write_text(json.dumps(summary, indent=2, default=str))
    with open(RUN_ROOT / 'results.jsonl', 'a') as fh:
        fh.write(json.dumps({k: v for k, v in summary.items() if k != 'samples'},
                            default=str) + '\n')
    log(f'=== task done: {key} pass@1={summary["pass_at_1"]:.2f} '
        f'final_geomean={summary.get("final_geomean_speedup")}')


def main():
    args = parse_args()
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks(args.task)
    log(f'campaign start (phase={args.phase}): {len(tasks)} tasks, '
        f'order: ' + ' -> '.join(t['key'] for t in tasks[:3]) + ' ...')
    pool = GPUPool([int(g) for g in args.gpus.split(',')], args.max_parallel, args.poll_seconds)
    client = OpenAI(api_key='EMPTY', base_url=args.server, timeout=900, max_retries=2)

    def eligible(task):
        return not (RUN_ROOT / 'tasks' / task['key'] / 'summary.json').exists()

    def safe_run(phase):
        def _run(item):
            i, task = item
            try:
                run_task(pool, client, args, task, i, phase)
            except Exception as e:
                log(f'TASK FAILED ({phase}) {task["key"]}: {type(e).__name__}: {e}')
        return _run

    if args.phase in ('scan', 'both'):
        for i, task in enumerate(t for t in tasks if eligible(t)):
            safe_run('scan')((i, task))
        log('scan phase complete -> ' + str(RUN_ROOT / 'scan_results.jsonl'))
    if args.phase in ('continue', 'both'):
        todo = [(i, t) for i, t in enumerate(tasks) if eligible(t)]
        with ThreadPoolExecutor(max_workers=args.concurrent_tasks) as ex:
            list(ex.map(safe_run('continue'), todo))
    log('campaign complete. results: ' + str(RUN_ROOT / 'results.jsonl'))


if __name__ == '__main__':
    main()
