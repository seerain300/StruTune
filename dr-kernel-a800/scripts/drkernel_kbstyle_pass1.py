#!/usr/bin/env python3
"""Pass@1 experiment for drkernel-8b using the official DR.Kernel prompt/sampling contract.

Differences vs drtriton_feedback_10r.py:
- official KernelBench-style 1-shot prompt (Model/get_inputs + ModelNew + codeblocks)
- full workload axes list appended to the prompt
- sampling: temperature=1.0, top_p=0.95, stop tokens fixed (no more blank padding to max_tokens)
- extraction: last markdown codeblock containing ModelNew; no triton-only gates
- solution wrapper: def run(*args): return ModelNew()(*args)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import signal
import time
from pathlib import Path

from openai import OpenAI

WS = Path('/data1/workspace/weihongren')
DRT = WS / 'DRTriton'
SOL = Path('/data1/workspace/ziming/dataset/SOL-ExecBench')
FIB = WS / 'dataset/flashinfer-test'
MODEL_NAME = 'drkernel-stop-8b'
SERVER = 'http://127.0.0.1:8001/v1'
PLAN = WS / 'baseline/drtriton/tts_feedback_10r_20260916/task_plan.json'
OLD_ROOT = WS / 'baseline/drtriton/tts_feedback_10r_20260916/feedback'
RUN_ROOT = WS / 'baseline/drtriton/kbstyle_pass1_20260918'
FIB_EVAL = WS / 'evaluators/evaluate.py'
SOL_EVAL = WS / 'evaluators/evaluate_sol.py'
MTMC_PY = Path('/data1/workspace/ziming/miniconda3/envs/mtmc/bin/python')
SOL_PY = SOL / '.venv/bin/python'

os.environ['NO_PROXY'] = ','.join(filter(None, [os.environ.get('NO_PROXY', ''), '127.0.0.1', 'localhost']))
os.environ['no_proxy'] = ','.join(filter(None, [os.environ.get('no_proxy', ''), '127.0.0.1', 'localhost']))

DEFAULT_TASKS = [
    'flashinfer/gemm_n4096_k4096',
    'flashinfer/rmsnorm_h4096',
    'flashinfer/gqa_paged_decode_h32_kv8_d128_ps1',
    'flashinfer/mla_paged_decode_h16_ckv512_kpe64_ps1',
    'flashinfer/gdn_decode_qk4_v8_d128_k_last',
    'SOL/L1/053_gaussian_topk_sparse_activation',
    'SOL/L1/092_gqa_attention_with_qk_norm',
    'SOL/L1/018_fused_rope_with_qk_norm_and_kv_cache_update',
    'SOL/L1/008_expert_output_weighted_index_add_accumulation',
    'SOL/L2/043_mamba_chunk_scan_with_segsum',
]

EXAMPLE_REF_CODE = '''\
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a, b):
        return a + b

def get_inputs():
    # randomly generate input tensors based on the model architecture
    a = torch.randn(1, 128).cuda()
    b = torch.randn(1, 128).cuda()
    return [a, b]

def get_init_inputs():
    # randomly generate tensors required for initialization based on the model architecture
    return []
'''

EXAMPLE_KERNEL_CODE = '''\
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def add_kernel(
    x_ptr,  # Pointer to first input
    y_ptr,  # Pointer to second input
    out_ptr,  # Pointer to output
    n_elements,  # Total number of elements in input/output
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < n_elements
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    # Perform the elementwise addition
    out = x + y
    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)

def triton_add(x: torch.Tensor, y: torch.Tensor):
    """
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert x.is_cuda and y.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    y = y.contiguous()

    # Prepare output tensor
    out = torch.empty_like(x)

    # Number of elements in the tensor
    n_elements = x.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out

class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a, b):
        # Instead of "return a + b", call our Triton-based addition
        return triton_add(a, b)
'''

PROMPT_TEMPLATE = """\
You write custom Triton kernels to replace the pytorch operators in the given architecture to get speedups.

You have complete freedom to choose the set of operators you want to replace. You may make the decision to replace some operators with custom Triton kernels and leave others unchanged. You may replace multiple operators with custom implementations, consider operator fusion opportunities (combining multiple operators into a single kernel, for example, combining matmul+relu), or algorithmic changes (such as online softmax). You are only limited by your imagination.

Here's an example to show you the syntax of inline embedding custom Triton kernels in torch: The example given architecture is:

```python
{example_ref_code}
```

The example new arch with custom Triton kernels looks like this:

```python
{example_kernel_code}
```

You are given the following architecture:
```python
{ref_code}
```

Optimize the architecture named Model with custom Triton operators! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Let's think step by step.
"""

WORKLOAD_SECTION = """

Before you write the kernel, note how it will be evaluated: ModelNew will be benchmarked on the {n} workloads listed below, one JSON object of axis values per line (each line is a separate evaluation configuration). Your implementation must be correct and efficient on ALL of these configurations. Axes that vary across workloads are dynamic dimensions — handle them generically (or specialize per launch); constants asserted in the reference implementation may be treated as fixed.

{axes_lines}
"""

TRITON_ONLY_SECTION = """

STRICT REQUIREMENT — TRITON-ONLY COMPUTATION:
1. ALL numerical computation must be performed by custom @triton.jit kernels that you write. The replacement of the PyTorch operators must be real: the computation done by torch operators in the reference must be done by your Triton kernel(s).
2. ModelNew.forward (and any host-side helper it calls) may ONLY: compute shapes/strides/grid sizes, allocate output tensors (torch.empty / torch.empty_like / torch.zeros / torch.zeros_like / torch.empty_strided), make tensors contiguous, and launch your Triton kernel(s) as kernel[grid](...).
3. Do NOT use any torch computation in the host code: no torch.matmul / torch.nn.functional.linear / softmax / einsum / fft / elementwise math / reductions / @ operator on tensors. Tensor methods like .contiguous(), .to(dtype), .view(), .reshape(), .transpose(), .stride(), .shape are fine (they are data movement/metadata, not computation).
4. The @triton.jit kernel(s) you define MUST actually be launched by ModelNew.forward. Defining a kernel that is never called (while computing the result with torch instead) is invalid and will be rejected.
"""


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, default=6)
    p.add_argument('--task', action='append', default=[])
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--top-p', type=float, default=0.95)
    p.add_argument('--max-tokens', type=int, default=8192)
    p.add_argument('--seed', type=int, default=20260918)
    p.add_argument('--eval-timeout', type=int, default=900)
    p.add_argument('--poll-seconds', type=int, default=15)
    p.add_argument('--skip-eval', action='store_true', help='generation + extraction only, no evaluator runs')
    return p.parse_args()


def load_tasks(wanted):
    rows = json.loads(PLAN.read_text())
    order = {k: i for i, k in enumerate(wanted)}
    return sorted([r for r in rows if r['key'] in order], key=lambda r: order[r['key']])


def workload_axes_lines(task):
    lines = []
    for raw in Path(task['workload']).read_text().splitlines():
        if not raw.strip():
            continue
        row = json.loads(raw)
        axes = row.get('axes') or row.get('workload', {}).get('axes') or {}
        lines.append(json.dumps(axes, ensure_ascii=False))
    return lines


def build_prompt(task):
    ref_code = task['pytorch'].rstrip() + '''

class Model(torch.nn.Module):
    def forward(self, *args):
        return run(*args)
'''
    prompt = PROMPT_TEMPLATE.format(
        example_ref_code=EXAMPLE_REF_CODE,
        example_kernel_code=EXAMPLE_KERNEL_CODE,
        ref_code=ref_code,
    )
    axes_lines = workload_axes_lines(task)
    if axes_lines:
        prompt += WORKLOAD_SECTION.format(n=len(axes_lines), axes_lines='\n'.join(axes_lines))
    prompt += TRITON_ONLY_SECTION
    return prompt, len(axes_lines)


def generate(client, task, prompt, args, index):
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{'role': 'user', 'content': prompt}],
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=['<|im_end|>', '<|endoftext|>'],
        seed=args.seed + index,
    )
    choice = resp.choices[0]
    return choice.message.content or '', {
        'finish_reason': choice.finish_reason,
        'prompt_tokens': getattr(resp.usage, 'prompt_tokens', None),
        'completion_tokens': getattr(resp.usage, 'completion_tokens', None),
        'seed': args.seed + index,
    }


FENCE_RE = re.compile(r'```(?:python|py)?\s*\n(.*?)(?:```|\Z)', re.DOTALL)


def extract_modelnew(text):
    blocks = FENCE_RE.findall(text)
    candidates = [b for b in blocks if 'ModelNew' in b and 'class' in b]
    for block in reversed(candidates):
        code = block.strip('\n')
        try:
            compile(code, '<modelnew>', 'exec')
        except SyntaxError:
            continue
        return code
    return None


def solution_source(modelnew_code):
    return modelnew_code.rstrip() + '\n\n\ndef run(*args):\n    return ModelNew()(*args)\n'


def sol_document(task, source):
    return {
        'name': f"drkernel_kbstyle_{task['name']}", 'definition': task['name'],
        'author': 'drkernel', 'description': 'official-style pass@1 candidate',
        'spec': {'languages': ['triton'], 'target_hardware': ['LOCAL'],
                 'entry_point': 'solution.py::run', 'dependencies': ['torch', 'triton'],
                 'destination_passing_style': False, 'binding': None},
        'sources': [{'path': 'solution.py', 'content': source}],
    }


def gpu_processes(gpu):
    result = subprocess.run(['nvidia-smi', '-i', str(gpu), '--query-compute-apps=pid,process_name,used_memory',
                             '--format=csv,noheader'], capture_output=True, text=True, check=False)
    return [x for x in result.stdout.splitlines() if x.strip()]


def wait_gpu(gpu, poll):
    while True:
        procs = gpu_processes(gpu)
        if not procs:
            return
        print(f'GPU {gpu} busy; waiting {poll}s: {procs}', flush=True)
        time.sleep(poll)


def run_timed(command, env, log, timeout):
    started = time.time()
    with log.open('w') as stream:
        proc = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            rc = proc.wait(timeout=timeout)
            timed_out = False
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            rc = 124
    return {'exit_code': rc, 'timed_out': timed_out, 'elapsed_seconds': round(time.time() - started, 2)}


def evaluate(task, source, out_dir, gpu, stage, timeout, poll):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'solution.py').write_text(source)
    output = out_dir / 'evaluation.json'
    log = out_dir / 'evaluation.log'
    wait_gpu(gpu, poll)
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu)}
    if task['benchmark'] == 'flashinfer':
        cmd = [str(MTMC_PY), str(FIB_EVAL), '--definition', task['definition'], '--workload', task['workload'],
               '--solution', str(out_dir / 'solution.py'), '--entry', 'run', '--dataset-root', str(FIB),
               '--device', 'cuda:0', '--json', str(output), '--trials', '1']
        if stage == 'correctness':
            cmd += ['--correctness-only']
        else:
            cmd += ['--warmup', '3', '--iters', '100']
    else:
        solution_json = out_dir / 'solution.json'
        solution_json.write_text(json.dumps(sol_document(task, source), indent=2))
        cmd = [str(SOL_PY), str(SOL_EVAL), '--definition', task['definition'], '--workload', task['workload'],
               '--solution', str(solution_json), '--output', str(output), '--timeout', str(timeout), '--rerun']
        if stage == 'correctness':
            cmd += ['--correctness-only', '--warmup', '0']
        else:
            cmd += ['--iterations', '100', '--warmup', '10']
    status = run_timed(cmd, env, log, timeout)
    try:
        result = json.loads(output.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        result = {}
    status.update({'stage': stage, 'result': result})
    return status


def old_campaign_result(key):
    summary = OLD_ROOT / key / 'summary.json'
    if not summary.exists():
        return {'note': 'no old feedback run (was tts task)'}
    data = json.loads(summary.read_text())
    perf = data.get('final_performance') or {}
    geo = (perf.get('result') or {}).get('geomean_speedup')
    valid = data.get('valid_feedback_rounds', 0)
    return {'old_valid_rounds': valid, 'old_selected_score': data.get('selected_score'),
            'old_final_geomean': geo if geo is not None else None}


def main():
    args = parse_args()
    wanted = args.task or DEFAULT_TASKS
    tasks = load_tasks(wanted)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    client = OpenAI(api_key='EMPTY', base_url=SERVER, timeout=600, max_retries=2)

    rows = []
    for index, task in enumerate(tasks):
        key = task['key']
        task_dir = RUN_ROOT / 'tasks' / key
        task_dir.mkdir(parents=True, exist_ok=True)
        print(f'=== pass@1 task {index + 1}/{len(tasks)}: {key} ===', flush=True)

        prompt, n_workloads = build_prompt(task)
        (task_dir / 'prompt.txt').write_text(prompt)

        raw, usage = generate(client, task, prompt, args, index)
        (task_dir / 'response.txt').write_text(raw)

        code = extract_modelnew(raw)
        record = {'task': key, 'workloads_listed': n_workloads, 'usage': usage,
                  'has_triton_jit': bool(code and '@triton.jit' in code),
                  'code_blocks': len(FENCE_RE.findall(raw))}
        if code is None:
            record.update({'status': 'EXTRACTION_ERROR'})
            print(f'[{key}] EXTRACTION_ERROR finish={usage["finish_reason"]}', flush=True)
        else:
            (task_dir / 'modelnew.py').write_text(code)
            source = solution_source(code)
            (task_dir / 'solution.py').write_text(source)
            record['status'] = 'GENERATED'
            print(f'[{key}] GENERATED jit={record["has_triton_jit"]} finish={usage["finish_reason"]}', flush=True)
            if not args.skip_eval:
                correctness = evaluate(task, source, task_dir / 'correctness', args.gpu, 'correctness',
                                       args.eval_timeout, args.poll_seconds)
                record['correctness'] = correctness
                result = correctness.get('result') or {}
                if result.get('valid') is True:
                    perf = evaluate(task, source, task_dir / 'performance', args.gpu, 'performance',
                                    args.eval_timeout, args.poll_seconds)
                    record['performance'] = perf
                    speed = (perf.get('result') or {}).get('geomean_speedup')
                    record['status'] = 'PASSED'
                    record['geomean_speedup'] = speed
                    print(f'[{key}] PASSED geomean={speed}', flush=True)
                else:
                    failed = (result.get('per_workload') or [{}])[0]
                    record['status'] = f'FAILED::{result.get("passed", 0)}/{result.get("total", "?")} {failed.get("status", "")}'
                    print(f'[{key}] FAILED {record["status"]}', flush=True)
        record['old'] = old_campaign_result(key)
        rows.append(record)
        (task_dir / 'record.json').write_text(json.dumps(record, indent=2, ensure_ascii=False, default=str))

    passed = [r for r in rows if r['status'] == 'PASSED']
    generated = [r for r in rows if r['status'] in ('PASSED', 'GENERATED')]
    truncated = [r for r in rows if (r.get('usage') or {}).get('finish_reason') == 'length']
    summary = {
        'date': '2026-09-18', 'model': MODEL_NAME, 'sampling': {'temperature': args.temperature, 'top_p': args.top_p,
        'max_tokens': args.max_tokens, 'seed_base': args.seed, 'stop': ['<|im_end|>', '<|endoftext|>']},
        'skip_eval': args.skip_eval,
        'tasks': len(rows), 'generated': len(generated), 'passed': len(passed),
        'pass_rate': (len(passed) / len(rows)) if not args.skip_eval else None,
        'truncated_responses': len(truncated),
        'results': rows,
    }
    (RUN_ROOT / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    print('\n===== pass@1 summary =====')
    for r in rows:
        speed = r.get('geomean_speedup')
        usage = r.get('usage') or {}
        print(f"{r['status']:<28} jit={str(r['has_triton_jit']):<5} "
              f"tok={usage.get('completion_tokens') or '-':<6} fin={usage.get('finish_reason') or '-':<8} "
              f"geomean={speed if speed is not None else '-':<8} {r['task']}")
    print(f"generated {len(generated)}/{len(rows)}  passed {len(passed)}/{len(rows)}  truncated {len(truncated)}/{len(rows)}")


if __name__ == '__main__':
    main()
