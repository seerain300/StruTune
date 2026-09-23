#!/usr/bin/env python3
"""Flashinfer reference cache + feedback evaluator for the kbstyle campaign.

Modes (run under the mtmc python, same env as evaluators/evaluate.py):
  precompute  — per workload: seed RNG, generate inputs, run reference, time it
                (warmup/iters), save normalized outputs + ref_ms to cache dir.
  feedback    — per workload: regenerate identical inputs (same seed), run the
                candidate, compare against cached outputs with evaluate.py's
                exact tolerance logic, time the candidate only, speedup =
                cached ref_ms / sol_ms. Aggregates like evaluate.py.
  screen      — feedback on workload index 0 only (cheap first-turn gate).

Reuses evaluators/evaluate.py machinery by import so comparison/timing
semantics are identical to the official final evaluation.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'evaluators'))
import evaluate as official  # noqa: E402  (evaluators/evaluate.py)

from flashinfer_bench.bench.config import BenchmarkConfig  # noqa: E402
from flashinfer_bench.bench.timing import time_runnable  # noqa: E402
from flashinfer_bench.bench.utils import (  # noqa: E402
    gen_inputs, load_safetensors, normalize_outputs,
)
from flashinfer_bench.compile import BuilderRegistry  # noqa: E402
from flashinfer_bench.data import Definition  # noqa: E402

BASE_SEED = 20260918


def seed_for(uuid: str) -> int:
    return (zlib.crc32(uuid.encode()) + BASE_SEED) % (2 ** 31)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_workloads(workload_path: Path):
    return official._load_workload_records(workload_path)


def make_cfg(args):
    tol = dict(official.DEFAULT_TOL)
    definition = Definition.model_validate(json.load(open(args.definition)))
    tol.update(official.OP_TYPE_TOL.get(definition.op_type, {}))
    return definition, BenchmarkConfig(
        warmup_runs=args.warmup, iterations=args.iters, num_trials=1,
        atol=tol['atol'], rtol=tol['rtol'],
        required_matched_ratio=tol['required_matched_ratio'], profile_baseline=False)


def make_inputs(definition, wl, device, dataset_root):
    safe = (load_safetensors(definition, wl, dataset_root)
            if any(d.type == 'safetensors' for d in wl.inputs.values()) else {})
    set_seed(seed_for(wl.uuid))
    return gen_inputs(definition, wl, device=device, safe_tensors=safe)


def mode_precompute(args):
    device = args.device
    torch.cuda.set_device(device)
    definition, cfg = make_cfg(args)
    workloads = load_workloads(Path(args.workload))
    ref_runnable = BuilderRegistry.get_instance().build_reference(definition)
    output_names = list(definition.outputs.keys())
    output_dtypes = {k: official.dtype_str_to_torch_dtype(v.dtype) for k, v in definition.outputs.items()}
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i, wl in enumerate(workloads):
        path = cache_dir / f'{wl.uuid}.pt'
        if path.exists():
            manifest.append({'uuid': wl.uuid, 'cached': True})
            continue
        started = time.time()
        inputs = make_inputs(definition, wl, device, Path(args.dataset_root))
        with torch.no_grad():
            r = ref_runnable(*inputs)
        torch.cuda.synchronize(device)
        ref_out = normalize_outputs(r, device=torch.device(device),
                                    output_names=output_names, output_dtypes=output_dtypes)
        ref_ms = time_runnable(ref_runnable, inputs, cfg.warmup_runs, cfg.iterations, device)
        payload = {'uuid': wl.uuid, 'axes': dict(wl.axes), 'ref_ms': float(ref_ms),
                   'outputs': {k: v.detach().to('cpu', torch.float32).clone() for k, v in ref_out.items()}}
        torch.save(payload, path)
        manifest.append({'uuid': wl.uuid, 'cached': False, 'ref_ms': float(ref_ms),
                         'seconds': round(time.time() - started, 2)})
        print(f'[{i + 1}/{len(workloads)}] {wl.uuid[:8]} ref_ms={ref_ms:.4f} '
              f'({manifest[-1]["seconds"]}s)', flush=True)
    (cache_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f'precompute done: {len(manifest)} workloads -> {cache_dir}')


def mode_feedback(args, screen: bool):
    device = args.device
    torch.cuda.set_device(device)
    definition, cfg = make_cfg(args)
    workloads = load_workloads(Path(args.workload))
    if screen:
        workloads = workloads[:1]
    run_fn = official._load_solution_run(Path(args.solution), 'run')
    output_names = list(definition.outputs.keys())
    output_dtypes = {k: official.dtype_str_to_torch_dtype(v.dtype) for k, v in definition.outputs.items()}
    cache_dir = Path(args.cache_dir)

    per_workload = []
    all_pass = True
    for i, wl in enumerate(workloads):
        tag = f'[{i + 1}/{len(workloads)}] {wl.uuid[:8]}'
        cache_path = cache_dir / f'{wl.uuid}.pt'
        if not cache_path.exists():
            all_pass = False
            per_workload.append({'uuid': wl.uuid, 'status': 'CACHE_MISS', 'speedup': None,
                                 'error_log': f'reference cache missing: {cache_path}'})
            continue
        cached = torch.load(cache_path, map_location='cpu', weights_only=False)
        try:
            inputs = make_inputs(definition, wl, device, Path(args.dataset_root))
            with torch.no_grad():
                res = run_fn(*inputs)
            torch.cuda.synchronize(device)
            sol_out = normalize_outputs(res, device=torch.device(device),
                                        output_names=output_names, output_dtypes=output_dtypes)
        except Exception as e:
            all_pass = False
            detail = f'{type(e).__name__}: {e}'
            print(f'{tag}: RUNTIME_ERROR  {detail}', flush=True)
            per_workload.append({'uuid': wl.uuid, 'status': 'RUNTIME_ERROR', 'speedup': None,
                                 'error_log': detail})
            continue

        status, detail, max_abs, max_rel = 'PASSED', '', 0.0, 0.0
        for name in output_names:
            s = sol_out[name].detach().to('cpu', torch.float32)
            r = cached['outputs'][name].to(torch.float32)
            status, detail, max_abs, max_rel = official._compare_output(name, s, r, cfg)
            if status != 'PASSED':
                break
        if status != 'PASSED':
            all_pass = False
            print(f'{tag}: {status}  {detail}', flush=True)
            per_workload.append({'uuid': wl.uuid, 'status': status, 'speedup': None,
                                 'max_abs': max_abs, 'max_rel': max_rel, 'error_log': detail})
            continue

        sol_ms = time_runnable(run_fn, inputs, cfg.warmup_runs, cfg.iterations, device)
        speedup = cached['ref_ms'] / sol_ms if sol_ms > 0 else float('inf')
        print(f'{tag}: PASSED ref={cached["ref_ms"]:.4f}ms sol={sol_ms:.4f}ms '
              f'speedup={speedup:.2f}x', flush=True)
        per_workload.append({'uuid': wl.uuid, 'status': 'PASSED', 'speedup': speedup,
                             'ref_ms': cached['ref_ms'], 'sol_ms': float(sol_ms),
                             'max_abs': max_abs, 'max_rel': max_rel, 'axes': dict(wl.axes)})

    speedups = [w['speedup'] for w in per_workload
                if w['status'] == 'PASSED' and w.get('speedup') is not None]
    passed = sum(1 for w in per_workload if w['status'] == 'PASSED')
    geo = official._geomean(speedups) if speedups else float('nan')
    summary = {'definition': definition.name, 'op_type': definition.op_type,
               'valid': all_pass, 'passed': passed, 'total': len(per_workload),
               'geomean_speedup': geo if speedups else None,
               'arithmetic_mean_speedup': (sum(speedups) / len(speedups)) if speedups else None,
               'tolerance': {'atol': cfg.atol, 'rtol': cfg.rtol,
                             'required_matched_ratio': cfg.required_matched_ratio},
               'per_workload': per_workload}
    if args.json:
        Path(args.json).write_text(json.dumps(summary, indent=2))
    print(f'RESULT: valid={all_pass} passed={passed}/{len(per_workload)} '
          f'geomean={summary["geomean_speedup"]}')
    return 0 if all_pass else 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['precompute', 'feedback', 'screen'], required=True)
    p.add_argument('--definition', required=True)
    p.add_argument('--workload', required=True)
    p.add_argument('--solution', default=None)
    p.add_argument('--cache-dir', required=True)
    p.add_argument('--dataset-root', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--warmup', type=int, default=3)
    p.add_argument('--iters', type=int, default=10)
    p.add_argument('--json', default=None)
    args = p.parse_args()
    if args.mode == 'precompute':
        mode_precompute(args)
    else:
        sys.exit(mode_feedback(args, screen=(args.mode == 'screen')))


if __name__ == '__main__':
    main()
