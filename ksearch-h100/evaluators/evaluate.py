#!/usr/bin/env python3
"""Correctness-then-speed evaluator for MLSys'26 FlashInfer-Bench kernels.

This is a *standalone* evaluator built on top of flashinfer-bench's own
primitives (``gen_inputs`` / ``load_safetensors`` / ``compute_error_stats`` /
``time_runnable`` / ``build_reference``), so its correctness and timing semantics
match the official contest harness, while being:

  * self-contained (one file, drives a single ``solution.py`` directly),
  * A800 / sm_80 friendly (no Hopper/Blackwell-only assumptions),
  * geometric-mean oriented (the metric this workflow optimizes).

Evaluation order (per the task):
  1. CORRECTNESS first — for every workload, run the solution on the same
     inputs as the reference and compare (shape + dtype + finite + per-element
     tolerance, identical math to flashinfer_bench DefaultEvaluator).
  2. SPEED only if correct — time reference and solution with CUDA events
     (L2 flushed, args cloned per-iter), per-workload speedup = ref_ms/sol_ms.
  3. SCORE — geometric mean of per-workload speedups (also reports the
     arithmetic mean AKO4X/flashinfer-bench use, for cross-reference).

A workload that fails correctness makes the run INVALID (matches the contest:
all workloads must PASS). The geomean is still printed over the PASSED subset
for debugging, but clearly flagged.

Usage
-----
    python scripts/evaluate.py \
        --definition task/definition.json \
        --workload   task/workload.jsonl \
        --solution   solution/solution.py \
        --dataset-root $FIB_DATASET_PATH

    python scripts/evaluate.py \
        --workload task/workload.jsonl \
        --solution solution/solution.py \
        --workload-index 1,2,3

The solution file must expose a value-returning ``run(...)`` whose positional
args are the definition's inputs *in declaration order* and whose return is the
definition's outputs in declaration order (a single value, a tuple, or a dict
keyed by output name).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List

import torch

from flashinfer_bench.bench.config import BenchmarkConfig
from flashinfer_bench.bench.timing import time_runnable

try:
    from gpu_occupancy import gpu_occupancy_lease, physical_gpu_from_environment
except ImportError:
    import sys as _sys
    _sys.path.insert(0, "/home/ziming/MTMC-baseline/agent-generation/scripts")
    from gpu_occupancy import gpu_occupancy_lease, physical_gpu_from_environment
from flashinfer_bench.bench.utils import (
    compute_error_stats,
    gen_inputs,
    load_safetensors,
    normalize_outputs,
)
from flashinfer_bench.compile import BuilderRegistry
from flashinfer_bench.data import Definition, Workload
from flashinfer_bench.utils import dtype_str_to_torch_dtype

# Per-op_type default tolerances. Mirror AKO4X/flashinfer-bench:
#   default: atol=rtol=1e-2, every element must match.
#   moe:     atol=1.0, rtol=0.3, 90% of elements must match (routing ties).
DEFAULT_TOL = {"atol": 1e-2, "rtol": 1e-2, "required_matched_ratio": 0.99}
OP_TYPE_TOL = {
    "moe": {"atol": 1.0, "rtol": 0.3, "required_matched_ratio": 0.9},
}


def _load_solution_run(solution_path: Path, entry: str):
    spec = importlib.util.spec_from_file_location("user_solution", str(solution_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import solution from {solution_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["user_solution"] = module
    spec.loader.exec_module(module)  # compiles cuda ext at import (load_inline)
    if not hasattr(module, entry):
        raise RuntimeError(f"Solution {solution_path} has no '{entry}' function")
    return getattr(module, entry)


def _load_workload_records(path: Path) -> List[Workload]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            wl = obj["workload"] if "workload" in obj else obj
            out.append(Workload.model_validate(wl))
    return out


def _geomean(xs: List[float]) -> float:
    if not xs:
        return float("nan")
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def _fmt(x: float) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "  n/a "
    return f"{x:8.2f}x"


def _parse_workload_indices(value: str | None) -> List[int] | None:
    if value is None:
        return None
    items = [part.strip() for part in value.split(",")]
    if not items or any(not part for part in items):
        raise ValueError("--workload-index expects a comma-separated list of integers")
    try:
        indices = [int(part) for part in items]
    except ValueError as exc:
        raise ValueError("--workload-index expects a comma-separated list of integers") from exc
    if any(idx < 0 for idx in indices):
        raise ValueError("--workload-index values must be non-negative")
    return indices


def _compare_output(name: str, candidate: torch.Tensor, reference: torch.Tensor, cfg):
    """Compare outputs while preserving the operator's valid -inf sentinel."""
    if tuple(candidate.shape) != tuple(reference.shape):
        return "INCORRECT_SHAPE", f"{name}: {tuple(candidate.shape)} vs {tuple(reference.shape)}", 0.0, 0.0
    if candidate.dtype != reference.dtype:
        return "INCORRECT_DTYPE", f"{name}: {candidate.dtype} vs {reference.dtype}", 0.0, 0.0

    candidate_nan = torch.isnan(candidate)
    reference_nan = torch.isnan(reference)
    if (candidate_nan | reference_nan).any().item():
        return "INCORRECT_NUMERICAL", f"{name}: NaN output", 0.0, 0.0

    matching_neg_inf = torch.isneginf(candidate) & torch.isneginf(reference)
    nonfinite_mismatch = (torch.isinf(candidate) | torch.isinf(reference)) & ~matching_neg_inf
    if nonfinite_mismatch.any().item():
        return "INCORRECT_NUMERICAL", f"{name}: non-finite output mismatch", 0.0, 0.0

    finite = torch.isfinite(candidate) & torch.isfinite(reference)
    if not finite.any().item():
        return "PASSED", "", 0.0, 0.0

    max_abs, max_rel, exceeds, ratio = compute_error_stats(
        candidate[finite], reference[finite], cfg
    )
    if exceeds:
        return (
            "INCORRECT_NUMERICAL",
            f"{name}: matched_ratio={ratio:.4f} abs={max_abs:.3e} rel={max_rel:.3e}",
            max_abs,
            max_rel,
        )
    return "PASSED", "", max_abs, max_rel


def _evaluate(args) -> dict:
    device = args.device
    torch.cuda.set_device(device)

    definition = Definition.model_validate(json.load(open(args.definition)))
    workloads = _load_workload_records(Path(args.workload))
    if args.limit > 0:
        workloads = workloads[: args.limit]
    selected_indices = _parse_workload_indices(args.workload_index)
    if selected_indices is not None:
        max_index = len(workloads) - 1
        for idx in selected_indices:
            if idx > max_index:
                raise ValueError(
                    f"--workload-index {idx} is out of range for {len(workloads)} workloads"
                )
        workloads = [workloads[idx] for idx in selected_indices]

    dataset_root = args.dataset_root or os.environ.get("FIB_DATASET_PATH")
    if dataset_root:
        dataset_root = Path(dataset_root)

    # Tolerances: op_type default unless overridden on the CLI.
    tol = dict(DEFAULT_TOL)
    tol.update(OP_TYPE_TOL.get(definition.op_type, {}))
    if args.atol is not None:
        tol["atol"] = args.atol
    if args.rtol is not None:
        tol["rtol"] = args.rtol
    if args.matched_ratio is not None:
        tol["required_matched_ratio"] = args.matched_ratio
    cfg = BenchmarkConfig(
        warmup_runs=args.warmup,
        iterations=args.iters,
        num_trials=args.trials,
        atol=tol["atol"],
        rtol=tol["rtol"],
        required_matched_ratio=tol["required_matched_ratio"],
        profile_baseline=False,
    )

    ref_cache = None
    if args.ref_cache:
        import json as _json
        ref_cache = _json.loads(Path(args.ref_cache).read_text())["per_workload"]
        print(f"RefCache  : {args.ref_cache} ({len(ref_cache)} entries, live ref timing SKIPPED)")

    run_fn = _load_solution_run(Path(args.solution), args.entry)
    ref_runnable = BuilderRegistry.get_instance().build_reference(definition)

    output_names = list(definition.outputs.keys())
    output_dtypes = {k: dtype_str_to_torch_dtype(v.dtype) for k, v in definition.outputs.items()}

    print("=" * 78)
    print(f"Definition : {definition.name}  (op_type={definition.op_type})")
    print(f"Solution   : {args.solution}")
    print(f"Device     : {device}  ({torch.cuda.get_device_name(device)})")
    print(f"Workloads  : {len(workloads)}   trials={cfg.num_trials} "
          f"warmup={cfg.warmup_runs} iters={cfg.iterations}")
    if selected_indices is not None:
        print(f"Selected   : {selected_indices}")
    print(f"Tolerance  : atol={cfg.atol} rtol={cfg.rtol} "
          f"matched_ratio={cfg.required_matched_ratio}")
    print("=" * 78)

    per_workload = []
    all_pass = True

    for i, wl in enumerate(workloads):
        tag = f"[{i + 1}/{len(workloads)}] {wl.uuid[:8]} axes={dict(wl.axes)}"
        try:
            safe = (
                load_safetensors(definition, wl, dataset_root)
                if any(d.type == "safetensors" for d in wl.inputs.values())
                else {}
            )
            # Fresh input set per trial (matches contest's between-trial freshness).
            trials_inp = [
                gen_inputs(definition, wl, device=device, safe_tensors=safe)
                for _ in range(cfg.num_trials)
            ]
            ref_outs = []
            for inp in trials_inp:
                with torch.no_grad():
                    r = ref_runnable(*inp)
                torch.cuda.synchronize(device)
                ref_outs.append(
                    normalize_outputs(
                        r, device=torch.device(device),
                        output_names=output_names, output_dtypes=output_dtypes,
                    )
                )
        except Exception as e:
            all_pass = False
            error_log = traceback.format_exc()
            print(f"{tag}: REFERENCE/LOAD ERROR -> {type(e).__name__}: {e}")
            per_workload.append({
                "uuid": wl.uuid,
                "status": "REF_ERROR",
                "speedup": None,
                "error_log": error_log,
            })
            continue

        # ---- correctness ----
        status = "PASSED"
        max_abs = max_rel = 0.0
        detail = ""
        for inp, ref_out in zip(trials_inp, ref_outs):
            try:
                with torch.no_grad():
                    res = run_fn(*inp)
                torch.cuda.synchronize(device)
                sol_out = normalize_outputs(
                    res, device=torch.device(device),
                    output_names=output_names, output_dtypes=output_dtypes,
                )
            except Exception as e:
                status = "RUNTIME_ERROR"
                detail = f"{type(e).__name__}: {e}"
                if args.verbose:
                    traceback.print_exc()
                break

            for name in output_names:
                s, rt = sol_out[name], ref_out[name]
                status, detail, a, r = _compare_output(name, s, rt, cfg)
                max_abs, max_rel = max(max_abs, a), max(max_rel, r)
                if status != "PASSED":
                    break
            if status != "PASSED":
                break

        if status != "PASSED":
            all_pass = False
            print(f"{tag}: {status}  {detail}")
            per_workload.append({"uuid": wl.uuid, "status": status, "speedup": None,
                                 "max_abs": max_abs, "max_rel": max_rel,
                                 "error_log": detail})
            continue

        if args.correctness_only:
            print(f"{tag}: PASSED  abs={max_abs:.2e} rel={max_rel:.2e}")
            per_workload.append({"uuid": wl.uuid, "status": "PASSED", "speedup": None,
                                 "max_abs": max_abs, "max_rel": max_rel})
            continue

        # ---- speed (only reached when correct) ----
        if ref_cache is not None and str(wl.uuid) in ref_cache:
            ref_ms = float(ref_cache[str(wl.uuid)])
            print(f"{tag}: using cached ref={ref_ms:.4f}ms")
        else:
            ref_ms = sum(time_runnable(ref_runnable, inp, cfg.warmup_runs, cfg.iterations, device)
                         for inp in trials_inp) / len(trials_inp)
        sol_ms = sum(time_runnable(run_fn, inp, cfg.warmup_runs, cfg.iterations, device)
                     for inp in trials_inp) / len(trials_inp)
        speedup = ref_ms / sol_ms if sol_ms > 0 else float("inf")
        print(f"{tag}: PASSED  ref={ref_ms:.4f}ms sol={sol_ms:.4f}ms  "
              f"speedup={speedup:.2f}x  abs={max_abs:.2e} rel={max_rel:.2e}")
        per_workload.append({"uuid": wl.uuid, "status": "PASSED", "speedup": speedup,
                             "ref_ms": ref_ms, "sol_ms": sol_ms,
                             "max_abs": max_abs, "max_rel": max_rel,
                             "axes": dict(wl.axes)})

    # ---- aggregate ----
    speedups = [w["speedup"] for w in per_workload
                if w["status"] == "PASSED" and w.get("speedup") is not None]
    passed = sum(1 for w in per_workload if w["status"] == "PASSED")
    geo = _geomean(speedups) if speedups else float("nan")
    arith = (sum(speedups) / len(speedups)) if speedups else float("nan")

    print("=" * 78)
    print(f"Passed: {passed}/{len(per_workload)}    valid_run={all_pass}")
    if not args.correctness_only and speedups:
        print(f"GEOMEAN speedup (optimization target): {_fmt(geo).strip()}")
        print(f"  arithmetic mean (fib/AKO4X metric):  {_fmt(arith).strip()}")
        print(f"  min={_fmt(min(speedups)).strip()}  max={_fmt(max(speedups)).strip()}")
    if not all_pass:
        print("RESULT: INVALID — not all workloads passed correctness "
              "(geomean above is over the PASSED subset only).")
    elif not args.correctness_only:
        print(f"RESULT: VALID — geomean speedup = {_fmt(geo).strip()}")
    else:
        print("RESULT: VALID — all workloads correct.")
    print("=" * 78)

    summary = {
        "definition": definition.name,
        "op_type": definition.op_type,
        "valid": all_pass,
        "passed": passed,
        "total": len(per_workload),
        "geomean_speedup": geo if speedups else None,
        "arithmetic_mean_speedup": arith if speedups else None,
        "tolerance": {"atol": cfg.atol, "rtol": cfg.rtol,
                      "required_matched_ratio": cfg.required_matched_ratio},
        "per_workload": per_workload,
    }
    if args.json:
        Path(args.json).write_text(json.dumps(summary, indent=2))
        print(f"Wrote {args.json}")
    return summary


def evaluate(args) -> dict:
    """Evaluate while temporarily releasing the runner's GPU memory holder."""
    physical_gpu = physical_gpu_from_environment()
    gpus = [physical_gpu] if physical_gpu is not None else []
    with gpu_occupancy_lease(gpus):
        return _evaluate(args)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--definition", default="task/definition.json")
    p.add_argument("--workload", default="task/workload.jsonl")
    p.add_argument("--solution", default="solution/solution.py")
    p.add_argument("--entry", default="run", help="entry function name in the solution file")
    p.add_argument(
        "--workload-index",
        default=None,
        help="comma-separated zero-based workload indices to evaluate, e.g. 1,2,3",
    )
    p.add_argument("--dataset-root", default=None,
                   help="contest root for resolving ./blob/... safetensors "
                        "(defaults to $FIB_DATASET_PATH / $AKO_DATASET_PATH)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--limit", type=int, default=0, help="evaluate only the first N workloads")
    p.add_argument("--atol", type=float, default=None)
    p.add_argument("--rtol", type=float, default=None)
    p.add_argument("--matched-ratio", type=float, default=None)
    p.add_argument("--correctness-only", action="store_true")
    p.add_argument("--json", default=None, help="write a JSON summary to this path")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--ref-cache", default=None, metavar="JSON",
                   help="inject cached reference latencies (ms) keyed by workload uuid; "
                        "skips live reference timing (2026-09-23, gdn_prefill terminal eval)")
    args = p.parse_args()

    summary = evaluate(args)
    # Non-zero exit when the run is invalid, so CI / agents can gate on it.
    sys.exit(0 if summary["valid"] else 1)


if __name__ == "__main__":
    main()
