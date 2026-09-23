#!/usr/bin/env python3
"""Evaluate one SOL-ExecBench solution using explicit file paths.

Unlike ``run_dataset.py``, this script does not discover problems from a dataset
directory and does not search for a solution filename inside a problem
directory.  The definition, workload, and solution paths are supplied
independently.

Examples
--------
Evaluate a Python source file with the SOL-ExecBench project environment::

    uv run --project /path/to/SOL-ExecBench --no-sync python \
        /path/to/test/scripts/evaluate_sol.py \
        --definition /path/to/definition.json \
        --workload /path/to/workload.jsonl \
        --solution /path/to/solution.py \
        -o /path/to/performance.json \
        --warmup 3 --iterations 50

Evaluate a Python source file without output json file:

    uv run --project /path/to/SOL-ExecBench --no-sync python \
        /path/to/test/scripts/evaluate_sol.py \
        --definition /path/to/definition.json \
        --workload /path/to/workload.jsonl \
        --solution /path/to/solution.py \
        --warmup 3 --iterations 1
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_reference_cache(path: Path | None, cache_key: str | None) -> dict[str, float]:
    if path is None or cache_key is None or not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if payload.get("cache_key") != cache_key:
        return {}
    values = payload.get("reference_latency_ms") or {}
    return {
        str(workload_id): float(latency)
        for workload_id, latency in values.items()
        if isinstance(latency, (int, float)) and float(latency) > 0
    }


def _write_reference_cache(
    path: Path,
    cache_key: str,
    definition_path: Path,
    values: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "kda-sol-reference-cache-v1",
        "cache_key": cache_key,
        "definition_sha256": _sha256(definition_path),
        "reference_latency_ms": values,
    }
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_path, path)


def _inject_cached_references(traces: list[dict], values: dict[str, float]) -> None:
    for trace in traces:
        workload_id = str((trace.get("workload") or {}).get("uuid") or "")
        evaluation = trace.get("evaluation") or {}
        performance = evaluation.get("performance") or None
        reference_latency = values.get(workload_id)
        if not isinstance(performance, dict) or reference_latency is None:
            continue
        latency = performance.get("latency_ms")
        if (
            evaluation.get("status") == "PASSED"
            and isinstance(latency, (int, float))
            and float(latency) > 0
        ):
            performance["reference_latency_ms"] = reference_latency
            performance["speedup_factor"] = reference_latency / float(latency)


def _existing_file(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"file not found: {path}")
    return path


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return number


def _nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("value must be zero or greater")
    return number


def _resolve_performance_path(output: Path | None) -> Path | None:
    """Resolve ``-o`` as either a JSON file path or an output directory."""
    if output is None:
        return None

    path = output.expanduser()
    if path.exists() and path.is_dir():
        path = path / "performance.json"
    elif path.suffix.lower() != ".json":
        path = path / "performance.json"

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _infer_dps(code: str, definition: dict) -> bool:
    """Infer destination-passing style from the candidate ``run()`` signature."""
    output_names = list(definition.get("outputs", {}).keys())
    if not output_names:
        return False

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False

    last_output = output_names[-1]
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run":
            if node.args.args:
                return node.args.args[-1].arg == last_output
            break
    return False


def build_custom_solution(definition: dict, solution_path: Path) -> dict:
    """Wrap an arbitrary Python source file as a SOL ``Solution`` dictionary."""
    code = solution_path.read_text(encoding="utf-8")
    code = code.replace("stream", "strm")
    name = definition["name"]
    uses_triton = "import triton" in code or "from triton" in code

    return {
        "name": f"custom_{name}",
        "definition": name,
        "author": "evaluate_sol",
        "description": f"Custom solution from {solution_path}.",
        "spec": {
            "languages": ["triton" if uses_triton else "pytorch"],
            "target_hardware": ["LOCAL"],
            "entry_point": "solution.py::run",
            "dependencies": ["torch", "triton"] if uses_triton else ["torch"],
            "destination_passing_style": _infer_dps(code, definition),
        },
        "sources": [
            {
                "path": "solution.py",
                "content": code,
            }
        ],
    }


def load_solution(definition: dict, solution_path: Path) -> dict:
    """Load a SOL solution JSON or wrap a Python source file."""
    if solution_path.suffix.lower() == ".json":
        solution = json.loads(solution_path.read_text(encoding="utf-8"))
        if not isinstance(solution, dict):
            raise ValueError(f"solution JSON must contain an object: {solution_path}")
        return solution
    if solution_path.suffix.lower() == ".py":
        return build_custom_solution(definition, solution_path)
    raise ValueError(
        f"unsupported solution file type {solution_path.suffix!r}; expected .py or .json"
    )


def run_cli(
    definition_path: Path,
    workload_path: Path,
    solution_path: Path,
    timeout: int,
    config_path: Path | None = None,
    keep_staging: bool = False,
    verbose: bool = False,
) -> list[dict]:
    """Invoke ``sol-execbench`` and return parsed trace dictionaries."""
    cli_path = Path(sys.executable).parent / "sol-execbench"
    if not cli_path.is_file():
        raise FileNotFoundError(
            f"sol-execbench executable not found next to Python: {cli_path}. "
            "Run this script with the SOL-ExecBench virtual environment."
        )

    command = [
        str(cli_path),
        "--definition",
        str(definition_path),
        "--workload",
        str(workload_path),
        "--solution",
        str(solution_path),
        "--timeout",
        str(timeout),
        "--json",
    ]
    if config_path is not None:
        command.extend(["--config", str(config_path)])
    if keep_staging:
        command.append("--keep-staging")
    if verbose:
        command.append("--verbose")

    python_bin_dir = str(Path(sys.executable).parent)
    current_path = os.environ.get("PATH", "")
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout + 60,
        env={
            **os.environ,
            "PATH": f"{python_bin_dir}:{current_path}",
        },
    )

    traces = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            traces.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    if not traces:
        print(
            "SOL CLI produced no traces.\n"
            f"exit code: {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}",
            file=sys.stderr,
        )
    return traces


def _performance_tolerance(workload_records: list[dict]) -> dict:
    """Convert SOL tolerance names to the existing performance JSON schema."""
    tolerance = workload_records[0].get("tolerance") or {}
    return {
        "atol": tolerance.get("max_atol", 1e-2),
        "rtol": tolerance.get("max_rtol", 1e-2),
        "required_matched_ratio": tolerance.get("required_matched_ratio", 0.99),
    }


def build_performance(
    traces: list[dict],
    definition: dict,
    workload_records: list[dict],
    correctness_only: bool = False,
) -> dict:
    """Convert SOL trace dictionaries to the MTMC performance JSON schema."""
    passed = 0
    speedups = []
    per_workload = []

    traces_by_uuid = {
        (trace.get("workload") or {}).get("uuid"): trace
        for trace in traces
        if (trace.get("workload") or {}).get("uuid") is not None
    }

    for workload_record in workload_records:
        workload_id = workload_record.get("uuid", "unknown")
        trace = traces_by_uuid.get(workload_id)
        if trace is None:
            per_workload.append(
                {
                    "uuid": workload_id,
                    "status": "RUNTIME_ERROR",
                    "speedup": None,
                    "ref_ms": None,
                    "sol_ms": None,
                    "max_abs": 0.0,
                    "max_rel": 0.0,
                    "axes": workload_record.get("axes") or {},
                }
            )
            continue

        workload = trace.get("workload") or workload_record
        evaluation = trace.get("evaluation") or {}
        status = evaluation.get("status", "UNKNOWN")
        correctness = evaluation.get("correctness") or {}
        performance = evaluation.get("performance") or {}
        speedup = None if correctness_only else performance.get("speedup_factor")
        if speedup is not None and speedup <= 0:
            speedup = None

        workload_performance = {
            "uuid": workload_id,
            "status": status,
            "speedup": speedup,
            "ref_ms": None if correctness_only else performance.get("reference_latency_ms"),
            "sol_ms": None if correctness_only else performance.get("latency_ms"),
            "max_abs": correctness.get("max_absolute_error", 0.0),
            "max_rel": correctness.get("max_relative_error", 0.0),
            "axes": workload.get("axes") or {},
        }

        if status == "PASSED":
            passed += 1
            if speedup is not None:
                speedups.append(speedup)
        per_workload.append(workload_performance)

    geomean_speedup = None
    arithmetic_mean_speedup = None
    if speedups and len(speedups) == passed:
        geomean_speedup = math.exp(sum(math.log(value) for value in speedups) / len(speedups))
        arithmetic_mean_speedup = sum(speedups) / len(speedups)

    return {
        "definition": definition["name"],
        "op_type": definition.get("op_type"),
        "valid": bool(workload_records) and passed == len(workload_records),
        "passed": passed,
        "total": len(workload_records),
        "geomean_speedup": geomean_speedup,
        "arithmetic_mean_speedup": arithmetic_mean_speedup,
        "tolerance": _performance_tolerance(workload_records),
        "per_workload": per_workload,
    }


def print_performance(performance: dict, correctness_only: bool = False) -> None:
    """Print a compact view of an MTMC performance result."""
    status = "OK" if performance["valid"] else "FAIL"
    print("\n" + "=" * 78)
    print(f"Problem : {performance['definition']}")
    print(f"Passed  : {performance['passed']}/{performance['total']}")
    print(f"Status  : {status}")
    for index, workload in enumerate(performance["per_workload"], 1):
        if workload["status"] != "PASSED":
            print(f"[{index}] {workload['uuid']}: {workload['status']}")
            continue
        if workload["speedup"] is None or workload["speedup"] <= 0:
            message = (
                "correctness only; timing skipped"
                if correctness_only
                else "reference speedup unavailable"
            )
            print(f"[{index}] {workload['uuid']}: {message}")
            continue
        print(
            f"[{index}] {workload['uuid']}: "
            f"reference={workload['ref_ms']:.6f} ms, "
            f"candidate={workload['sol_ms']:.6f} ms, "
            f"speedup={workload['speedup']:.4f}x"
        )
    if performance["geomean_speedup"] is not None:
        print(f"Geomean : {performance['geomean_speedup']:.4f}x")
        print(f"Average : {performance['arithmetic_mean_speedup']:.4f}x")
    elif correctness_only:
        print("Speedup : unavailable (correctness-only mode)")
    print("=" * 78)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate one SOL-ExecBench solution using explicit file paths."
    )
    parser.add_argument("--definition", required=True, type=_existing_file)
    parser.add_argument("--workload", required=True, type=_existing_file)
    parser.add_argument(
        "--solution",
        required=True,
        type=_existing_file,
        help="Path to a solution.py or SOL solution.json file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "Output .json file path or directory for performance.json; "
            "omit to avoid writing a JSON file."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=_positive_int,
        default=10800,
        help="GPU evaluation timeout in seconds (default: 10800).",
    )
    parser.add_argument(
        "--max-workloads",
        type=_positive_int,
        default=None,
        help="Evaluate only the first N non-empty workload records.",
    )
    parser.add_argument(
        "--iterations",
        type=_positive_int,
        default=None,
        help="Number of timing iterations per workload.",
    )
    parser.add_argument(
        "--warmup",
        type=_nonnegative_int,
        default=3,
        help="Number of untimed warmup iterations per workload (default: 3).",
    )
    parser.add_argument(
        "--seed",
        type=_nonnegative_int,
        default=200,
        help="Official SOL input-generation seed (default: 200).",
    )
    parser.add_argument(
        "--reference-cache",
        type=Path,
        default=None,
        help="Optional feedback-only reference latency cache file.",
    )
    parser.add_argument(
        "--reference-cache-key",
        default=None,
        help="Hardware/software fingerprint required with --reference-cache.",
    )
    parser.add_argument(
        "--correctness-only",
        action="store_true",
        help="Run correctness validation without latency benchmarking.",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Re-evaluate even when an existing result already passed.",
    )
    parser.add_argument(
        "--keep-staging",
        action="store_true",
        help="Keep the temporary SOL evaluator staging directory.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Pass --verbose to the sol-execbench CLI.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if (args.reference_cache is None) != (args.reference_cache_key is None):
        raise ValueError("--reference-cache and --reference-cache-key must be used together")
    performance_path = _resolve_performance_path(args.output)

    definition = json.loads(args.definition.read_text(encoding="utf-8"))
    if not isinstance(definition, dict):
        raise ValueError(f"definition JSON must contain an object: {args.definition}")
    if not isinstance(definition.get("name"), str) or not definition["name"]:
        raise ValueError("definition.name must be a non-empty string")

    if (
        performance_path is not None
        and not args.rerun
        and performance_path.is_file()
    ):
        existing_performance = json.loads(performance_path.read_text(encoding="utf-8"))
        if existing_performance.get("valid") is True and not args.correctness_only:
            print("Skipping: existing result already passed. Use --rerun to evaluate again.")
            print_performance(existing_performance)
            return 0

    workload_lines = [
        line for line in args.workload.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not workload_lines:
        raise ValueError(f"workload file is empty: {args.workload}")
    selected_lines = workload_lines[: args.max_workloads] if args.max_workloads else workload_lines
    workload_records = [json.loads(line) for line in selected_lines]
    selected_uuids = [str(record.get("uuid") or "") for record in workload_records]
    cached_references = _load_reference_cache(
        args.reference_cache, args.reference_cache_key
    )
    use_cached_references = bool(selected_uuids) and all(
        workload_id in cached_references for workload_id in selected_uuids
    )

    with tempfile.TemporaryDirectory(prefix="sol_evaluate_sol_") as temporary_dir:
        temporary_root = Path(temporary_dir)
        workload_path = temporary_root / "workload.jsonl"
        workload_path.write_text("\n".join(selected_lines) + "\n", encoding="utf-8")

        solution = load_solution(definition, args.solution)
        packaged_solution_path = temporary_root / "solution.json"
        packaged_solution_path.write_text(
            json.dumps(solution, indent=2), encoding="utf-8"
        )

        config = {
            "benchmark_reference": not args.correctness_only and not use_cached_references,
            "correctness_only": args.correctness_only,
            "warmup_runs": args.warmup,
            "seed": args.seed,
        }
        if args.iterations is not None:
            config["iterations"] = args.iterations
        config_path = temporary_root / "config.json"
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

        print(f"Definition: {args.definition}")
        print(f"Workload  : {args.workload}")
        print(f"Solution  : {args.solution}")
        if performance_path is not None:
            print(f"Output    : {performance_path}")
        else:
            print("Output    : disabled")

        traces = run_cli(
            definition_path=args.definition,
            workload_path=workload_path,
            solution_path=packaged_solution_path,
            timeout=args.timeout,
            config_path=config_path,
            keep_staging=args.keep_staging,
            verbose=args.verbose,
        )

    if use_cached_references:
        _inject_cached_references(traces, cached_references)
    elif args.reference_cache is not None and not args.correctness_only:
        updated_references = dict(cached_references)
        for trace in traces:
            workload_id = str((trace.get("workload") or {}).get("uuid") or "")
            performance = (trace.get("evaluation") or {}).get("performance") or {}
            reference_latency = performance.get("reference_latency_ms")
            if (
                workload_id
                and isinstance(reference_latency, (int, float))
                and float(reference_latency) > 0
            ):
                updated_references[workload_id] = float(reference_latency)
        if updated_references:
            _write_reference_cache(
                args.reference_cache,
                args.reference_cache_key,
                args.definition,
                updated_references,
            )

    performance = build_performance(
        traces,
        definition,
        workload_records,
        correctness_only=args.correctness_only,
    )
    performance["evaluation_config"] = {
        "warmup_runs": args.warmup,
        "iterations": args.iterations,
        "seed": args.seed,
        "benchmark_reference": not args.correctness_only,
        "reference_latency_cached": use_cached_references,
        "timeout_seconds": args.timeout,
    }
    print_performance(performance, correctness_only=args.correctness_only)
    if performance_path is not None:
        performance_path.write_text(json.dumps(performance, indent=2), encoding="utf-8")
        print(f"Performance: {performance_path}")
    return 0 if performance["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
