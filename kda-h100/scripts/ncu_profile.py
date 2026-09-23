#!/usr/bin/env python3
"""Card-coordinated ncu profiling launcher for KDA task workspaces.

The agent-facing entry is scripts/ncu_profile.sh inside each task workspace.
This controller script performs the coordination the agent is not allowed to
do itself (no direct nvidia-smi / GPU management in the workspace contract):

  1. Pick a GPU from KDA_ALLOWED_GPUS that is currently usable:
     not leased by an evaluation, not used by another profiling run,
     utilization 0%, and memory either idle or held by our own occupancy
     holder (which will be released for the profiling window).
  2. Stop the occupancy holder on the chosen card for the profiling window
     and restart it afterwards (same lease mechanism as evaluations), so
     profiling harnesses get the card's full VRAM.
  3. Point CUDA_VISIBLE_DEVICES at the card and prepend the benchmark's
     evaluation Python (torch/triton capable) plus CUDA_HOME/bin to PATH,
     so `python harness.py` works without hardcoding interpreter paths.
  4. Run ncu with the caller's arguments verbatim inside the workspace.

Exit codes: 0 success; 2 KDA_ALLOWED_GPUS missing/invalid; 3 no usable GPU.

Usage (from a task workspace):
    ./scripts/ncu_profile.sh --set basic -o profile/r1 python harness.py
    ./scripts/ncu_profile.sh --status        # only print GPU selection
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

CONTROLLER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CONTROLLER_DIR))
import evaluate_candidate as ec  # reuse the trusted GPU/occupancy helpers


def profile_lock_path(hostname: str, gpu_index: int) -> Path:
    return ec.GPU_LOCK_ROOT / hostname / f"gpu{gpu_index}.profile.lock"


def try_profile_lock(hostname: str, gpu_index: int):
    """Non-blocking exclusive lock marking a card as under profiling."""
    path = profile_lock_path(hostname, gpu_index)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        return None
    return lock


def release_profile_lock(lock) -> None:
    fcntl.flock(lock, fcntl.LOCK_UN)
    lock.close()


def evaluator_lease_taken(hostname: str, gpu_index: int) -> bool:
    """True while an evaluation holds the card lease (timing in progress)."""
    lock = ec.try_gpu_lock(hostname, gpu_index)
    if lock is None:
        return True
    fcntl.flock(lock, fcntl.LOCK_UN)
    lock.close()
    return False


def select_gpu(allowed: list[int]) -> tuple[int | None, object, list[dict]]:
    """Return (selected index, held profile lock, per-card decision report)."""
    hostname = socket.gethostname()
    snapshots = {gpu["index"]: gpu for gpu in ec.query_gpus()}
    report: list[dict] = []
    for index in allowed:
        gpu = snapshots.get(index)
        if gpu is None:
            report.append({"gpu": index, "reason": "not present or not in nvidia-smi output"})
            continue
        entry = {
            "gpu": index,
            "memory_used_mib": gpu["memory_used_mib"],
            "utilization_percent": gpu["utilization_percent"],
            "occupancy_held": ec.occupancy_holder_running(index),
        }
        lock = try_profile_lock(hostname, index)
        if lock is None:
            entry["reason"] = "another profiling run is active on this card"
            report.append(entry)
            continue
        if evaluator_lease_taken(hostname, index):
            entry["reason"] = "evaluation lease active (never disturb timing)"
            report.append(entry)
            release_profile_lock(lock)
            continue
        if gpu["utilization_percent"] > 0:
            entry["reason"] = "GPU busy (utilization > 0)"
            report.append(entry)
            release_profile_lock(lock)
            continue
        if gpu["memory_used_mib"] >= ec.GPU_MEMORY_THRESHOLD_MIB and not entry["occupancy_held"]:
            entry["reason"] = "foreign memory resident (not our holder)"
            report.append(entry)
            release_profile_lock(lock)
            continue
        entry["reason"] = "selected"
        report.append(entry)
        return index, lock, report
    return None, None, report


def benchmark_python_bin(workspace: Path) -> Path:
    """Directory of the torch/triton-capable Python for this task's benchmark."""
    control = ec.CONTROL_ROOT / workspace.name
    task_path = control / "task.json"
    benchmark = ""
    if task_path.is_file():
        benchmark = str(json.loads(task_path.read_text(encoding="utf-8")).get("benchmark") or "")
    if benchmark == "sol_execbench":
        return ec.SOL_PYTHON.parent
    return ec.FLASHINFER_PYTHON.parent


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--status", action="store_true",
                        help="only print the GPU selection decision; do not run ncu")
    parser.add_argument("--workspace", type=Path,
                        default=Path(os.environ.get("KDA_WORKSPACE") or Path.cwd()))
    args, ncu_args = parser.parse_known_args()

    allowed_raw = os.environ.get("KDA_ALLOWED_GPUS", "").strip()
    if not allowed_raw:
        print("[ncu-profile] KDA_ALLOWED_GPUS is not set; profiling requires the GPU whitelist",
              file=sys.stderr)
        return 2
    try:
        allowed = [int(value.strip()) for value in allowed_raw.split(",") if value.strip()]
    except ValueError:
        print(f"[ncu-profile] invalid KDA_ALLOWED_GPUS={allowed_raw!r}", file=sys.stderr)
        return 2
    if not allowed:
        print("[ncu-profile] KDA_ALLOWED_GPUS is empty", file=sys.stderr)
        return 2

    index, lock, report = select_gpu(allowed)
    for entry in report:
        print(f"[ncu-profile] gpu{entry['gpu']}: {entry['reason']}"
              + (f" (mem={entry.get('memory_used_mib')}MiB util={entry.get('utilization_percent')}%"
                 f" occupancy_held={entry.get('occupancy_held')})"
                 if "memory_used_mib" in entry else ""), flush=True)

    if index is None or lock is None:
        print("[ncu-profile] no usable GPU in the whitelist right now; "
              "retry later or finish the running evaluation first", file=sys.stderr)
        return 3

    if args.status or not ncu_args:
        release_profile_lock(lock)
        if not args.status and not ncu_args:
            print("[ncu-profile] nothing to run: pass ncu arguments "
                  "(e.g. ./scripts/ncu_profile.sh --set basic python harness.py)",
                  file=sys.stderr)
            return 2
        return 0

    ncu_bin = shutil.which("ncu") or str(ec.CUDA_HOME / "bin" / "ncu")
    if not Path(ncu_bin).is_file():
        print(f"[ncu-profile] ncu binary not found (looked in PATH and {ec.CUDA_HOME / 'bin'})",
              file=sys.stderr)
        release_profile_lock(lock)
        return 2

    python_bin_dir = benchmark_python_bin(args.workspace)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(index)
    env["PATH"] = f"{python_bin_dir}:{ec.CUDA_HOME / 'bin'}:{env.get('PATH', '')}"
    env.setdefault("TORCH_CUDA_ARCH_LIST", "9.0")

    print(f"[ncu-profile] gpu={index} | occupancy lease engaging (holder yields during "
          f"profiling) | python bin: {python_bin_dir} | ncu: {ncu_bin}", flush=True)
    try:
        with ec.occupancy_lease(index):
            completed = subprocess.run(
                [ncu_bin, *ncu_args], env=env, cwd=str(args.workspace)
            )
    finally:
        release_profile_lock(lock)
    print(f"[ncu-profile] ncu exited rc={completed.returncode}; occupancy holder re-acquired",
          flush=True)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
