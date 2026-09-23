#!/usr/bin/env python3
"""Trusted FlashInfer and SOL evaluator for isolated KDA task workspaces."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import io
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import tokenize
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path


ROOT = Path("/home/ziming/kda-ops")
CONTROL_ROOT = ROOT / "kda-control"
RUN_ROOT = ROOT / "kda-runs"
FLASHINFER_DATASET = Path("/home/ziming/dataset/flashinfer-test")
FLASHINFER_EVALUATOR = ROOT / "evaluators/evaluate.py"
FLASHINFER_PYTHON = Path("/home/ziming/miniconda3/envs/fib/bin/python")
SOL_ROOT = Path("/home/ziming/dataset/SOL-ExecBench")
SOL_DATASET = SOL_ROOT / "data/benchmark"
SOL_EVALUATOR = ROOT / "evaluators/evaluate_sol.py"
SOL_PYTHON = SOL_ROOT / ".venv/bin/python"
SOL_CLI = SOL_ROOT / ".venv/bin/sol-execbench"
CUDA_HOME = Path("/usr/local/cuda-12.8")
GPU_LOCK_ROOT = CONTROL_ROOT / "gpu-locks"
GPU_MEMORY_THRESHOLD_MIB = 200
GPU_UTILIZATION_THRESHOLD_PERCENT = 0
GPU_WAIT_TIMEOUT_SECONDS = 1800
GPU_POLL_INTERVAL_SECONDS = 10
GPU_MONITOR_INTERVAL_SECONDS = 0.1
# GPU 占卡系统（脚本位于 /home/ziming/MTMC-baseline/agent-generation/scripts/，
# 用法说明见 ~/whr/notes/gpu-occupancy-and-eval-hook.md）：
# 在两次评测的间隙，由 fill_vram.py 进程持有空闲卡的显存，防止其他用户抢占；
# 真正评测时本控制器先停掉持有器、让出整张卡，评测结束后再自动重新拉起持有器。
# 本控制器只调用该目录下 gpu_occupancy 模块的公开接口，不修改评测器与占卡脚本本身。
OCCUPANCY_SCRIPTS_DIR = Path("/home/ziming/MTMC-baseline/agent-generation/scripts")
# 持有器让位后等待显存释放的参数（fill_vram 逐块 free 73GB 需数十秒）
OCCUPANCY_RELEASE_TIMEOUT_SECONDS = 120
OCCUPANCY_RELEASED_THRESHOLD_MIB = 1024

try:
    sys.path.insert(0, str(OCCUPANCY_SCRIPTS_DIR))
    import gpu_occupancy
except Exception:
    # 占卡模块加载失败时（例如目录被移动），退回原始行为：
    # 只租约完全空闲的卡，不做任何占卡协调。
    gpu_occupancy = None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def validate_candidate_id(value: str) -> str:
    if not re.fullmatch(r"c\d{3}", value):
        raise ValueError("candidate must match cNNN, for example c001")
    return value


def static_check(source: Path) -> list[str]:
    text = source.read_text(encoding="utf-8")
    code = tokenize.untokenize(
        token
        for token in tokenize.generate_tokens(io.StringIO(text).readline)
        if token.type not in {tokenize.COMMENT, tokenize.STRING}
    )
    errors = []
    if "import triton" not in code or "@triton.jit" not in code:
        errors.append("submission must contain a Triton JIT kernel")
    forbidden = [
        (r"torch\.compile\s*\(", "torch.compile is forbidden"),
        (r"\.cpu\s*\(", "CPU fallback is forbidden"),
        (r"\.numpy\s*\(", "NumPy fallback is forbidden"),
        (r"import\s+numpy|from\s+numpy", "NumPy is forbidden"),
        (r"except[^:]*:\s*(?:\n\s*)+return\s+torch\.", "Torch exception fallback is forbidden"),
    ]
    for pattern, message in forbidden:
        if re.search(pattern, code, re.MULTILINE):
            errors.append(message)
    computational = re.findall(
        r"torch\.(?:matmul|mm|bmm|einsum|sum|mean|rsqrt|sqrt|pow|norm|softmax|topk|sort|conv\w*|linear)\s*\(",
        code,
    )
    if computational:
        errors.append("computational torch operators are forbidden in the submitted path")
    return errors


def query_gpus() -> list[dict]:
    allowed_raw = os.environ.get("KDA_ALLOWED_GPUS", "").strip()
    allowed = None
    if allowed_raw:
        try:
            allowed = {int(value.strip()) for value in allowed_raw.split(",") if value.strip()}
        except ValueError as error:
            raise RuntimeError(f"invalid KDA_ALLOWED_GPUS={allowed_raw!r}") from error
        if not allowed:
            raise RuntimeError("KDA_ALLOWED_GPUS did not contain any GPU indices")
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    gpus = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",", 4)]
        if len(fields) != 5:
            raise RuntimeError(f"unexpected nvidia-smi output: {line}")
        index, uuid, name, memory_used_mib, utilization_percent = fields
        if allowed is not None and int(index) not in allowed:
            continue
        gpus.append({
            "index": int(index),
            "uuid": uuid,
            "name": name,
            "memory_used_mib": int(memory_used_mib),
            "utilization_percent": int(utilization_percent),
        })
    return sorted(gpus, key=lambda gpu: gpu["index"])


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def occupancy_coordination_path(hostname: str, gpu_index: int) -> Path:
    """共享租约状态文件（计数式占卡让位）的路径。"""
    return GPU_LOCK_ROOT / hostname / f"gpu{gpu_index}.occupancy-coordination.json"


def coordination_lock_path(hostname: str, gpu_index: int) -> Path:
    return GPU_LOCK_ROOT / hostname / f"gpu{gpu_index}.occ-coord.lock"


@contextmanager
def coordination_flock(hostname: str, gpu_index: int):
    """保护共享租约状态文件变更的短临界区互斥锁。"""
    path = coordination_lock_path(hostname, gpu_index)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def load_coordination_state(path: Path) -> dict:
    """读取共享租约状态，剔除已死亡进程的计数（防止击杀泄漏）。"""
    state = {"holders": [], "holder_was_running": False}
    if path.is_file():
        try:
            state.update(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            pass
    holders = [
        entry for entry in (state.get("holders") or [])
        if isinstance(entry, dict) and _pid_alive(int(entry.get("pid", 0)))
    ]
    state["holders"] = holders
    return state


def write_coordination_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def occupancy_holder_running(gpu_index: int) -> bool:
    """判断指定显卡当前是否被占卡系统的显存持有器（fill_vram.py）占用。

    判定方式与占卡模块内部 _is_manager_process 一致：读取占卡状态目录中的
    gpu-<卡号>.pid 文件，确认该 pid 仍然存活、其命令行确实是 fill_vram.py、
    且 --device 参数指向这张卡。

    注意：只认可由管理器（gpu_occupancy.py start）启动的持有器；用
    fill_gpus.sh 等方式手动启动的持有器没有 pid 记录，本函数无法识别，
    其占用的卡也不会被判为可租约——这是占卡笔记 §5 明确的约定。
    """
    if gpu_occupancy is None:
        return False
    try:
        pid = int(
            (gpu_occupancy.state_dir() / f"gpu-{gpu_index}.pid").read_text().strip()
        )
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    except (FileNotFoundError, ValueError, OSError, UnicodeDecodeError):
        return False
    return "fill_vram.py" in command and f"--device {gpu_index}" in command


def gpu_is_empty(gpu: dict) -> bool:
    """判断一张卡当前是否可被评测租约。

    两种情形视为可租约：
    1. 利用率为 0，且显存占用低于阈值（完全空闲）——原始判定，保持不变；
    2. 利用率为 0、显存占用高，但占用进程是我们自己的显存持有器——
       这种卡会在评测开始前被 occupancy_lease 临时释放，因此仍然可用。

    除持有器以外的任何占用（其他用户的进程、残留进程等）都视为不可租约。
    """
    if gpu["utilization_percent"] > GPU_UTILIZATION_THRESHOLD_PERCENT:
        return False
    if gpu["memory_used_mib"] < GPU_MEMORY_THRESHOLD_MIB:
        return True
    return occupancy_holder_running(gpu["index"])


@contextmanager
def occupancy_lease(gpu_index: int):
    """在评测/剖析计时期间让指定卡上的显存持有器停止（跨进程共享、计数式）。

    与旧的"每次评测停持有器、评完立即占回"不同，本实现是批量让位语义，
    对应"队列忙时保持让位、队列空了再占回"的占卡策略：

    - 第一个进入的租约持有者（跨所有并发评测/剖析进程）停掉该卡的持有器；
    - 后续租约只增加计数，不再碰持有器——连续的评测之间不做停/启切换；
    - 最后一个租约退出时只登记空闲时间戳，由常驻守护进程 holder_keeper 在
      空闲超过 KDA_OCCUPANCY_IDLE_SECONDS（默认 90 秒）后重新拉起持有器占回；
    - 租约状态记录在各持有进程的 pid：持有进程被杀（例如 timer 窗口击杀）时，
      守护进程会清理死 pid 的计数，不会因泄漏而让卡一直裸奔。

    评测正确性不受影响：持有器停着 = 显存全释放，外部进程监控期间持有器
    不在场；单任务场景下评测间隔通常数分钟 > 空闲阈值，持有器照常在评测
    间隙占回，防抢卡效果不变。
    """
    hostname = socket.gethostname()
    if gpu_occupancy is None:
        yield {"engaged": False, "reason": "occupancy module unavailable"}
        return
    os.environ.setdefault("GPU_OCCUPANCY_ENABLED", "1")
    os.environ.setdefault("GPU_OCCUPANCY_PYTHON", str(FLASHINFER_PYTHON))
    if occupancy_holder_running(gpu_index):
        os.environ["GPU_OCCUPANCY_ENABLED"] = "1"

    state_path = occupancy_coordination_path(hostname, gpu_index)
    holder_running_before = False
    with coordination_flock(hostname, gpu_index):
        state = load_coordination_state(state_path)
        holder_running_before = occupancy_holder_running(gpu_index)
        if holder_running_before and not state["holders"]:
            # 本进程是本批第一个租约：停持有器（gpu_occupancy 内部有互斥锁）
            with gpu_occupancy._gpu_lock(gpu_index):
                gpu_occupancy._stop_locked(gpu_index)
            state["holder_was_running"] = True
            # 等待显存真正释放：fill_vram 收到 SIGINT 后需数秒~数十秒逐块 free 73GB。
            # 不等的话评测进程进场即 CUDA OOM（16/16 RUNTIME_ERROR）或被外部进程
            # 监控判废（rc=3）——2026-09-23 L2/040 事故的根因。
            release_deadline = time.time() + OCCUPANCY_RELEASE_TIMEOUT_SECONDS
            while time.time() < release_deadline:
                current = next(
                    (g for g in query_gpus() if g["index"] == gpu_index), None
                )
                if current is None or current["memory_used_mib"] < OCCUPANCY_RELEASED_THRESHOLD_MIB:
                    break
                if occupancy_holder_running(gpu_index):
                    # 持有器没死透还活着：再补一刀并继续等
                    with gpu_occupancy._gpu_lock(gpu_index):
                        gpu_occupancy._stop_locked(gpu_index)
                time.sleep(2)
            else:
                raise TimeoutError(
                    f"occupancy holder on gpu{gpu_index} did not release VRAM within "
                    f"{OCCUPANCY_RELEASE_TIMEOUT_SECONDS}s; aborting evaluation to avoid "
                    f"contaminated timing"
                )
        state["holders"].append({"pid": os.getpid(), "since": now_iso()})
        state.pop("idle_since", None)
        write_coordination_state(state_path, state)

    try:
        yield {
            "engaged": gpu_occupancy.occupancy_enabled(),
            "holder_running_before": holder_running_before,
            "mode": "shared-count",
            "concurrent_leases": len(state["holders"]),
        }
    finally:
        with coordination_flock(hostname, gpu_index):
            state = load_coordination_state(state_path)
            state["holders"] = [
                entry for entry in state["holders"] if entry.get("pid") != os.getpid()
            ]
            if not state["holders"]:
                state["idle_since"] = now_iso()
            write_coordination_state(state_path, state)


def gpu_lock_path(hostname: str, gpu_index: int) -> Path:
    return GPU_LOCK_ROOT / hostname / f"gpu{gpu_index}.lock"


def try_gpu_lock(hostname: str, gpu_index: int):
    path = gpu_lock_path(hostname, gpu_index)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        return None
    return lock


def gpu_status() -> dict:
    hostname = socket.gethostname()
    entries = []
    for gpu in query_gpus():
        lock = try_gpu_lock(hostname, gpu["index"])
        controller_locked = lock is None
        if lock is not None:
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()
        entries.append({
            **gpu,
            "empty": gpu_is_empty(gpu),
            "controller_locked": controller_locked,
            "occupancy_held": occupancy_holder_running(gpu["index"]),
            "eligible": gpu_is_empty(gpu) and not controller_locked,
        })
    eligible = [gpu["index"] for gpu in entries if gpu["eligible"]]
    return {
        "hostname": hostname,
        "memory_threshold_mib": GPU_MEMORY_THRESHOLD_MIB,
        "utilization_threshold_percent": GPU_UTILIZATION_THRESHOLD_PERCENT,
        "selected_gpu": min(eligible) if eligible else None,
        "gpus": entries,
    }


def profile_lock_path(hostname: str, gpu_index: int) -> Path:
    return GPU_LOCK_ROOT / hostname / f"gpu{gpu_index}.profile.lock"


def profile_lock_held(hostname: str, gpu_index: int) -> bool:
    """该卡是否正被 ncu 剖析占用（剖析期间绝不同卡评测，规则见计划 §3.2.2）。"""
    path = profile_lock_path(hostname, gpu_index)
    if not path.is_file():
        return False
    lock = path.open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        return True
    fcntl.flock(lock, fcntl.LOCK_UN)
    lock.close()
    return False


@contextmanager
def lease_empty_gpu(timeout_seconds: int, poll_interval_seconds: int):
    hostname = socket.gethostname()
    started = time.monotonic()
    deadline = started + timeout_seconds
    while True:
        for gpu in query_gpus():
            if not gpu_is_empty(gpu):
                continue
            if profile_lock_held(hostname, gpu["index"]):
                continue
            lock = try_gpu_lock(hostname, gpu["index"])
            if lock is None:
                continue
            try:
                current = next(
                    candidate for candidate in query_gpus() if candidate["index"] == gpu["index"]
                )
                if not gpu_is_empty(current):
                    continue
                if profile_lock_held(hostname, gpu["index"]):
                    continue
                yield {
                    "hostname": hostname,
                    "physical_gpu": current["index"],
                    "gpu_uuid": current["uuid"],
                    "gpu_name": current["name"],
                    "preflight_memory_used_mib": current["memory_used_mib"],
                    "preflight_utilization_percent": current["utilization_percent"],
                    "wait_seconds": round(time.monotonic() - started, 3),
                    "lock_path": str(gpu_lock_path(hostname, current["index"])),
                }
                return
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
                lock.close()
        if time.monotonic() >= deadline:
            raise TimeoutError(f"no empty GPU became available within {timeout_seconds} seconds")
        time.sleep(min(poll_interval_seconds, max(0, deadline - time.monotonic())))


def query_compute_processes(gpu_uuid: str) -> list[dict]:
    command = [
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,used_gpu_memory,process_name",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    processes = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",", 3)]
        if len(fields) != 4 or fields[0] != gpu_uuid:
            continue
        processes.append({
            "pid": int(fields[1]),
            "used_memory_mib": int(fields[2]),
            "process_name": fields[3],
        })
    return processes


def parent_pid(pid: int) -> int | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("PPid:"):
                return int(line.split()[1])
    except (FileNotFoundError, PermissionError, ValueError):
        return None
    return None


def belongs_to_process_tree(pid: int, root_pid: int) -> bool:
    seen = set()
    while pid > 1 and pid not in seen:
        if pid == root_pid:
            return True
        seen.add(pid)
        next_pid = parent_pid(pid)
        if next_pid is None:
            return False
        pid = next_pid
    return False


def run_with_gpu_monitor(cmd: list[str], env: dict, output, gpu: dict) -> tuple[subprocess.CompletedProcess, dict]:
    process = subprocess.Popen(cmd, stdout=output, stderr=subprocess.STDOUT, env=env)
    monitor = {
        "interval_seconds": GPU_MONITOR_INTERVAL_SECONDS,
        "foreign_process_detected": False,
        "foreign_processes": [],
    }
    while process.poll() is None:
        foreign = [
            entry
            for entry in query_compute_processes(gpu["gpu_uuid"])
            if not belongs_to_process_tree(entry["pid"], process.pid)
        ]
        if foreign:
            monitor["foreign_process_detected"] = True
            monitor["foreign_processes"] = foreign
            monitor["detected_at"] = datetime.now().astimezone().isoformat()
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            output.write(
                "\n[kda-controller] timing invalidated: foreign GPU process detected: "
                + json.dumps(foreign, ensure_ascii=False)
                + "\n"
            )
            output.flush()
            return subprocess.CompletedProcess(cmd, returncode=3), monitor
        time.sleep(GPU_MONITOR_INTERVAL_SECONDS)
    return subprocess.CompletedProcess(cmd, returncode=process.returncode), monitor


def update_result_metadata(result_path: Path, metadata: dict) -> None:
    try:
        payload = read_json(result_path) if result_path.exists() else {}
    except json.JSONDecodeError:
        return
    if isinstance(payload, dict):
        payload["kda_controller"] = metadata
        write_json(result_path, payload)


def require_file(path: Path, expected_sha256: str, label: str) -> Path:
    if not path.is_file():
        raise SystemExit(f"missing {label}: {path}")
    actual = sha256(path)
    if actual != expected_sha256:
        raise SystemExit(f"{label} checksum mismatch: expected {expected_sha256}, got {actual}")
    return path


def trusted_runtime(task_config: dict, control: Path, stage: str) -> dict:
    benchmark = str(task_config.get("benchmark") or "")
    if benchmark == "flashinfer":
        evaluator = FLASHINFER_EVALUATOR
        python = FLASHINFER_PYTHON
        dataset = FLASHINFER_DATASET
    elif benchmark == "sol_execbench":
        evaluator = SOL_EVALUATOR
        python = SOL_PYTHON
        dataset = SOL_DATASET
    else:
        raise SystemExit(f"unsupported benchmark: {benchmark}")

    expected_evaluator_sha = task_config.get("evaluator_sha256")
    if expected_evaluator_sha is None and benchmark == "flashinfer":
        expected_evaluator_sha = "08cf436decf15ae1ec5582808b91584f0d7cb2d06b673f76fe3879e6a0634a07"
    require_file(evaluator, str(expected_evaluator_sha or ""), "evaluator")
    if not python.is_file():
        raise SystemExit(f"missing benchmark Python: {python}")

    definition = dataset / str(task_config["definition_relpath"])
    full_workload = dataset / str(task_config["workload_relpath"])
    require_file(definition, str(task_config["definition_sha256"]), "definition")
    require_file(full_workload, str(task_config["workload_sha256"]), "workload")

    workload = full_workload
    if benchmark == "sol_execbench" and stage == "feedback":
        workload = control / "feedback_workloads.jsonl"
        require_file(workload, str(task_config["feedback_workload_sha256"]), "feedback workload")

    return {
        "benchmark": benchmark,
        "evaluator": evaluator,
        "python": python,
        "dataset": dataset,
        "definition": definition,
        "workload": workload,
    }


def sol_reference_cache(task_config: dict, gpu: dict) -> tuple[Path, str]:
    evaluation = task_config["evaluation"]["feedback"]
    identity = {
        "schema": "kda-sol-reference-cache-identity-v1",
        "task": task_config["task"],
        "definition_sha256": task_config["definition_sha256"],
        "evaluator_sha256": task_config["evaluator_sha256"],
        "sol_cli_sha256": sha256(SOL_CLI),
        "gpu_name": gpu["gpu_name"],
        "warmup_runs": evaluation["warmup"],
        "iterations": evaluation["iterations"],
        "seed": evaluation["eval_seed"],
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    cache_key = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    safe_task = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(task_config["task"]))
    path = CONTROL_ROOT / "reference-cache" / "sol" / safe_task / f"{cache_key}.json"
    return path, cache_key


def evaluation_command(
    runtime: dict,
    task_config: dict,
    snapshot: Path,
    result_path: Path,
    stage: str,
    gpu: dict,
) -> list[str]:
    evaluation = task_config.get("evaluation", {}).get(stage, {})
    if runtime["benchmark"] == "flashinfer":
        command = [
            str(runtime["python"]), str(runtime["evaluator"]),
            "--definition", str(runtime["definition"]),
            "--workload", str(runtime["workload"]),
            "--solution", str(snapshot),
            "--entry", "run",
            "--dataset-root", str(runtime["dataset"]),
            "--device", "cuda:0",
            "--warmup", str(evaluation.get("warmup", 3)),
            "--iters", str(evaluation.get("iterations", 100)),
            "--trials", str(evaluation.get("trials", 1)),
            "--json", str(result_path),
        ]
        if stage == "feedback":
            command += ["--workload-index", ",".join(map(str, task_config["feedback_indices"]))]
        return command

    workload_count = len(task_config["feedback_uuids"]) if stage == "feedback" else int(task_config["workload_count"])
    timeout_seconds = max(
        int(evaluation.get("timeout_base_seconds", 1200)),
        int(evaluation.get("timeout_per_workload_seconds", 1800)) * workload_count,
    )
    command = [
        str(runtime["python"]), str(runtime["evaluator"]),
        "--definition", str(runtime["definition"]),
        "--workload", str(runtime["workload"]),
        "--solution", str(snapshot),
        "--output", str(result_path),
        "--warmup", str(evaluation.get("warmup", 10)),
        "--iterations", str(evaluation.get("iterations", 100)),
        "--seed", str(evaluation.get("eval_seed", 200)),
        "--timeout", str(timeout_seconds),
        "--rerun",
    ]
    if stage == "feedback" and evaluation.get("reference_cache", True):
        cache_path, cache_key = sol_reference_cache(task_config, gpu)
        command += ["--reference-cache", str(cache_path), "--reference-cache-key", cache_key]
    return command


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", nargs="?", choices=["feedback", "final"])
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--candidate")
    parser.add_argument("--gpu-status", action="store_true")
    parser.add_argument("--gpu-wait-timeout", type=int, default=GPU_WAIT_TIMEOUT_SECONDS)
    parser.add_argument("--gpu-poll-interval", type=int, default=GPU_POLL_INTERVAL_SECONDS)
    args = parser.parse_args()

    if args.gpu_status:
        print(json.dumps(gpu_status(), ensure_ascii=False, indent=2))
        return 0
    if args.stage is None or args.workspace is None or args.candidate is None:
        parser.error("stage, --workspace, and --candidate are required for evaluation")
    if args.gpu_wait_timeout < 0 or args.gpu_poll_interval <= 0:
        parser.error("GPU wait timeout must be nonnegative and poll interval must be positive")

    workspace = args.workspace.resolve()
    if not workspace.is_relative_to(RUN_ROOT.resolve()):
        raise SystemExit(f"workspace must be under {RUN_ROOT}")
    candidate = validate_candidate_id(args.candidate)
    run_id = workspace.name
    control = CONTROL_ROOT / run_id
    task_config_path = control / "task.json"
    if not task_config_path.is_file():
        raise SystemExit(f"missing trusted task config: {task_config_path}")
    task_config = read_json(task_config_path)
    if task_config.get("run_id") != run_id:
        raise SystemExit("trusted task config run_id mismatch")
    task_name = str(task_config["task"])
    runtime = trusted_runtime(task_config, control, args.stage)
    solution = workspace / "solution" / "solution.py"
    if not solution.is_file():
        raise SystemExit(f"missing candidate source: {solution}")

    source_hash = sha256(solution)
    control.mkdir(parents=True, exist_ok=True)
    lock_path = control / "state.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state_path = control / "state.json"
        state = read_json(state_path) if state_path.exists() else {
            "schema": "kda-controller-state-v2",
            "run_id": run_id,
            "task": task_name,
            "benchmark": runtime["benchmark"],
            "candidate_evaluations": 0,
            "final_evaluations": 0,
            "candidates": {},
        }
        previous = state["candidates"].get(candidate)
        if previous and previous["source_sha256"] != source_hash:
            raise SystemExit(f"{candidate} already belongs to another source hash; use the next candidate id")
        if args.stage == "feedback" and previous and previous.get("feedback_completed"):
            raise SystemExit(f"{candidate} feedback evaluation already completed")
        retrying_static_rejection = bool(
            args.stage == "feedback"
            and previous
            and previous.get("static_check_errors")
            and not previous.get("feedback_completed")
        )
        candidate_budget = int(task_config.get("candidate_evaluation_budget", 100))
        final_budget = int(task_config.get("final_evaluation_budget", 1))
        if args.stage == "feedback" and not retrying_static_rejection and state["candidate_evaluations"] >= candidate_budget:
            raise SystemExit(f"{candidate_budget} candidate-evaluation budget exhausted")
        if args.stage == "final" and state["final_evaluations"] >= final_budget:
            raise SystemExit("final full-evaluation budget exhausted")

        errors = static_check(solution)
        snapshot = control / "candidates" / candidate / "solution.py"
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        if snapshot.exists() and sha256(snapshot) != source_hash:
            raise SystemExit(f"immutable snapshot conflict for {candidate}")
        if not snapshot.exists():
            shutil.copy2(solution, snapshot)

        record = previous or {
            "id": candidate,
            "source_sha256": source_hash,
            "created_at": datetime.now().astimezone().isoformat(),
        }
        state["candidates"][candidate] = record
        # v3 counting: candidate_evaluations is incremented when the evaluation
        # finishes (or fails the static check), never when it starts, so an
        # interrupted benchmark leaves no orphaned count. Idempotent via the
        # feedback_counted flag; see notes/KDA_WATCHDOG_AND_TIMER_20260918.md.
        count_on_completion = args.stage == "feedback" and not retrying_static_rejection
        if args.stage != "feedback":
            state["final_evaluations"] += 1

        output_dir = workspace / "runs" / "candidates" / candidate
        output_dir.mkdir(parents=True, exist_ok=True)
        result_path = output_dir / f"{args.stage}.json"
        log_path = output_dir / f"{args.stage}.log"
        record[f"{args.stage}_started_at"] = datetime.now().astimezone().isoformat()
        record["static_check_errors"] = errors
        record[f"{args.stage}_workloads"] = list(task_config.get("feedback_uuids", [])) if args.stage == "feedback" else "all"
        write_json(state_path, state)

    def count_feedback_evaluation() -> None:
        with lock_path.open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            fresh = read_json(state_path)
            record = fresh["candidates"].get(candidate)
            if record is None or record.get("feedback_counted"):
                return
            fresh["candidate_evaluations"] = int(fresh.get("candidate_evaluations", 0)) + 1
            record["feedback_counted"] = True
            record["evaluation_index"] = fresh["candidate_evaluations"]
            write_json(state_path, fresh)

    if errors:
        if count_on_completion:
            count_feedback_evaluation()
        payload = {
            "valid": False,
            "status": "STATIC_CHECK_FAILED",
            "candidate": candidate,
            "source_sha256": source_hash,
            "errors": errors,
        }
        write_json(result_path, payload)
        log_path.write_text("\n".join(errors) + "\n", encoding="utf-8")
        return 1

    gpu_metadata = None
    gpu_monitor = None
    command: list[str] = []
    try:
        with lease_empty_gpu(args.gpu_wait_timeout, args.gpu_poll_interval) as gpu_metadata:
            # 评测计时窗口：先让选中卡上的显存持有器停止（若有），评测结束后自动占回。
            occupancy_metadata = None
            with occupancy_lease(gpu_metadata["physical_gpu"]) as occupancy_metadata:
                command = evaluation_command(runtime, task_config, snapshot, result_path, args.stage, gpu_metadata)
                env = os.environ.copy()
                env.update({
                    "CUDA_VISIBLE_DEVICES": str(gpu_metadata["physical_gpu"]),
                    "CUDA_HOME": str(CUDA_HOME),
                    "PATH": f"{CUDA_HOME / 'bin'}:{env.get('PATH', '')}",
                    "LD_LIBRARY_PATH": f"{CUDA_HOME / 'lib64'}:{env.get('LD_LIBRARY_PATH', '')}",
                    "FIB_DATASET_PATH": str(FLASHINFER_DATASET),
                    "FIB_CACHE_PATH": str(ROOT / ".cache/flashinfer_bench"),
                    # 本机为 H100（计算能力 9.0）。原值 "8.0" 对应旧机器 A800，
                    # 用于 flashinfer-bench JIT 编译 torch 扩展时的目标架构。
                    "TORCH_CUDA_ARCH_LIST": "9.0",
                    "KDA_GPU_NAME": str(gpu_metadata["gpu_name"]),
                })
                with log_path.open("w", encoding="utf-8") as output:
                    completed, gpu_monitor = run_with_gpu_monitor(command, env, output, gpu_metadata)
            if occupancy_metadata is not None:
                gpu_metadata = {**gpu_metadata, "occupancy_lease": occupancy_metadata}
    except (subprocess.CalledProcessError, RuntimeError, TimeoutError) as error:
        completed = subprocess.CompletedProcess(command, returncode=2)
        log_path.write_text(f"GPU lease failed: {error}\n", encoding="utf-8")

    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = read_json(state_path)
        record = state["candidates"][candidate]
        record[f"{args.stage}_completed_at"] = datetime.now().astimezone().isoformat()
        record[f"{args.stage}_returncode"] = completed.returncode
        record[f"{args.stage}_completed"] = True
        record[f"{args.stage}_result"] = str(result_path)
        record[f"{args.stage}_gpu"] = gpu_metadata
        record[f"{args.stage}_gpu_monitor"] = gpu_monitor
        record[f"{args.stage}_command"] = command
        if count_on_completion and not record.get("feedback_counted"):
            state["candidate_evaluations"] = int(state.get("candidate_evaluations", 0)) + 1
            record["feedback_counted"] = True
            record["evaluation_index"] = state["candidate_evaluations"]
        write_json(state_path, state)

    if gpu_metadata is not None:
        update_result_metadata(result_path, {**gpu_metadata, "monitor": gpu_monitor})

    if log_path.exists():
        print(log_path.read_text(encoding="utf-8"), end="")
    print(f"[kda-controller] candidate={candidate} stage={args.stage} source_sha256={source_hash}")
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
