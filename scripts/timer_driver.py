#!/usr/bin/env python3
"""Wall-clock-windowed KDA stage driver with graceful deadlines.

Per task (processed sequentially within one driver instance):
  for each window up to --max-windows:
    1. Skip if terminal (SEARCH_COMPLETE / TOKEN_LIMIT_REACHED / TIME_BUDGET_REACHED).
    2. Reconcile the state/ledger invariant (reconcile_ledger).
    3. Determine the next stage (draft -> plan -> candidate).
    4. Spawn run_task_stage.py in its own process group, wait up to
       --window-minutes.
    5. At the deadline: if the task is NOT in the evaluation danger window
       (no live evaluator for the workspace AND counter == ledger records),
       kill the process group immediately (safe point). Otherwise defer up to
       --grace-minutes, re-checking every 30s; kill the moment the window
       closes, or hard-kill (SIGKILL) when grace runs out. Reconciliation on
       the next window covers the forced case.

Per-task total time budget (--task-budget-minutes, e.g. 60 for one hour):
  Caps the cumulative wall time a task may consume across ALL of its windows
  (draft + plan + every candidate stage combined). Accounting is replayed
  from the campaign timer log at driver startup, so restarting the driver
  does not reset a task's budget. Each window is shortened to the remaining
  budget; once exhausted, the workspace receives a TIME_BUDGET_REACHED
  marker, which pool / driver / campaign_status / watchdog all treat as a
  terminal state. The evaluation danger-window grace may overshoot the
  budget by at most --grace-minutes (this is required for kill safety).

Stop semantics (SIGTERM/SIGINT): the driver kills the in-flight stage
process group and self-logs the window record ("killed-by-signal", author
"timer_driver") before exiting — manual ledger corrections are therefore
never needed and must not be written. load_task_usage additionally
de-duplicates same-window records (driver rows win over foreign rows) as a
safety net against historical double bookkeeping.

Claude is never told about any time limit. Windows and budgets are operator
parameters. Concurrency: waits while global live `claude -p` count is at
--max-claude.

Usage:
  timer_driver.py --campaign <campaign.json> [--tasks pattern[,...]]
      [--window-minutes 40] [--grace-minutes 20] [--max-windows 6]
      [--task-budget-minutes 0] [--max-turns 40] [--max-claude 4] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path("/home/ziming/kda-ops")
CONTROL_ROOT = ROOT / "kda-control"
RUNNER = ROOT / "kda-controller/run_task_stage.py"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from reconcile_ledger import load_ledger, reconcile_run


def now() -> str:
    return datetime.now().astimezone().isoformat()


def append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as out:
        out.write(json.dumps(payload, ensure_ascii=False) + "\n")


class TerminatedBySignal(Exception):
    """driver 收到 SIGTERM/SIGINT 时从信号处理器抛出，用于中断当前窗口并自记账。"""


def _termination_signal_handler(signum, _frame):
    raise TerminatedBySignal(f"signal {signum}")


def install_signal_handlers() -> None:
    signal.signal(signal.SIGTERM, _termination_signal_handler)
    signal.signal(signal.SIGINT, _termination_signal_handler)


def live_claude_count() -> int:
    completed = subprocess.run(["pgrep", "-fc", r"bin/claude -p"], capture_output=True, text=True)
    try:
        return int(completed.stdout.strip() or "0")
    except ValueError:
        return 0


def evaluator_alive(workspace: Path) -> bool:
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            cmdline = (proc / "cmdline").read_bytes().decode("utf-8", "replace")
        except OSError:
            continue
        if "evaluate_candidate" not in cmdline:
            continue
        # The candidate-stage prompt text itself mentions evaluate_candidate,
        # so the claude process must never count as a live evaluator.
        if "bin/claude" in cmdline:
            continue
        try:
            cwd = os.readlink(proc / "cwd")
        except OSError:
            continue
        if str(workspace) in (cwd or "") or str(workspace) in cmdline:
            return True
    return False


def invariant_state(control: Path, workspace: Path) -> bool:
    state = json.loads((control / "state.json").read_text(encoding="utf-8"))
    records, _, _ = load_ledger(workspace / "candidates.jsonl")
    return int(state.get("candidate_evaluations", 0)) == len(records)


def in_danger_window(control: Path, workspace: Path) -> bool:
    return evaluator_alive(workspace) or not invariant_state(control, workspace)


def next_stage(item: dict) -> str | None:
    workspace = Path(item["workspace"])
    control = Path(item["control"])
    terminal_markers = ("SEARCH_COMPLETE", "TOKEN_LIMIT_REACHED", "TIME_BUDGET_REACHED")
    if any((workspace / marker).is_file() for marker in terminal_markers):
        return None
    if not (workspace / "docs/draft.md").is_file():
        return "draft"
    if not (workspace / "docs/plan.md").is_file():
        return "plan"
    state = json.loads((control / "state.json").read_text(encoding="utf-8"))
    config = json.loads((control / "task.json").read_text(encoding="utf-8"))
    if int(state.get("candidate_evaluations", 0)) >= int(config["candidate_evaluation_budget"]):
        return None
    return "candidate"


# driver 自产行的 outcome 前缀（含 author 标记启用前的历史行也按此识别）。
DRIVER_OUTCOME_PREFIXES = (
    "completed rc=", "completed-during-grace", "killed-safe",
    "killed-after-grace", "killed-forced", "killed-by-signal",
)


def _is_driver_row(record: dict) -> bool:
    if record.get("author") == "timer_driver":
        return True
    return str(record.get("outcome", "")).startswith(DRIVER_OUTCOME_PREFIXES)


def load_task_usage(timer_log: Path) -> dict[str, float]:
    """Replay per-task cumulative window time (minutes) from the timer log.

    The timer log is append-only and records the elapsed minutes of every
    window, so replaying it at startup keeps the per-task budget accounting
    intact across driver restarts.

    De-duplication safety net: each row defines a time interval
    [time - elapsed, time]. Two rows of the SAME task whose intervals overlap
    by more than 60 seconds describe one window booked twice — only one of
    them is counted (the driver-authored row wins over foreign/manual rows;
    among equals, the longer elapsed wins). Rows that merely share a window
    NUMBER are not duplicates: every driver instance numbers its windows
    from 1, so a re-run task legitimately logs window 1 twice at disjoint
    times. Rows with an "event" field are audit events, never counted.
    """
    from datetime import timedelta

    rows_by_task: dict[str, list[dict]] = {}
    if not timer_log.is_file():
        return {}
    for line in timer_log.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        run_id = record.get("run_id")
        minutes = record.get("elapsed_minutes")
        if not run_id or not isinstance(minutes, (int, float)) or record.get("event"):
            continue
        try:
            ended = datetime.fromisoformat(record.get("time", ""))
        except ValueError:
            continue
        rows_by_task.setdefault(run_id, []).append({
            "ended": ended,
            "minutes": float(minutes),
            "is_driver": _is_driver_row(record),
        })

    usage: dict[str, float] = {}
    for run_id, rows in rows_by_task.items():
        accepted: list[dict] = []
        for row in sorted(rows, key=lambda r: r["ended"]):
            start = row["ended"] - timedelta(minutes=row["minutes"])
            duplicate_of = None
            for kept in accepted:
                kept_start = kept["ended"] - timedelta(minutes=kept["minutes"])
                overlap = (min(row["ended"], kept["ended"]) - max(start, kept_start)).total_seconds()
                if overlap > 60:
                    duplicate_of = kept
                    break
            if duplicate_of is None:
                accepted.append(row)
                continue
            # 与已接受行重叠 >60s：同一窗口的重复记账，择一保留。
            if row["is_driver"] and not duplicate_of["is_driver"]:
                accepted.remove(duplicate_of)
                accepted.append(row)
            elif row["is_driver"] == duplicate_of["is_driver"] and row["minutes"] > duplicate_of["minutes"]:
                accepted.remove(duplicate_of)
                accepted.append(row)
        usage[run_id] = round(sum(row["minutes"] for row in accepted), 1)
    return usage


def write_time_budget_marker(
    workspace: Path, control: Path, run_id: str, budget_minutes: int, cumulative_minutes: float
) -> None:
    """Mark a workspace as terminated by its per-task time budget."""
    state = json.loads((control / "state.json").read_text(encoding="utf-8"))
    payload = {
        "schema": "kda-time-budget-reached-v1",
        "written_by": "timer_driver",
        "run_id": run_id,
        "task_budget_minutes": budget_minutes,
        "cumulative_window_minutes": round(cumulative_minutes, 1),
        "candidate_evaluations": int(state.get("candidate_evaluations", 0)),
        "time": now(),
    }
    (workspace / "TIME_BUDGET_REACHED").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def kill_group(pid: int, grace_seconds: int = 120) -> str:
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return "already-exited"
    deadline = time.time() + grace_seconds
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return "terminated"
        time.sleep(5)
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    return "killed"


def wait_for_slot(max_claude: int, timeout_minutes: int) -> bool:
    deadline = time.time() + timeout_minutes * 60
    while time.time() < deadline:
        if live_claude_count() < max_claude:
            return True
        time.sleep(60)
    return False


def run_window(
    item: dict, stage: str, index: int, args, timer_log: Path, window_minutes: float
) -> tuple[str, float]:
    workspace = Path(item["workspace"])
    control = Path(item["control"])
    log_file = control / f"timer-{index:02d}-{stage}.log"
    command = [sys.executable, str(RUNNER), stage, "--workspace", str(workspace), "--max-turns", str(args.max_turns)]
    started = time.time()
    outcome = "unknown"
    terminated = False
    with log_file.open("w", encoding="utf-8") as output:
        output.write(f"[{now()}] {' '.join(command)}\n")
        output.flush()
        process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)

        deadline = started + window_minutes * 60
        try:
            while True:
                if process.poll() is not None:
                    outcome = f"completed rc={process.returncode}"
                    break
                if time.time() >= deadline:
                    if not in_danger_window(control, workspace):
                        outcome = "killed-safe"
                        outcome += f" ({kill_group(process.pid)})"
                        break
                    grace_deadline = time.time() + args.grace_minutes * 60
                    while time.time() < grace_deadline:
                        time.sleep(30)
                        if process.poll() is not None:
                            outcome = f"completed-during-grace rc={process.returncode}"
                            break
                        if not in_danger_window(control, workspace):
                            outcome = "killed-after-grace"
                            outcome += f" ({kill_group(process.pid)})"
                            break
                    else:
                        outcome = f"killed-forced ({kill_group(process.pid, grace_seconds=15)})"
                    break
                time.sleep(15)
        except TerminatedBySignal as signal_info:
            # driver 被 SIGTERM/SIGINT 终止：先杀本窗 stage 进程组，再自行补记窗口耗时，
            # 然后继续抛出让调度循环停止（不再开新窗口）。
            # 有了这条路径，运营者停 driver 不再需要（也不应该）手工补记。
            outcome = f"killed-by-signal ({kill_group(process.pid)}, {signal_info})"
            terminated = True

    append_jsonl(timer_log, {
        "time": now(), "run_id": item["run_id"], "window": index, "stage": stage,
        "outcome": outcome, "author": "timer_driver",
        "elapsed_minutes": round((time.time() - started) / 60, 1),
    })
    elapsed_minutes = round((time.time() - started) / 60, 1)
    print(f"  window {index} {stage}: {outcome}", flush=True)
    if terminated:
        raise TerminatedBySignal("window self-logged; stopping driver")
    return outcome, elapsed_minutes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--tasks", default="*",
                        help="comma-separated fnmatch patterns (matched against task or run_id)")
    parser.add_argument("--window-minutes", type=int, default=40)
    parser.add_argument("--grace-minutes", type=int, default=20)
    parser.add_argument("--max-windows", type=int, default=6)
    parser.add_argument(
        "--task-budget-minutes", type=int, default=0,
        help="cumulative wall-clock cap per task across ALL its windows "
             "(draft + plan + candidates); 0 disables the cap. Windows are "
             "shortened to the remaining budget and exhaustion writes a "
             "TIME_BUDGET_REACHED marker (terminal state).",
    )
    parser.add_argument("--round-robin", action="store_true", help="one window per task per pass (overnight fair mode)")
    parser.add_argument("--max-passes", type=int, default=8, help="passes in round-robin mode")
    parser.add_argument("--max-turns", type=int, default=40)
    parser.add_argument("--max-claude", type=int, default=4)
    parser.add_argument("--slot-wait-minutes", type=int, default=30)
    parser.add_argument("--stages", default="draft,plan,candidate",
                        help="comma-separated subset of stages this driver may run "
                             "(e.g. 'draft,plan' for GPU-free preparation batches; "
                             "tasks whose next stage is outside the subset are skipped)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    allowed_stages = {part.strip() for part in args.stages.split(",") if part.strip()}

    import fnmatch
    campaign = json.loads(args.campaign.read_text(encoding="utf-8"))
    patterns = [part.strip() for part in args.tasks.split(",") if part.strip()]
    selected = [
        item for item in campaign["tasks"]
        if any(
            fnmatch.fnmatch(item["task"], pattern) or fnmatch.fnmatch(item["run_id"], pattern)
            for pattern in patterns
        )
    ]
    if not selected:
        raise SystemExit("no tasks matched")

    print(f"timer driver: {len(selected)} task(s), window={args.window_minutes}m "
          f"grace={args.grace_minutes}m max-windows={args.max_windows} "
          f"task-budget={args.task_budget_minutes or 'off'}m "
          f"max-claude={args.max_claude}", flush=True)

    timer_log = CONTROL_ROOT / "campaigns" / f"{campaign['tag']}.timer.jsonl"
    usage = load_task_usage(timer_log) if args.task_budget_minutes > 0 else {}

    def remaining_minutes(item: dict) -> float | None:
        """Remaining time budget for a task; None when the cap is disabled."""
        if args.task_budget_minutes <= 0:
            return None
        return args.task_budget_minutes - usage.get(item["run_id"], 0.0)

    def budget_exhausted(item: dict) -> bool:
        remaining = remaining_minutes(item)
        return remaining is not None and remaining <= 0

    def effective_window(item: dict) -> float:
        remaining = remaining_minutes(item)
        if remaining is None:
            return float(args.window_minutes)
        return max(min(float(args.window_minutes), remaining), 0.0)

    def settle_budget(item: dict, elapsed_minutes: float) -> None:
        """Record a finished window and mark the task if its budget ran out."""
        if args.task_budget_minutes <= 0:
            return
        usage[item["run_id"]] = usage.get(item["run_id"], 0.0) + elapsed_minutes
        if budget_exhausted(item):
            workspace = Path(item["workspace"])
            if not (workspace / "TIME_BUDGET_REACHED").is_file():
                write_time_budget_marker(
                    workspace, Path(item["control"]), item["run_id"],
                    args.task_budget_minutes, usage[item["run_id"]],
                )
                append_jsonl(timer_log, {
                    "time": now(), "run_id": item["run_id"],
                    "event": "task-budget-exhausted",
                    "task_budget_minutes": args.task_budget_minutes,
                    "cumulative_window_minutes": round(usage[item["run_id"]], 1),
                })
                print(f"  time budget exhausted "
                      f"({usage[item['run_id']]:.1f}m >= {args.task_budget_minutes}m); "
                      f"TIME_BUDGET_REACHED written", flush=True)

    if args.dry_run:
        for item in selected:
            workspace = Path(item["workspace"])
            config = json.loads((Path(item["control"]) / "task.json").read_text(encoding="utf-8"))
            budget = 0
            obs = Path(item["control"]) / "observability.json"
            if obs.is_file():
                budget = int((json.loads(obs.read_text(encoding="utf-8")).get("usage") or {}).get("budget_tokens", 0))
            warn = ""
            if budget >= int(config["token_soft_limit"]):
                warn = "  [WARN budget >= soft limit; raise token limits in task.json before real run]"
            time_note = ""
            if args.task_budget_minutes > 0:
                used = usage.get(item["run_id"], 0.0)
                remaining = args.task_budget_minutes - used
                time_note = f"\ttime_used={used:.0f}m time_remaining={max(remaining, 0):.0f}m"
                if remaining <= 0:
                    time_note += "  [TIME BUDGET EXHAUSTED]"
            print(f"  {next_stage(item) or 'terminal'}\t{item['task'][:50]}\tbudget={budget}{warn}{time_note}", flush=True)
        return 0

    # 安装信号自记账：SIGTERM/SIGINT 会让在途窗口先补记再退出（见 run_window）。
    install_signal_handlers()

    try:
        if args.round_robin:
            # Overnight fair scheduling: each pass gives every non-terminal task
            # exactly one window, so no task starves behind a long-running one.
            for passes in range(1, args.max_passes + 1):
                progressed = False
                for item in selected:
                    if next_stage(item) is None:
                        continue
                    stage_now = next_stage(item)
                    if stage_now not in allowed_stages:
                        continue  # 该题下一阶段不在本 driver 的阶段集合内（prep 批次跳过）
                    if budget_exhausted(item):
                        settle_budget(item, 0.0)  # ensure the marker exists, no time to add
                        continue
                    progressed = True
                    print(f"== pass {passes} {item['run_id'].split('--')[-1]}", flush=True)
                    reconciliation = reconcile_run(item["run_id"], apply=True)
                    if reconciliation.get("needs_human"):
                        print(f"  reconciliation needs human: {reconciliation['needs_human']}; skipping", flush=True)
                        continue
                    if not wait_for_slot(args.max_claude, args.slot_wait_minutes):
                        print("  no concurrency slot; skipping this pass", flush=True)
                        continue
                    window = effective_window(item)
                    outcome, elapsed = run_window(item, next_stage(item), passes, args, timer_log, window)
                    settle_budget(item, elapsed)
                    time.sleep(10)
                if not progressed:
                    print("all tasks terminal", flush=True)
                    break
        else:
            for item in selected:
                print(f"== {item['run_id'].split('--')[-1]}", flush=True)
                for index in range(1, args.max_windows + 1):
                    stage = next_stage(item)
                    if stage is None:
                        print("  terminal; next task", flush=True)
                        break
                    if stage not in allowed_stages:
                        print(f"  next stage '{stage}' outside --stages; next task", flush=True)
                        break
                    if budget_exhausted(item):
                        settle_budget(item, 0.0)  # ensure the marker exists, no time to add
                        print("  time budget exhausted; next task", flush=True)
                        break
                    reconciliation = reconcile_run(item["run_id"], apply=True)
                    if reconciliation.get("needs_human"):
                        print(f"  reconciliation needs human: {reconciliation['needs_human']}; skipping task", flush=True)
                        break
                    if not wait_for_slot(args.max_claude, args.slot_wait_minutes):
                        print("  no concurrency slot; moving on", flush=True)
                        break
                    window = effective_window(item)
                    outcome, elapsed = run_window(item, stage, index, args, timer_log, window)
                    settle_budget(item, elapsed)
                    if outcome.startswith("completed rc=0") and stage in {"draft", "plan"}:
                        continue  # advance to the next stage in the following window
                    time.sleep(10)
    except TerminatedBySignal:
        # 在途窗口已在 run_window 内自行补记并杀掉 stage；这里干净退出。
        print("driver terminated by signal; in-flight window self-logged", flush=True)
        return 0
    print("driver finished", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
