#!/usr/bin/env python3
"""Wall-clock-windowed KDA stage driver with graceful deadlines.

Per task (processed sequentially within one driver instance):
  for each window up to --max-windows:
    1. Skip if terminal (SEARCH_COMPLETE / TOKEN_LIMIT_REACHED).
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

Claude is never told about any time limit. Windows and budgets are operator
parameters. Concurrency: waits while global live `claude -p` count is at
--max-claude.

Usage:
  timer_driver.py --campaign <campaign.json> [--tasks pattern]
      [--window-minutes 40] [--grace-minutes 20] [--max-windows 6]
      [--max-turns 40] [--max-claude 4] [--dry-run]
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

ROOT = Path("/data1/workspace/weihongren")
CONTROL_ROOT = ROOT / "kda-control"
RUNNER = ROOT / "kda-controller/run_task_stage.py"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from reconcile_ledger import load_ledger, reconcile_run


def now() -> str:
    return datetime.now().astimezone().isoformat()


def append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as out:
        out.write(json.dumps(payload, ensure_ascii=False) + "\n")


def live_claude_count() -> int:
    completed = subprocess.run(["pgrep", "-fc", r".local/bin/claude -p"], capture_output=True, text=True)
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
        if ".local/bin/claude" in cmdline:
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
    if (workspace / "SEARCH_COMPLETE").is_file() or (workspace / "TOKEN_LIMIT_REACHED").is_file():
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


def run_window(item: dict, stage: str, index: int, args, timer_log: Path) -> str:
    workspace = Path(item["workspace"])
    control = Path(item["control"])
    log_file = control / f"timer-{index:02d}-{stage}.log"
    command = [sys.executable, str(RUNNER), stage, "--workspace", str(workspace), "--max-turns", str(args.max_turns)]
    started = time.time()
    with log_file.open("w", encoding="utf-8") as output:
        output.write(f"[{now()}] {' '.join(command)}\n")
        output.flush()
        process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)

        deadline = started + args.window_minutes * 60
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

    append_jsonl(timer_log, {
        "time": now(), "run_id": item["run_id"], "window": index, "stage": stage,
        "outcome": outcome, "elapsed_minutes": round((time.time() - started) / 60, 1),
    })
    print(f"  window {index} {stage}: {outcome}", flush=True)
    return outcome


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--tasks", default="*")
    parser.add_argument("--window-minutes", type=int, default=40)
    parser.add_argument("--grace-minutes", type=int, default=20)
    parser.add_argument("--max-windows", type=int, default=6)
    parser.add_argument("--round-robin", action="store_true", help="one window per task per pass (overnight fair mode)")
    parser.add_argument("--max-passes", type=int, default=8, help="passes in round-robin mode")
    parser.add_argument("--max-turns", type=int, default=40)
    parser.add_argument("--max-claude", type=int, default=4)
    parser.add_argument("--slot-wait-minutes", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    import fnmatch
    campaign = json.loads(args.campaign.read_text(encoding="utf-8"))
    selected = [
        item for item in campaign["tasks"]
        if fnmatch.fnmatch(item["task"], args.tasks) or fnmatch.fnmatch(item["run_id"], args.tasks)
    ]
    if not selected:
        raise SystemExit("no tasks matched")

    print(f"timer driver: {len(selected)} task(s), window={args.window_minutes}m "
          f"grace={args.grace_minutes}m max-windows={args.max_windows} "
          f"max-claude={args.max_claude}", flush=True)
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
            print(f"  {next_stage(item) or 'terminal'}\t{item['task'][:50]}\tbudget={budget}{warn}", flush=True)
        return 0

    timer_log = CONTROL_ROOT / "campaigns" / f"{campaign['tag']}.timer.jsonl"
    if args.round_robin:
        # Overnight fair scheduling: each pass gives every non-terminal task
        # exactly one window, so no task starves behind a long-running one.
        for passes in range(1, args.max_passes + 1):
            progressed = False
            for item in selected:
                if next_stage(item) is None:
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
                run_window(item, next_stage(item), passes, args, timer_log)
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
                reconciliation = reconcile_run(item["run_id"], apply=True)
                if reconciliation.get("needs_human"):
                    print(f"  reconciliation needs human: {reconciliation['needs_human']}; skipping task", flush=True)
                    break
                if not wait_for_slot(args.max_claude, args.slot_wait_minutes):
                    print("  no concurrency slot; moving on", flush=True)
                    break
                outcome = run_window(item, stage, index, args, timer_log)
                if outcome.startswith("completed rc=0") and stage in {"draft", "plan"}:
                    continue  # advance to the next stage in the following window
                time.sleep(10)
    print("driver finished", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
