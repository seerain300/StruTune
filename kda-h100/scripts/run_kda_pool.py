#!/usr/bin/env python3
"""Persistent six-worker pool for the full supervised KDA search workflow."""

from __future__ import annotations

import argparse
import fnmatch
import json
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path


ROOT = Path("/home/ziming/kda-ops")
CONTROL_ROOT = ROOT / "kda-control"
RUNNER = ROOT / "kda-controller/run_task_stage.py"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def append_jsonl(path: Path, payload: dict, lock: threading.Lock) -> None:
    line = json.dumps(payload, ensure_ascii=False) + "\n"
    with lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as output:
            output.write(line)


def task_status(item: dict) -> dict:
    workspace = Path(item["workspace"])
    control = Path(item["control"])
    config = read_json(control / "task.json")
    state = read_json(control / "state.json")
    return {
        "workspace": workspace,
        "control": control,
        "config": config,
        "state": state,
        "draft": (workspace / "docs/draft.md").is_file(),
        "plan": (workspace / "docs/plan.md").is_file(),
        "complete": (workspace / "SEARCH_COMPLETE").is_file(),
        "token_stopped": (workspace / "TOKEN_LIMIT_REACHED").is_file(),
        "time_stopped": (workspace / "TIME_BUDGET_REACHED").is_file(),
    }


def next_stage(status: dict) -> str | None:
    if status["complete"] or status["token_stopped"] or status["time_stopped"]:
        return None
    if not status["draft"]:
        return "draft"
    if not status["plan"]:
        return "plan"
    if int(status["state"].get("candidate_evaluations", 0)) >= int(status["config"]["candidate_evaluation_budget"]):
        return None
    return "candidate"


def run_task(item: dict, max_turns: int, events: Path, event_lock: threading.Lock) -> dict:
    workspace = Path(item["workspace"])
    driver_log = Path(item["control"]) / "pool-driver.log"
    while True:
        status = task_status(item)
        stage = next_stage(status)
        if stage is None:
            reason = (
                "complete" if status["complete"]
                else "token-limit" if status["token_stopped"]
                else "time-budget" if status["time_stopped"]
                else "candidate-budget"
            )
            result = {"time": datetime.now().astimezone().isoformat(), "run_id": item["run_id"], "event": "task-stopped", "reason": reason}
            append_jsonl(events, result, event_lock)
            return result

        event = {"time": datetime.now().astimezone().isoformat(), "run_id": item["run_id"], "event": "stage-start", "stage": stage}
        append_jsonl(events, event, event_lock)
        command = [sys.executable, str(RUNNER), stage, "--workspace", str(workspace), "--max-turns", str(max_turns)]
        with driver_log.open("a", encoding="utf-8") as output:
            output.write(f"\n[{event['time']}] {' '.join(command)}\n")
            output.flush()
            completed = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, text=True)
        event = {
            "time": datetime.now().astimezone().isoformat(),
            "run_id": item["run_id"],
            "event": "stage-end",
            "stage": stage,
            "returncode": completed.returncode,
        }
        append_jsonl(events, event, event_lock)
        if completed.returncode != 0:
            blocked = workspace / "POOL_BLOCKED"
            blocked.write_text(f"stage={stage} returncode={completed.returncode}\n", encoding="utf-8")
            return {**event, "event": "task-blocked"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--tasks", default="*", help="fnmatch pattern over task names or run IDs")
    parser.add_argument("--benchmark", choices=["flashinfer", "sol_execbench"])
    parser.add_argument("--max-turns", type=int, default=80)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.workers <= 0 or args.workers > 6:
        parser.error("--workers must be between 1 and 6")

    campaign_path = args.campaign.resolve()
    if not campaign_path.is_relative_to((CONTROL_ROOT / "campaigns").resolve()):
        raise SystemExit("campaign must be under kda-control/campaigns")
    campaign = read_json(campaign_path)
    selected = [
        item for item in campaign["tasks"]
        if (args.benchmark is None or item["benchmark"] == args.benchmark)
        and (fnmatch.fnmatch(item["task"], args.tasks) or fnmatch.fnmatch(item["run_id"], args.tasks))
    ]
    if not selected:
        raise SystemExit("no campaign tasks matched")
    if args.dry_run:
        for item in selected:
            status = task_status(item)
            print(f"{next_stage(status) or 'stopped'}\t{item['benchmark']}\t{item['task']}")
        print(f"pool workers={args.workers} tasks={len(selected)} final_evaluation=disabled")
        return 0

    events = CONTROL_ROOT / "campaigns" / f"{campaign['tag']}.events.jsonl"
    event_lock = threading.Lock()
    append_jsonl(events, {
        "time": datetime.now().astimezone().isoformat(),
        "event": "pool-start",
        "workers": args.workers,
        "tasks": len(selected),
        "final_evaluation": False,
    }, event_lock)
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_task, item, args.max_turns, events, event_lock): item
            for item in selected
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    blocked = [result for result in results if result.get("event") == "task-blocked"]
    print(f"tasks={len(results)} blocked={len(blocked)}")
    return 1 if blocked else 0


if __name__ == "__main__":
    raise SystemExit(main())
