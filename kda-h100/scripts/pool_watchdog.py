#!/usr/bin/env python3
"""Watchdog: auto-rescue KDA tasks killed by relay API-timeout failures.

Behavior:
- Poll campaign tasks for workspace POOL_BLOCKED markers whose returncode is 1
  (API timeout / server_error death). Never touch rc=143 (operator kills).
- Before rescuing, require: relay API probe returns HTTP 200, no live stage
  runner for that task, per-task rescue budget not exhausted, and global
  live `claude -p` process count below the cap (concurrency control).
- Rescue = delete POOL_BLOCKED and spawn a single-task pool
  (run_kda_pool.py --tasks <run_id> --workers 1), which resumes the existing
  Claude session via .claude-session-id.
- Per-task rescue counter resets after any later stage-end rc=0 for that task.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reconcile_ledger import reconcile_run

ROOT = Path("/home/ziming/kda-ops")
CONTROL_ROOT = ROOT / "kda-control"
POOL = ROOT / "kda-controller/run_kda_pool.py"
LLM_ENV = ROOT / "llm.env"  # 可选兜底；单机部署密钥在环境变量
RELAY = "https://llmapi.isrc.ac.cn/v1/chat/completions"


def now() -> str:
    return datetime.now().astimezone().isoformat()


def append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as out:
        out.write(json.dumps(payload, ensure_ascii=False) + "\n")


def relay_key() -> str | None:
    # 单机部署：密钥来自环境变量（~/.bashrc export KDA_API_KEY），优先于 llm.env 文件。
    priority = (
        "KDA_DEF_FEY", "KDA_API_KEY", "KDA_KEY",
        "ISRC_API_KEY", "LLM_API_KEY", "NEW_API_KEY",
    )
    for name in priority:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    if not LLM_ENV.is_file():
        return None
    found: dict[str, str] = {}
    for line in LLM_ENV.read_text(encoding="utf-8").splitlines():
        match = re.match(
            r'\s*(?:export\s+)?([A-Za-z_]+)\s*=\s*["\']?([^"\'\n]+)', line
        )
        if match and match.group(1) in priority:
            found.setdefault(match.group(1), match.group(2).strip())
    for name in priority:
        if name in found:
            return found[name]
    return None


def api_healthy() -> bool:
    # The relay returns 503 to Python urllib (TLS/HTTP-version fingerprinting)
    # while accepting curl, so the probe must go through curl.
    key = relay_key()
    if not key:
        return False
    completed = subprocess.run(
        [
            "curl", "-sS", "-o", "/dev/null", "-w", "%{http_code}",
            RELAY,
            "-H", f"Authorization: Bearer {key}",
            "-H", "Content-Type: application/json",
            "--max-time", "30",
            "-d", json.dumps({
                "model": "Claude-Opus-4.8",
                "messages": [{"role": "user", "content": "Reply only: OK"}],
                "max_tokens": 8,
            }),
        ],
        capture_output=True, text=True,
    )
    return completed.stdout.strip().endswith("200")


def pgrep(pattern: str) -> int:
    completed = subprocess.run(
        ["pgrep", "-fc", pattern], capture_output=True, text=True
    )
    try:
        return int(completed.stdout.strip() or "0")
    except ValueError:
        return 0


def live_claude_count() -> int:
    return pgrep(r"bin/claude -p")


def task_busy(run_id: str) -> bool:
    return pgrep(f"run_task_stage {run_id}") > 0 or pgrep(f"run_task_stage.*{run_id}") > 0


def blocked_returncode(marker: Path) -> int | None:
    text = marker.read_text(encoding="utf-8")
    match = re.search(r"returncode=(-?\d+)", text)
    return int(match.group(1)) if match else None


def next_stage_exists(item: dict) -> bool:
    workspace = Path(item["workspace"])
    state = json.loads((Path(item["control"]) / "state.json").read_text(encoding="utf-8"))
    config = json.loads((Path(item["control"]) / "task.json").read_text(encoding="utf-8"))
    terminal_markers = ("SEARCH_COMPLETE", "TOKEN_LIMIT_REACHED", "TIME_BUDGET_REACHED")
    if any((workspace / marker).is_file() for marker in terminal_markers):
        return False
    if not (workspace / "docs/draft.md").is_file():
        return True
    if not (workspace / "docs/plan.md").is_file():
        return True
    return int(state.get("candidate_evaluations", 0)) < int(config["candidate_evaluation_budget"])


def successful_stage_after(events: Path, run_id: str, since: str) -> bool:
    if not events.is_file():
        return False
    for line in events.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            event.get("run_id") == run_id
            and event.get("event") == "stage-end"
            and event.get("returncode") == 0
            and str(event.get("time", "")) > since
        ):
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--max-claude", type=int, default=4, help="global cap on live claude -p processes")
    parser.add_argument("--max-rescues", type=int, default=3, help="rescue attempts per task before giving up")
    parser.add_argument("--max-turns", type=int, default=40)
    parser.add_argument("--interval", type=int, default=60)
    args = parser.parse_args()

    campaign_path = args.campaign.resolve()
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    tag = campaign["tag"]
    events = CONTROL_ROOT / "campaigns" / f"{tag}.events.jsonl"
    state_path = CONTROL_ROOT / "campaigns" / f"{tag}.watchdog-state.json"
    log_path = CONTROL_ROOT / "campaigns" / f"{tag}.watchdog.jsonl"
    state: dict = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}

    append_jsonl(log_path, {
        "time": now(), "event": "watchdog-start",
        "campaign": str(campaign_path), "max_claude": args.max_claude,
        "max_rescues": args.max_rescues, "pid": os.getpid(),
    })
    print(f"watchdog started pid={os.getpid()} campaign={tag}", flush=True)

    while True:
        try:
            live = live_claude_count()
            rescued_this_tick = False
            for item in campaign["tasks"]:
                run_id = item["run_id"]
                marker = Path(item["workspace"]) / "POOL_BLOCKED"
                record = state.setdefault(run_id, {"rescues": 0, "last_spawn": "", "pid": None})

                if record["last_spawn"] and successful_stage_after(events, run_id, record["last_spawn"]):
                    if record["rescues"]:
                        append_jsonl(log_path, {"time": now(), "run_id": run_id, "event": "rescue-counter-reset"})
                    record.update(rescues=0, last_spawn="")

                if not marker.is_file():
                    continue
                rc = blocked_returncode(marker)
                # rc=1: API timeout death. rc=4: stage ended cleanly but broke the
                # exactly-one-evaluation gate (often a false alarm when the model
                # evaluates two candidates in one stage) — recycle both.
                if rc not in (1, 4):
                    continue
                if not next_stage_exists(item):
                    continue
                if record["rescues"] >= args.max_rescues:
                    continue
                if task_busy(run_id):
                    continue
                if live >= args.max_claude:
                    append_jsonl(log_path, {"time": now(), "run_id": run_id, "event": "rescue-deferred", "reason": f"claude cap {live}/{args.max_claude}"})
                    continue
                if not api_healthy():
                    append_jsonl(log_path, {"time": now(), "run_id": run_id, "event": "rescue-deferred", "reason": "api probe not 200"})
                    continue

                marker.unlink()
                reconciliation = reconcile_run(run_id, apply=True)
                append_jsonl(log_path, {
                    "time": now(), "run_id": run_id, "event": "pre-rescue-reconciliation",
                    "changed": reconciliation.get("changed"),
                    "needs_human": reconciliation.get("needs_human"),
                })
                log_file = CONTROL_ROOT / "campaigns" / f"{tag}.rescue-{run_id}-{record['rescues'] + 1}.log"
                command = [
                    sys.executable, str(POOL),
                    "--campaign", str(campaign_path),
                    "--tasks", run_id,
                    "--workers", "1",
                    "--max-turns", str(args.max_turns),
                ]
                with log_file.open("w", encoding="utf-8") as output:
                    process = subprocess.Popen(
                        command, stdout=output, stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                record.update(rescues=record["rescues"] + 1, last_spawn=now(), pid=process.pid)
                append_jsonl(log_path, {
                    "time": now(), "run_id": run_id, "event": "rescue-spawned",
                    "attempt": record["rescues"], "pid": process.pid, "log": str(log_file),
                })
                state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                live += 1
                rescued_this_tick = True
                break  # at most one rescue per tick

            state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            if not rescued_this_tick:
                print(f"[{now()[11:19]}] live_claude={live} nothing to rescue", flush=True)
        except Exception as error:  # keep the watchdog alive on unexpected errors
            append_jsonl(log_path, {"time": now(), "event": "watchdog-error", "error": repr(error)})
        try:
            import time
            time.sleep(args.interval)
        except KeyboardInterrupt:
            append_jsonl(log_path, {"time": now(), "event": "watchdog-stop"})
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
