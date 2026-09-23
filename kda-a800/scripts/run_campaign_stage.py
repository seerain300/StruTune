#!/usr/bin/env python3
"""Run one supervised KDA stage across selected campaign tasks."""

from __future__ import annotations

import argparse
import fnmatch
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path("/data1/workspace/weihongren")
CONTROL_ROOT = ROOT / "kda-control"
RUNNER = ROOT / "kda-controller/run_task_stage.py"


def run_one(workspace: str, stage: str, max_turns: int, force: bool) -> dict:
    command = [sys.executable, str(RUNNER), stage, "--workspace", workspace, "--max-turns", str(max_turns)]
    if force:
        command.append("--force")
    completed = subprocess.run(command, text=True)
    return {"workspace": workspace, "returncode": completed.returncode}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["draft", "plan", "candidate"])
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--tasks", default="*", help="fnmatch pattern over task names or run IDs")
    parser.add_argument("--benchmark", choices=["flashinfer", "sol_execbench"])
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=80)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.max_parallel <= 0:
        parser.error("--max-parallel must be positive")

    campaign_path = args.campaign.resolve()
    if not campaign_path.is_relative_to((CONTROL_ROOT / "campaigns").resolve()):
        raise SystemExit("campaign must be under kda-control/campaigns")
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    selected = [
        item for item in campaign["tasks"]
        if (args.benchmark is None or item["benchmark"] == args.benchmark)
        and (
            fnmatch.fnmatch(item["task"], args.tasks)
            or fnmatch.fnmatch(item["run_id"], args.tasks)
        )
    ]
    if not selected:
        raise SystemExit("no campaign tasks matched")
    if args.dry_run:
        for item in selected:
            print(f"{args.stage}\t{item['benchmark']}\t{item['task']}\t{item['workspace']}")
        return 0

    results = []
    with ThreadPoolExecutor(max_workers=args.max_parallel) as pool:
        futures = {
            pool.submit(run_one, item["workspace"], args.stage, args.max_turns, args.force): item
            for item in selected
        }
        for future in as_completed(futures):
            item = futures[future]
            result = future.result()
            result.update({"run_id": item["run_id"], "task": item["task"]})
            results.append(result)
            print(json.dumps(result, ensure_ascii=False))
    failed = [result for result in results if result["returncode"] != 0]
    print(f"completed={len(results) - len(failed)} failed={len(failed)} total={len(results)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
