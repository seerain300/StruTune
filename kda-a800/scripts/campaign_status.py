#!/usr/bin/env python3
"""Summarize KDA campaign progress from durable artifacts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True, type=Path)
    args = parser.parse_args()
    campaign = read_json(args.campaign)
    stages = Counter()
    evaluations = 0
    final_evaluations = 0
    budget_tokens = 0
    blocked = []
    api_retries = 0
    retry_rerun_recommended = []
    for item in campaign["tasks"]:
        workspace = Path(item["workspace"])
        state = read_json(Path(item["control"]) / "state.json")
        evaluations += int(state.get("candidate_evaluations", 0))
        final_evaluations += int(state.get("final_evaluations", 0))
        summary_path = Path(item["control"]) / "observability.json"
        if summary_path.is_file():
            summary = read_json(summary_path)
            budget_tokens += int((summary.get("usage") or {}).get("budget_tokens", 0))
            transport = summary.get("transport") or {}
            api_retries += int(transport.get("api_retries", 0))
            if transport.get("rerun_recommended"):
                retry_rerun_recommended.append(item["run_id"])
        if (workspace / "POOL_BLOCKED").is_file():
            stage = "blocked"
            blocked.append(item["run_id"])
        elif (workspace / "SEARCH_COMPLETE").is_file():
            stage = "search_complete"
        elif (workspace / "TOKEN_LIMIT_REACHED").is_file():
            stage = "token_stopped"
        elif state.get("candidate_evaluations", 0):
            stage = "searching"
        elif (workspace / "docs/plan.md").is_file():
            stage = "planned"
        elif (workspace / "docs/draft.md").is_file():
            stage = "drafted"
        else:
            stage = "pending"
        stages[stage] += 1
    payload = {
        "campaign": campaign["tag"],
        "tasks": len(campaign["tasks"]),
        "stages": dict(stages),
        "candidate_evaluations": evaluations,
        "final_evaluations": final_evaluations,
        "budget_tokens": budget_tokens,
        "api_retries": api_retries,
        "retry_rerun_recommended": retry_rerun_recommended,
        "blocked_run_ids": blocked,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
