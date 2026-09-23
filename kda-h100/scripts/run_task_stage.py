#!/usr/bin/env python3
"""Run one supervised Claude stage for an isolated KDA task."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path


ROOT = Path("/home/ziming/kda-ops")
RUN_ROOT = ROOT / "kda-runs"
CONTROL_ROOT = ROOT / "kda-control"
CLAUDE = ROOT / "bin/claude-kda-opus48"
SUMMARIZER = ROOT / "kda-observability/summarize_claude_transcript.py"


PROMPTS = {
    "draft": """Read CLAUDE.md, TASK.md, README.md, task/definition.json, and task/feedback_workloads.jsonl. Follow the complete KDA workflow. For this turn, create only docs/draft.md with a thorough analysis of the operation, constraints, numerical risks, Triton design space, and validation strategy. Do not create docs/plan.md or solution code. Stop after the draft is complete.""",
    "plan": """Review CLAUDE.md, TASK.md, the task files, and docs/draft.md. Create docs/plan.md containing an executable, sequential KDA optimization plan, candidate lineage strategy, correctness checks, performance hypotheses, stopping criteria, and evidence format. Do not implement or evaluate a candidate in this turn. Stop after the plan is complete.""",
    "candidate": """Continue the existing KDA task. Read CLAUDE.md, TASK.md, docs/draft.md, docs/plan.md, candidates.jsonl, and existing candidate artifacts. If the search has genuinely converged and no justified next candidate remains, create SEARCH_COMPLETE containing the reason and stop without evaluating. Otherwise implement exactly one next immutable Triton candidate in solution/solution.py, evaluate it exactly once with ./scripts/evaluate_candidate.sh feedback <candidate-id>, append its complete evidence record to candidates.jsonl, update the plan or decision notes if needed, and then stop. Do not run final evaluation. Never use a Torch, CPU, NumPy, or alternate computational fallback. If your evaluation is invalidated by an external GPU process (return code 3, "timing invalidated"), do not evaluate again in this turn: record the invalidated result and stop immediately; the next turn will re-evaluate. When invoking the evaluation launcher, use the bare command exactly as ./scripts/evaluate_candidate.sh feedback <candidate-id> with no extra shell syntax (no "; echo", no "&&", no environment prefixes) — compound commands are rejected by the permission system and the evaluation will never run.""",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def usage_summary(log_root: Path, output: Path) -> dict:
    files = sorted(log_root.glob("*.jsonl")) if log_root.is_dir() else []
    if not files:
        return {"usage": {"budget_tokens": 0}}
    completed = subprocess.run(
        [sys.executable, str(SUMMARIZER), *map(str, files), "--output", str(output)],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(completed.stdout)


def sync_retry_marker(control: Path, summary: dict) -> None:
    marker = control / "RETRY_RERUN_RECOMMENDED.json"
    transport = summary.get("transport") or {}
    if transport.get("rerun_recommended"):
        write_json(marker, {
            "reason": "excessive_api_retries",
            "api_retries": int(transport.get("api_retries", 0)),
            "retry_sequences": int(transport.get("retry_sequences", 0)),
            "max_retry_attempt": int(transport.get("max_retry_attempt", 0)),
            "retry_token_overhead_known": False,
            "recommendation": "review artifacts and rerun the affected stage with a fresh session if its token use or output is abnormal",
        })
    elif marker.exists():
        marker.unlink()


def validate_stage(workspace: Path, stage: str) -> None:
    if stage == "plan" and not (workspace / "docs/draft.md").is_file():
        raise SystemExit("docs/draft.md is required before the plan stage")
    if stage == "candidate" and not (workspace / "docs/plan.md").is_file():
        raise SystemExit("docs/plan.md is required before a candidate stage")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=sorted(PROMPTS))
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--max-turns", type=int, default=80)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    if not workspace.is_relative_to(RUN_ROOT.resolve()):
        raise SystemExit(f"workspace must be under {RUN_ROOT}")
    control = CONTROL_ROOT / workspace.name
    task_config = read_json(control / "task.json")
    validate_stage(workspace, args.stage)

    log_root = control / "claude"
    log_root.mkdir(parents=True, exist_ok=True)
    summary_path = control / "observability.json"
    before = usage_summary(log_root, summary_path)
    sync_retry_marker(control, before)
    budget_tokens = int((before.get("usage") or {}).get("budget_tokens", 0))
    soft_limit = int(task_config["token_soft_limit"])
    grace_limit = int(task_config.get("token_grace_limit", task_config["token_hard_limit"]))
    hard_limit = int(task_config["token_hard_limit"])
    if budget_tokens >= hard_limit:
        (workspace / "TOKEN_LIMIT_REACHED").write_text(
            f"hard token limit reached before {args.stage}: {budget_tokens} >= {hard_limit}\n",
            encoding="utf-8",
        )
        print(f"hard token limit reached: {budget_tokens} >= {hard_limit}")
        return 0
    if budget_tokens >= grace_limit:
        (workspace / "TOKEN_LIMIT_REACHED").write_text(
            f"normal completion token limit reached before {args.stage}: "
            f"{budget_tokens} >= {grace_limit}\n",
            encoding="utf-8",
        )
        print(f"normal completion token limit reached: {budget_tokens} >= {grace_limit}")
        return 0
    if args.stage == "candidate" and budget_tokens >= soft_limit:
        (workspace / "TOKEN_LIMIT_REACHED").write_text(
            f"soft token limit reached before candidate: {budget_tokens} >= {soft_limit}\n",
            encoding="utf-8",
        )
        print(f"soft token limit reached; refusing a new candidate: {budget_tokens} >= {soft_limit}")
        return 0

    target = workspace / ("docs/draft.md" if args.stage == "draft" else "docs/plan.md")
    if args.stage in {"draft", "plan"} and target.exists() and not args.force:
        raise SystemExit(f"stage artifact already exists: {target}; use --force to rerun")

    session_path = workspace / ".claude-session-id"
    if session_path.is_file():
        session_id = session_path.read_text(encoding="utf-8").strip()
        session_args = ["--resume", session_id]
    else:
        session_id = str(uuid.uuid4())
        session_path.write_text(session_id + "\n", encoding="utf-8")
        session_args = ["--session-id", session_id]

    sequence = len(list(log_root.glob("*.jsonl"))) + 1
    log_path = log_root / f"{sequence:04d}-{args.stage}.jsonl"
    # 工具白名单：文件读写编辑检索、KernelWiki 与 ncu-report-skill 两个 skill；
    # Bash 仅放行评测入口脚本和剖析启动器（剖析经 ncu_profile.sh 做选卡与占卡协调，
    # agent 不能直接运行 python/CUDA/裸 ncu，GPU 访问只存在于受控形式下）。
    command = [
        str(CLAUDE),
        "-p",
        PROMPTS[args.stage],
        *session_args,
        "--model", "Claude-Opus-4.8",
        "--effort", "xhigh",
        "--max-turns", str(args.max_turns),
        "--autocompact", "auto",
        "--output-format", "stream-json",
        "--verbose",
        "--tools", "Read,Write,Edit,Glob,Grep,Skill,Bash",
        "--allowedTools", "Read", "Write", "Edit", "Glob", "Grep", "Skill(KernelWiki)",
        "Skill(ncu-report-skill)",
        "Bash(./scripts/evaluate_candidate.sh *)",
        "Bash(./scripts/ncu_profile.sh *)",
        "--permission-mode", "dontAsk",
        "--strict-mcp-config",
        "--mcp-config", '{"mcpServers":{}}',
        "--no-chrome",
    ]
    env = os.environ.copy()

    started_at = datetime.now().astimezone().isoformat()
    state_path = control / "state.json"
    candidate_count_before = int(read_json(state_path).get("candidate_evaluations", 0))
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=workspace,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="")
        returncode = process.wait()

    after = usage_summary(log_root, summary_path)
    sync_retry_marker(control, after)
    final_tokens = int((after.get("usage") or {}).get("budget_tokens", 0))
    candidate_count_after = int(read_json(state_path).get("candidate_evaluations", 0))
    artifact_ok = True
    artifact_error = None
    if args.stage in {"draft", "plan"}:
        artifact_ok = target.is_file() and target.stat().st_size > 0
        if not artifact_ok:
            artifact_error = f"missing or empty stage artifact: {target}"
    elif not (workspace / "SEARCH_COMPLETE").is_file():
        artifact_ok = candidate_count_after == candidate_count_before + 1
        if not artifact_ok:
            artifact_error = (
                "candidate stage did not complete exactly one trusted evaluation: "
                f"before={candidate_count_before}, after={candidate_count_after}"
            )
    if final_tokens >= hard_limit:
        (workspace / "TOKEN_LIMIT_REACHED").write_text(
            f"absolute token limit reached after atomic stage: {final_tokens} >= {hard_limit}\n",
            encoding="utf-8",
        )
    elif final_tokens >= grace_limit:
        (workspace / "TOKEN_LIMIT_REACHED").write_text(
            f"normal completion token limit reached after atomic stage: "
            f"{final_tokens} >= {grace_limit}\n",
            encoding="utf-8",
        )
    status = {
        "schema": "kda-claude-stage-v1",
        "stage": args.stage,
        "session_id": session_id,
        "started_at": started_at,
        "completed_at": datetime.now().astimezone().isoformat(),
        "returncode": returncode,
        "log": str(log_path),
        "budget_tokens_before": budget_tokens,
        "budget_tokens_after": final_tokens,
        "token_soft_limit": soft_limit,
        "token_grace_limit": grace_limit,
        "token_hard_limit": hard_limit,
        "soft_limit_reached": final_tokens >= soft_limit,
        "grace_limit_reached": final_tokens >= grace_limit,
        "hard_limit_reached": final_tokens >= hard_limit,
        "artifact_ok": artifact_ok,
        "artifact_error": artifact_error,
    }
    write_json(control / "last_claude_stage.json", status)
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0 if artifact_ok else (returncode or 4)


if __name__ == "__main__":
    raise SystemExit(main())
