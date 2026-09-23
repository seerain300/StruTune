#!/usr/bin/env python3
"""Prepare isolated, auditable KDA workspaces from the frozen manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
from datetime import datetime
from pathlib import Path


ROOT = Path("/data1/workspace/weihongren")
MANIFEST = ROOT / "experiment_manifest.json"
RUN_ROOT = ROOT / "kda-runs"
CONTROL_ROOT = ROOT / "kda-control"
FLASHINFER_DATASET = ROOT / "dataset/flashinfer-test"
SOL_DATASET = Path("/data1/workspace/ziming/dataset/SOL-ExecBench/data/benchmark")
FLASHINFER_EVALUATOR = ROOT / "evaluators/evaluate.py"
SOL_EVALUATOR = ROOT / "evaluators/evaluate_sol.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")


def unique_match(root: Path, pattern: str, label: str) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {label} for {pattern}, found {len(matches)}")
    return matches[0]


def task_contract(benchmark: str, task: str, workload_count: int) -> str:
    benchmark_name = "FlashInfer" if benchmark == "flashinfer" else "SOL-ExecBench"
    return f"""# Task Contract

- Task: optimize official {benchmark_name} task `{task}` on NVIDIA A800 (`sm_80`).
- Read `task/definition.json` for the exact signature, reference implementation, dtypes, shapes, and tolerances.
- Submission: `solution/solution.py` exposing the required `run(...)` entry point.
- Primary implementation must use Triton. PyTorch is allowed only for tensor metadata and launch plumbing.
- No Torch computational fallback, CPU/NumPy fallback, CUDA-extension fallback, or alternate implementation fallback.
- Feedback evaluation runs the FULL official workload set (coarse: warmup 2 / 10 iterations) from `task/feedback_workloads.jsonl`, so every shape — including boundary cases — is checked during the search.
- One immutable kernel version over all five feedback workloads counts as one candidate evaluation.
- Candidate budget: 100 evaluations. Token soft limit: 1,000,000; normal completion limit: 1,500,000; absolute limit: 1,650,000.
- Token budget includes uncached input, cache creation input, cache-read input, and output tokens.
- Final evaluation: one full {workload_count}-workload evaluation for the best valid candidate, only after operator approval.
- Primary ranking metric: geometric mean speedup; every selected workload must pass correctness.
"""


def claude_instructions() -> str:
    return """# KDA Formal Task Instructions

Follow the complete KDA draft → executable plan → sequential candidate → evidence → decision workflow.

## Isolation

- Work only inside this task workspace.
- Do not inspect parent directories, other task workspaces, K-Search, DRTriton, baselines, archived experiments, evaluator implementations, controller code, or full workload files.
- The only permitted external knowledge source is the installed `KernelWiki` skill.
- Do not modify or copy the evaluator, dataset, controller, launcher, shared configuration, or evaluation script.
- Do not invoke Humanize, RLCR, Codex, Gemini, subagents, MCP tools, web search, or external agents.

## Required workflow

1. Read `TASK.md`, `README.md`, `task/definition.json`, and `task/feedback_workloads.jsonl`.
2. Write `docs/draft.md`; do not create code before the draft is complete.
3. Write an executable optimization plan to `docs/plan.md` before implementing candidates.
4. Implement immutable candidates `c001`, `c002`, ... sequentially, one source version at a time.
5. Evaluate only with `./scripts/evaluate_candidate.sh feedback <candidate-id>`.
6. Append one complete JSON object per evaluated candidate to `candidates.jsonl`; never rewrite earlier records.
7. Record parent, source hash, hypothesis, validation, per-workload result, geomean, decision, cumulative evaluation count, and skill usage.
8. Stop at the token/evaluation budget or when improvement has converged.
9. When improvement has genuinely converged, create `SEARCH_COMPLETE` with the reason.
10. Never run `final` without explicit operator approval.

## Evaluation rules

- Five fixed workloads together count as one candidate evaluation.
- Any meaningful source, configuration, or launch change requires a new candidate ID.
- Never reuse a candidate ID for changed source.
- Do not change the fixed feedback workloads.
- Do not directly run CUDA, a profiler, `nvidia-smi`, the external evaluator, or any alternate correctness harness.
- A failed Triton implementation is invalid; never replace it with a Torch/CPU/NumPy fallback.
"""


def readme(run_id: str, benchmark: str, task: str) -> str:
    return f"""# Isolated KDA Task: {task}

Run ID: `{run_id}`  
Benchmark: `{benchmark}`

This workspace intentionally exposes only the official definition, five fixed feedback workloads,
the candidate source location, and the trusted evaluation launcher. It contains no prior optimized solution.

```bash
./scripts/evaluate_candidate.sh feedback c001
```

The trusted controller selects and locks an empty GPU, rejects foreign-process interference,
checks immutable candidate IDs and hashes, and invokes the benchmark-specific official evaluator.

Final full evaluation is operator-only:

```bash
./scripts/evaluate_candidate.sh final <candidate-id>
```
"""


def make_task_config(
    *,
    run_id: str,
    benchmark: str,
    task: str,
    definition: Path,
    workload: Path,
    feedback_records: list[dict],
    feedback_indices: list[int],
) -> dict:
    dataset = FLASHINFER_DATASET if benchmark == "flashinfer" else SOL_DATASET
    evaluator = FLASHINFER_EVALUATOR if benchmark == "flashinfer" else SOL_EVALUATOR
    evaluation = {
        "feedback": {
            "warmup": 2,
            "iterations": 10,
            "trials": 1 if benchmark == "flashinfer" else None,
            "eval_seed": None if benchmark == "flashinfer" else 200,
            "timeout_base_seconds": 1200 if benchmark == "sol_execbench" else None,
            "timeout_per_workload_seconds": 1800 if benchmark == "sol_execbench" else None,
            "reference_cache": benchmark == "sol_execbench",
        },
        "final": {
            "warmup": 3 if benchmark == "flashinfer" else 10,
            "iterations": 100,
            "trials": 1 if benchmark == "flashinfer" else None,
            "eval_seed": None if benchmark == "flashinfer" else 200,
            "timeout_base_seconds": 1200 if benchmark == "sol_execbench" else None,
            "timeout_per_workload_seconds": 1800 if benchmark == "sol_execbench" else None,
            "reference_cache": False,
        },
    }
    return {
        "schema": "kda-trusted-task-config-v2",
        "run_id": run_id,
        "task": task,
        "benchmark": benchmark,
        "dataset_root": str(dataset),
        "definition_relpath": str(definition.relative_to(dataset)),
        "workload_relpath": str(workload.relative_to(dataset)),
        "definition_sha256": sha256(definition),
        "workload_sha256": sha256(workload),
        "evaluator_sha256": sha256(evaluator),
        "workload_count": len(load_jsonl(workload)),
        "feedback_strategy": "full-workload-coarse (v2, 2026-09-18): all indices, warmup2/iters10; final remains the precise 100-iteration measurement",
        "feedback_seed": 0,
        "feedback_indices": feedback_indices,
        "feedback_uuids": [
            str((record.get("workload") or record).get("uuid")) for record in feedback_records
        ],
        "candidate_evaluation_budget": 100,
        "final_evaluation_budget": 1,
        "token_soft_limit": 1_000_000,
        "token_grace_limit": 1_500_000,
        "token_hard_limit": 1_650_000,
        "token_definition": "input + cache_creation_input + cache_read_input + output",
        "evaluation": evaluation,
    }


def prepare_one(
    *,
    tag: str,
    benchmark: str,
    task: str,
    definition: Path,
    workload: Path,
    force: bool,
) -> dict:
    run_id = f"{slug(tag)}--{slug(benchmark)}--{slug(task)}"
    workspace = RUN_ROOT / run_id
    control = CONTROL_ROOT / run_id
    if (workspace.exists() or control.exists()) and not force:
        raise FileExistsError(f"run already exists; use a new tag or --force: {run_id}")
    if force:
        shutil.rmtree(workspace, ignore_errors=True)
        shutil.rmtree(control, ignore_errors=True)

    records = load_jsonl(workload)
    feedback_indices = list(range(len(records)))
    feedback_records = records

    workspace.joinpath("task").mkdir(parents=True)
    workspace.joinpath("solution").mkdir()
    workspace.joinpath("docs").mkdir()
    workspace.joinpath("runs/candidates").mkdir(parents=True)
    workspace.joinpath("scripts").mkdir()
    control.mkdir(parents=True)

    shutil.copy2(definition, workspace / "task/definition.json")
    write_jsonl(workspace / "task/feedback_workloads.jsonl", feedback_records)
    write_jsonl(control / "feedback_workloads.jsonl", feedback_records)

    config = make_task_config(
        run_id=run_id,
        benchmark=benchmark,
        task=task,
        definition=definition,
        workload=workload,
        feedback_records=feedback_records,
        feedback_indices=feedback_indices,
    )
    config["feedback_workload_sha256"] = sha256(control / "feedback_workloads.jsonl")
    write_json(control / "task.json", config)
    write_json(control / "state.json", {
        "schema": "kda-controller-state-v2",
        "run_id": run_id,
        "task": task,
        "benchmark": benchmark,
        "candidate_evaluations": 0,
        "final_evaluations": 0,
        "candidates": {},
    })
    (control / "state.lock").touch()

    (workspace / "CLAUDE.md").write_text(claude_instructions(), encoding="utf-8")
    (workspace / "TASK.md").write_text(task_contract(benchmark, task, len(records)), encoding="utf-8")
    (workspace / "README.md").write_text(readme(run_id, benchmark, task), encoding="utf-8")
    (workspace / "candidates.jsonl").touch()
    launcher = workspace / "scripts/evaluate_candidate.sh"
    launcher.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n\n"
        "stage=${1:?usage: evaluate_candidate.sh feedback|final cNNN}\n"
        "candidate=${2:?usage: evaluate_candidate.sh feedback|final cNNN}\n\n"
        f"export KDA_WORKSPACE={workspace}\n"
        f'exec {ROOT / "bin/kda-eval"} "$stage" --candidate "$candidate"\n',
        encoding="utf-8",
    )
    launcher.chmod(0o755)
    return {
        "run_id": run_id,
        "benchmark": benchmark,
        "task": task,
        "workspace": str(workspace),
        "control": str(control),
        "workload_count": len(records),
        "feedback_indices": feedback_indices,
        "feedback_uuids": config["feedback_uuids"],
    }


def manifest_tasks(manifest: dict) -> list[tuple[str, str, Path, Path]]:
    tasks = []
    for item in manifest["flashinfer_test"]["definitions"]:
        name = item["name"]
        definition = unique_match(FLASHINFER_DATASET, f"definitions/*/{name}.json", "definition")
        workload = unique_match(FLASHINFER_DATASET, f"workloads/*/{name}.jsonl", "workload")
        tasks.append(("flashinfer", name, definition, workload))
    for subset in ("L1", "L2"):
        for item in manifest["sol_execbench"]["subsets"][subset]["problems"]:
            relative = Path(item["path"])
            problem = SOL_DATASET / relative
            tasks.append(("sol_execbench", str(relative), problem / "definition.json", problem / "workload.jsonl"))
    return tasks


def validate_manifest_counts(manifest: dict, tasks: list[tuple[str, str, Path, Path]]) -> None:
    if len(tasks) != int(manifest["totals"]["definition_count"]):
        raise RuntimeError("manifest definition count mismatch")
    total_workloads = 0
    for _, task, definition, workload in tasks:
        if not definition.is_file() or not workload.is_file():
            raise FileNotFoundError(f"missing dataset files for {task}")
        total_workloads += len(load_jsonl(workload))
    if total_workloads != int(manifest["totals"]["workload_count"]):
        raise RuntimeError(
            f"manifest workload count mismatch: expected {manifest['totals']['workload_count']}, got {total_workloads}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True, help="Campaign tag included in every run ID.")
    parser.add_argument("--force", action="store_true", help="Replace workspaces with the same tag.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print tasks without writing.")
    args = parser.parse_args()

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    tasks = manifest_tasks(manifest)
    validate_manifest_counts(manifest, tasks)
    if args.dry_run:
        for benchmark, task, definition, workload in tasks:
            print(f"{benchmark}\t{task}\t{len(load_jsonl(workload))}\t{definition}")
        print(f"validated {len(tasks)} tasks / {manifest['totals']['workload_count']} workloads")
        return 0

    prepared = [
        prepare_one(
            tag=args.tag,
            benchmark=benchmark,
            task=task,
            definition=definition,
            workload=workload,
            force=args.force,
        )
        for benchmark, task, definition, workload in tasks
    ]
    campaign = {
        "schema": "kda-campaign-v1",
        "tag": args.tag,
        "created_at": datetime.now().astimezone().isoformat(),
        "manifest": str(MANIFEST),
        "manifest_sha256": sha256(MANIFEST),
        "task_count": len(prepared),
        "workload_count": sum(item["workload_count"] for item in prepared),
        "tasks": prepared,
    }
    campaign_path = CONTROL_ROOT / "campaigns" / f"{slug(args.tag)}.json"
    write_json(campaign_path, campaign)
    print(f"prepared {len(prepared)} isolated tasks")
    print(f"campaign: {campaign_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
