# KDA Formal Task Instructions

Follow the complete KDA draft → executable plan → sequential candidate → evidence → decision workflow.

## Isolation

- Work only inside this task workspace.
- Do not inspect parent directories, other task workspaces, K-Search, DRTriton, baselines, archived experiments, evaluator implementations, controller code, or full workload files.
- The permitted external knowledge sources are the installed `KernelWiki` and `ncu-report-skill` skills.
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

- The full feedback workload set together counts as one candidate evaluation.
- Any meaningful source, configuration, or launch change requires a new candidate ID.
- Never reuse a candidate ID for changed source.
- Do not change the fixed feedback workloads.
- Do not directly run CUDA, `nvidia-smi`, the external evaluator, or any alternate correctness harness.
- Performance profiling is allowed only through the `ncu-report-skill` workflow, using the workspace launcher `./scripts/ncu_profile.sh <ncu arguments...>` (for example `./scripts/ncu_profile.sh --set basic -o profile/r1 python harness.py`). The launcher automatically picks an idle whitelisted GPU, temporarily releases the card-occupancy holder, and points `python` at the correct torch/triton interpreter for this benchmark. Do not invoke `ncu` directly.
- Profiling and evaluation must never run at the same time. An evaluation times the kernel on an exclusively locked GPU; any other process that appears on that GPU during timing — including your own profiler — makes the controller discard the measurement (return code 3) and consumes one evaluation from the budget for nothing. Finish one before starting the other, and never launch profiling in the background while an evaluation is running.
- A failed Triton implementation is invalid; never replace it with a Torch/CPU/NumPy fallback.
