#!/usr/bin/env python3
"""无 GPU 校验：SOL-ExecBench 10 题全量 staging/schema 验证。

对每题构造 K-Search 候选（reference 源码）→ to_sol_solution → 写出
definition/workload(反馈抽样)/solution/config staging 文件 → 用 SOL 官方
venv 的 sol_execbench_validate.py（Definition/Solution/Workload/BenchmarkConfig
+ ProblemPackager）验证。不执行 GPU。
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "/data1/workspace/weihongren/K-Search")

from k_search.tasks.sol_execbench_task import SolExecBenchTask  # noqa: E402
from k_search.tasks.task_base import (  # noqa: E402
    BuildSpec,
    Solution,
    SourceFile,
    SupportedLanguages,
)

WS = Path("/data1/workspace/weihongren")
SOL_ROOT = Path("/data1/workspace/ziming/dataset/SOL-ExecBench")
VALIDATOR = WS / "scripts" / "sol_execbench_validate.py"

PROBLEMS = [
    "L1/002_vae_conv3x3_groupnorm_silu_residual_fused",
    "L1/005_conv_gated_projection_with_causal_conv",
    "L1/007_hyena_fft_size_padding_rfft",
    "L1/008_expert_output_weighted_index_add_accumulation",
    "L1/018_fused_rope_with_qk_norm_and_kv_cache_update",
    "L1/020_vision_patch_merger_spatial_shuffle_mlp",
    "L1/053_gaussian_topk_sparse_activation",
    "L1/058_moe_expert_token_radix_sort_with_prefix_sum",
    "L1/070_mamba2_fused_intra_chunk_diagonal_computation",
    "L1/094_time_decay_exponential_stabilization",
]


def main() -> int:
    failures = []
    with tempfile.TemporaryDirectory(prefix="ksearch_sol_validate_") as tmp:
        tmp_root = Path(tmp)
        for problem in PROBLEMS:
            task = SolExecBenchTask.from_cli_args(
                sol_root=str(SOL_ROOT),
                definition_name=problem,
                warmup_runs=10,
                iterations=10,
                feedback_workloads=None,
                num_feedback_workloads=5,
                artifacts_dir=None,
                seed=0,
            )
            candidate = Solution(
                name=f"validate_{task.name}",
                definition=task.name,
                author="ksearch",
                spec=BuildSpec(
                    language=SupportedLanguages.TRITON,
                    target_hardware=["A800"],
                    entry_point="main.py::run",
                    dependencies=[],
                    destination_passing_style=False,
                ),
                sources=[
                    SourceFile(
                        path="main.py",
                        content=str(task._definition.get("reference") or ""),
                    )
                ],
            )
            sol_solution = task.to_sol_solution(candidate)

            stage = tmp_root / task.name
            stage.mkdir(parents=True)
            (stage / "definition.json").write_text(
                json.dumps(task._definition, indent=2), encoding="utf-8"
            )
            (stage / "workload.jsonl").write_text(
                "".join(json.dumps(w) + "\n" for w in task._selected_workloads),
                encoding="utf-8",
            )
            (stage / "solution.json").write_text(
                json.dumps(sol_solution, indent=2), encoding="utf-8"
            )
            (stage / "config.json").write_text(
                json.dumps(
                    {
                        "warmup_runs": 10,
                        "iterations": 10,
                        "benchmark_reference": True,
                        "correctness_only": False,
                        "seed": 200,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            cmd = [
                str(SOL_ROOT / ".venv/bin/python"),
                str(VALIDATOR),
                "--definition", str(stage / "definition.json"),
                "--workload", str(stage / "workload.jsonl"),
                "--solution", str(stage / "solution.json"),
                "--config", str(stage / "config.json"),
                "--staging-dir", str(stage / "staging_check"),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(SOL_ROOT))
            status = "PASS" if result.returncode == 0 else "FAIL"
            print(f"[{status}] {task.name} (rc={result.returncode})", flush=True)
            if result.returncode != 0:
                failures.append(task.name)
                print(result.stderr.strip()[-1500:])

    print(f"\n{len(PROBLEMS) - len(failures)}/{len(PROBLEMS)} passed")
    if failures:
        print("failed:", failures)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
