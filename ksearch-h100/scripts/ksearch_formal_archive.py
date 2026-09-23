#!/usr/bin/env python3
"""正式实验归档：代码版本、配置快照、token 汇总、评测指标 → 独立文件夹。

用法（正式批次跑完后）:
    python scripts/ksearch_formal_archive.py --tag formal_20260914 [--with-eval]

产物结构:
    <WS>/experiments_archive/<tag>/
      manifest.json          # 任务清单、参数、wall time、DONE 状态
      versions.json          # K-Search commit+本地diff、mtmc/SOL 环境版本、GPU/驱动
      config/                # 入口脚本与关键源码快照
      usage/<task>.json      # 每题 llm-usage-summary（input/cached/output/reasoning）
      usage/total.json       # 20 题总计
      eval/<task>/           # --with-eval 时拷贝 unified/evaluation.json、
                             # performance.json、candidate.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

WS = Path("/home/ziming/ksearch_h100_portable")
KSEARCH = WS / "K-Search"
FLASHINFER_TASKS = [
    "dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64",
    "gdn_decode_qk4_v8_d128_k_last",
    "gdn_prefill_qk4_v8_d128_k_last",
    "gemm_n4096_k4096",
    "gqa_paged_decode_h32_kv8_d128_ps1",
    "gqa_paged_prefill_causal_h32_kv8_d128_ps1",
    "gqa_ragged_prefill_causal_h32_kv8_d128",
    "mla_paged_decode_h16_ckv512_kpe64_ps1",
    "mla_paged_prefill_causal_h16_ckv512_kpe64_ps1",
    "rmsnorm_h4096",
]
SOL_TASKS = [
    "002_vae_conv3x3_groupnorm_silu_residual_fused",
    "005_conv_gated_projection_with_causal_conv",
    "007_hyena_fft_size_padding_rfft",
    "008_expert_output_weighted_index_add_accumulation",
    "018_fused_rope_with_qk_norm_and_kv_cache_update",
    "020_vision_patch_merger_spatial_shuffle_mlp",
    "053_gaussian_topk_sparse_activation",
    "058_moe_expert_token_radix_sort_with_prefix_sum",
    "070_mamba2_fused_intra_chunk_diagonal_computation",
    "094_time_decay_exponential_stabilization",
]

CONFIG_FILES = [
    WS / "ksearch-run.sh",
    WS / "ksearch-token-run.py",
    WS / "scripts" / "ksearch_campaign.sh",
    WS / "scripts" / "ksearch_final_eval.py",
    WS / "scripts" / "ksearch_formal_archive.py",
    WS / "scripts" / "ksearch_sol_validate_all.py",
    KSEARCH / "generate_kernels_and_eval.py",
    KSEARCH / "k_search" / "tasks" / "sol_execbench_task.py",
    WS / "protocol.md",
    WS / "experiment_manifest.json",
]


def run_dir_for(task: str, tag: str, seed: int = 0) -> Path:
    if task in SOL_TASKS or not (WS / "baseline" / "ksearch" / "experiments" / tag / task).exists():
        base = WS / "baseline" / "ksearch-sol-execbench"
    else:
        base = WS / "baseline" / "ksearch"
    return base / "experiments" / tag / task / f"run_seed{seed}"


def usage_summary(run_dir: Path) -> dict | None:
    usage_path = run_dir / "usage.jsonl"
    if not usage_path.exists():
        return None
    calls = pin = pcache = pout = preas = 0
    for line in usage_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        calls += 1
        pin += r.get("input_tokens") or 0
        pcache += r.get("input_cached_tokens") or 0
        pout += r.get("output_tokens") or 0
        preas += r.get("reasoning_tokens") or 0
    return {
        "llm_calls": calls,
        "input_tokens": pin,
        "input_cached_tokens": pcache,
        "input_uncached_tokens": max(0, pin - pcache),
        "output_tokens": pout,
        "reasoning_tokens": preas,
        "total_tokens": pin + pout,
        "usage_log": str(usage_path),
    }


def versions_snapshot() -> dict:
    def sh(cmd: list[str], cwd: Path | None = None) -> str:
        try:
            return subprocess.run(
                cmd, capture_output=True, text=True, cwd=str(cwd) if cwd else None, timeout=60
            ).stdout.strip()
        except Exception as e:
            return f"<error: {e}>"

    ks_commit = sh(["git", "rev-parse", "HEAD"], cwd=KSEARCH)
    ks_status = sh(["git", "status", "--porcelain"], cwd=KSEARCH)
    ks_diff = sh(["git", "diff"], cwd=KSEARCH)
    import hashlib

    diff_hash = hashlib.sha256(ks_diff.encode()).hexdigest()[:16] if ks_diff else None

    mtmc_py = "/home/ziming/miniconda3/envs/ksearch/bin/python"
    mtmc_ver = sh([mtmc_py, "-c",
                   "import torch, triton, sys; print(sys.version.split()[0], torch.__version__, triton.__version__)"])
    sol_py = "/home/ziming/dataset/SOL-ExecBench/.venv/bin/python"
    sol_ver = sh([sol_py, "-c",
                  "import torch, triton, sys; print(sys.version.split()[0], torch.__version__, triton.__version__)"])
    gpu = sh(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])

    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "ksearch": {
            "repo": str(KSEARCH),
            "commit": ks_commit,
            "dirty_files": [l for l in ks_status.splitlines() if l.strip()],
            "local_diff_sha256_16": diff_hash,
        },
        "mtmc_env": {"python_bin": mtmc_py, "versions(python torch triton)": mtmc_ver},
        "sol_env": {"python_bin": sol_py, "versions(python torch triton)": sol_ver},
        "gpu": {"query": "name,driver_version", "result": gpu},
        "evaluator_flashinfer": str(
            WS / ".." / "ziming" / "MTMC-baseline" / "agent-generation" / "scripts" / "evaluate.py"
        ),
        "evaluator_sol": str(
            WS / ".." / "ziming" / "MTMC-baseline" / "test" / "scripts" / "evaluate_sol.py"
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="正式批次 tag，如 formal_20260914")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--with-eval", action="store_true", help="同时拷贝最终评测产物（unified/）")
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--wm-max-action-nodes", type=int, default=20)
    ap.add_argument("--wm-max-attempts-per-node", type=int, default=5)
    args = ap.parse_args()

    out_root = WS / "experiments_archive" / args.tag
    (out_root / "config").mkdir(parents=True, exist_ok=True)
    (out_root / "usage").mkdir(parents=True, exist_ok=True)
    (out_root / "eval").mkdir(exist_ok=True)

    # 1. 配置快照
    for f in CONFIG_FILES:
        if f.is_file():
            shutil.copy2(f, out_root / "config" / f.name)

    # 2. 版本指纹
    versions = versions_snapshot()
    (out_root / "versions.json").write_text(json.dumps(versions, indent=2, ensure_ascii=False))

    # 3. manifest + usage
    tasks_meta = []
    total = {"llm_calls": 0, "input_tokens": 0, "input_cached_tokens": 0,
             "input_uncached_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0,
             "total_tokens": 0}
    for task in FLASHINFER_TASKS + SOL_TASKS:
        rd = run_dir_for(task, args.tag, args.seed)
        meta = {
            "task": task,
            "bench": "sol" if task in SOL_TASKS else "flashinfer",
            "run_dir": str(rd),
            "done": (rd / "DONE").exists(),
            "exit_code": None,
        }
        ec_file = rd / "exit_code"
        if ec_file.exists():
            meta["exit_code"] = ec_file.read_text().strip()
        log = rd / "campaign_stdout.log"
        if log.exists():
            meta["campaign_log"] = str(log)
        u = usage_summary(rd)
        if u:
            meta["usage"] = u
            (out_root / "usage" / f"{task}.json").write_text(json.dumps(u, indent=2))
            for k in total:
                total[k] += u.get(k, 0)
        if args.with_eval:
            unified = rd / "unified"
            if unified.is_dir():
                dest = out_root / "eval" / task
                dest.mkdir(exist_ok=True)
                for name in ("evaluation.json", "performance.json", "candidate.json"):
                    src = unified / name
                    if src.is_file():
                        shutil.copy2(src, dest / name)
                meta["eval_files"] = str(dest)
        tasks_meta.append(meta)

    (out_root / "usage" / "total.json").write_text(json.dumps(total, indent=2))
    manifest = {
        "schema": "ksearch-formal-archive-v1",
        "tag": args.tag,
        "seed": args.seed,
        "budget": {
            "max_opt_rounds": args.rounds,
            "wm_max_action_nodes": args.wm_max_action_nodes,
            "wm_max_attempts_per_node": args.wm_max_attempts_per_node,
            "wm_stagnation_window": 5,
        },
        "feedback_eval": {
            "workloads": "seeded sample of 5",
            "flashinfer": "warmup 3 / iterations 100 / trials 5",
            "sol": "warmup 10 / iterations 100 (official CLI, eval seed 200)",
        },
        "final_eval": {
            "flashinfer": "unified_eval.py -> ziming evaluate.py, warmup3/iters100/trials5, all workloads",
            "sol": "ziming evaluate_sol.py --rerun --iterations 100, all workloads",
        },
        "tasks": tasks_meta,
        "token_total": total,
        "archived_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"[archive] -> {out_root}")
    print(f"[archive] tasks={len(tasks_meta)} done={sum(1 for t in tasks_meta if t['done'])}")
    print(f"[archive] tokens: calls={total['llm_calls']} in={total['input_tokens']} "
          f"cached={total['input_cached_tokens']} out={total['output_tokens']} "
          f"reasoning={total['reasoning_tokens']} total={total['total_tokens']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
