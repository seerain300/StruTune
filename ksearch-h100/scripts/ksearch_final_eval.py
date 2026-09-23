#!/usr/bin/env python3
"""K-Search 最终全量评测编排：FlashInfer → unified_eval.py；SOL → 官方 CLI。

用法（mtmc env）:
    python scripts/ksearch_final_eval.py --task rmsnorm_h4096 --run-dir baseline/ksearch/rmsnorm_h4096/run_seed0
    python scripts/ksearch_final_eval.py --task L1/053_gaussian_topk_sparse_activation --run-dir baseline/ksearch-sol-execbench/053_.../run_seed0

流程:
  1. 从 <run-dir>/ksearch-artifacts/<def>/solutions/<def>/*.json 取 mtime 最新的
     solution（即 generator 最终返回并保存的 best/last 候选）。
  2. FlashInfer 题：调 scripts/unified_eval.py（内部走 ziming 的 evaluate.py，
     warmup3/iters100/trials5 全量）。
  3. SOL 题：转换为 SOL schema solution.json，用 SOL 官方 venv python 跑 ziming
     的 evaluate_sol.py --rerun --iterations 100（全量 workloads，eval seed 200）。
  4. 两种路径都把 token 汇总（input/cached/output/reasoning）写入 candidate.json。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

WS = Path("/home/ziming/ksearch_h100_portable")
SOL_ROOT = Path("/home/ziming/dataset/SOL-ExecBench")
# 本地副本优先（原因见 evaluators/PROVENANCE.txt；与 ziming test/scripts 版逐字节相同）
ZIMING_EVAL_SOL = WS / "evaluators/evaluate_sol.py"
if not ZIMING_EVAL_SOL.is_file():
    ZIMING_EVAL_SOL = Path(
        "/home/ziming/MTMC-baseline/test/scripts/evaluate_sol.py"
    )


def latest_solution(run_dir: Path, def_name: str) -> Path:
    pattern = run_dir / "ksearch-artifacts" / def_name / "solutions" / def_name / "*.json"
    candidates = sorted(pattern.parent.glob("*.json"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"no saved solutions under {pattern.parent}")
    return candidates[-1]


def aggregate_usage(run_dir: Path):
    usage_path = run_dir / "usage.jsonl"
    if not usage_path.exists():
        return None
    calls = pin = pcached = pout = preasoning = 0
    for line in usage_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            calls += 1
            pin += int(rec.get("input_tokens", rec.get("prompt_tokens")) or 0)
            pcached += int(rec.get("input_cached_tokens") or 0)
            pout += int(rec.get("output_tokens", rec.get("completion_tokens")) or 0)
            preasoning += int(rec.get("reasoning_tokens") or 0)
        except Exception:
            continue
    return {
        "llm_calls": calls,
        "input_tokens": pin,
        "input_cached_tokens": pcached,
        "input_uncached_tokens": max(0, pin - pcached),
        "output_tokens": pout,
        "reasoning_tokens": preasoning,
        "total_tokens": pin + pout,
        "usage_log": str(usage_path),
    }


def to_sol_solution(ksearch_json: Path, definition: dict) -> dict:
    """K-Search solution JSON → SOL schema（与 sol_execbench_adapter 同规则）。"""
    obj = json.loads(ksearch_json.read_text())
    spec = obj.get("spec") or {}
    lang = str(spec.get("language") or "").strip().lower()
    lang_map = {"triton": "triton", "python": "pytorch", "pytorch": "pytorch"}
    sol_lang = lang_map.get(lang)
    if sol_lang is None:
        raise ValueError(f"unsupported language for SOL eval: {lang!r}")
    deps = [str(d) for d in (spec.get("dependencies") or [])]
    if sol_lang == "triton":
        deps = list(dict.fromkeys([*deps, "torch", "triton"]))
    else:
        deps = list(dict.fromkeys([*deps, "torch"]))
    return {
        "name": f"{obj.get('name')}_sol_execbench",
        "definition": definition["name"],
        "author": "ksearch",
        "description": (obj.get("description") or "") + " (final evaluation)",
        "spec": {
            "languages": [sol_lang],
            "target_hardware": ["LOCAL"],
            "entry_point": str(spec.get("entry_point") or "main.py::run"),
            "dependencies": deps,
            "destination_passing_style": bool(spec.get("destination_passing_style", False)),
            "binding": None,
        },
        "sources": obj.get("sources") or [],
    }


def run_flashinfer_eval(task: str, run_dir: Path, sol_json: Path, out_dir: Path, trials: int,
                        iters: int = 100, warmup: int = 3, ref_cache: str = "") -> int:
    cmd = [
        sys.executable, str(WS / "scripts" / "unified_eval.py"),
        "--task", task,
        "--ksearch-json", str(sol_json),
        "--run-dir", str(run_dir),
        "--method", "ksearch",
        "--trials", str(trials),
        "--iters", str(iters),
        "--warmup", str(warmup),
    ]
    if ref_cache:
        cmd += ["--ref-cache", ref_cache]
    print("[final_eval]", " ".join(cmd), flush=True)
    return subprocess.run(cmd).returncode


def run_sol_eval(task: str, run_dir: Path, sol_json: Path, out_dir: Path, iterations: int, timeout: int) -> int:
    problem_name = task.split("/")[-1]
    problem_dir = SOL_ROOT / "data" / "benchmark" / task if "/" in task else None
    if problem_dir is None:
        # 裸名字时在 benchmark 子集里找
        for subset in sorted((SOL_ROOT / "data" / "benchmark").iterdir()):
            cand = subset / task
            if (cand / "definition.json").is_file():
                problem_dir = cand
                break
    if problem_dir is None or not (problem_dir / "definition.json").is_file():
        raise FileNotFoundError(f"SOL problem dir for {task!r} not found")

    definition = json.loads((problem_dir / "definition.json").read_text())
    if definition.get("hf_id") == "":
        definition.pop("hf_id", None)
    sol_solution = to_sol_solution(sol_json, definition)
    out_dir.mkdir(parents=True, exist_ok=True)
    staged = out_dir / "solution.sol.json"
    staged.write_text(json.dumps(sol_solution, indent=2), encoding="utf-8")

    cmd = [
        str(SOL_ROOT / ".venv/bin/python"), str(ZIMING_EVAL_SOL),
        "--definition", str(problem_dir / "definition.json"),
        "--workload", str(problem_dir / "workload.jsonl"),
        "--solution", str(staged),
        "--output", str(out_dir / "performance.json"),
        "--rerun",
        "--iterations", str(iterations),
        "--timeout", str(timeout),
    ]
    print("[final_eval]", " ".join(cmd), flush=True)
    t0 = time.time()
    ret = subprocess.run(cmd).returncode
    wall = time.time() - t0

    usage = aggregate_usage(run_dir) or {}
    candidate = {
        "method": "ksearch",
        "task": task,
        "problem_dir": str(problem_dir),
        "run_dir": str(run_dir),
        "solution_json": str(sol_json),
        "eval_returncode": ret,
        "eval_wall_clock_s": round(wall, 2),
        "eval_cmd": " ".join(cmd),
    }
    candidate.update(usage)
    (out_dir / "candidate.json").write_text(
        json.dumps(candidate, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return ret


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="definition 名；SOL 题形如 L1/053_...")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--iterations", type=int, default=100, help="SOL 评测 timing iterations")
    ap.add_argument("--trials", type=int, default=1, help="FlashInfer 评测独立试验组数（v0.5 口径=1）")
    ap.add_argument("--warmup", type=int, default=3, help="warmup 次数（FI 透传）")
    ap.add_argument("--ref-cache", default="", help="参考延迟缓存 JSON（FI 按题可选）")
    ap.add_argument("--timeout", type=int, default=1800, help="SOL 评测子进程超时(秒)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    task = args.task
    def_name = task.split("/")[-1]
    out_dir = run_dir / "unified"

    sol_json = latest_solution(run_dir, def_name)
    print(f"[final_eval] task={task} solution={sol_json.name}")

    if "/" in task:
        ret = run_sol_eval(task, run_dir, sol_json, out_dir, args.iterations, args.timeout)
    else:
        ret = run_flashinfer_eval(task, run_dir, sol_json, out_dir, args.trials,
                                  iters=args.iterations, warmup=args.warmup,
                                  ref_cache=getattr(args, "ref_cache", ""))
    print(f"[final_eval] done rc={ret} -> {out_dir}")
    return ret


if __name__ == "__main__":
    raise SystemExit(main())
