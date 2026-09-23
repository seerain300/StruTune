#!/usr/bin/env python3
"""统一评测适配器：把 K-Search / DRTriton 产物接入 ziming 的统一 evaluator。

必须在 ksearch env 下运行（source activate-ksearch.sh）：
    python unified_eval.py --task rmsnorm_h4096 --ksearch-json <solution.json> --run-dir baseline/ksearch/rmsnorm_h4096/run_seed0
    python unified_eval.py --task rmsnorm_h4096 --solution solution.py --run-dir <dir> [--limit 3]

产出：
    <run-dir>/unified/solution.py      # 展开的候选源码
    <run-dir>/unified/evaluation.json  # 统一 evaluator 输出
    <run-dir>/unified/candidate.json   # 元数据（method/seed/token 汇总/评测指纹）
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

WS = Path("/home/ziming/ksearch_h100_portable")
DATASET = Path(os.environ.get("FIB_DATASET_PATH", "/home/ziming/dataset/flashinfer-test"))
# 本地副本优先（ziming 的 MTMC-baseline 目录会被他本人重构移动，2026-09-15 agent-generation 已被移除；
# test/scripts 版与原 agent-generation 版 diff 确认逐字节相同，指纹见 evaluators/PROVENANCE.txt）
EVAL_SCRIPT = WS / "evaluators/evaluate.py"
if not EVAL_SCRIPT.is_file():
    EVAL_SCRIPT = Path(
        "/home/ziming/MTMC-baseline/agent-generation/scripts/evaluate.py"
    )


def resolve_task(task: str):
    """按 definition 名在数据集 definitions/<category>/ 下定位文件。"""
    for cat_dir in sorted((DATASET / "definitions").iterdir()):
        if not cat_dir.is_dir():
            continue
        d = cat_dir / f"{task}.json"
        w = DATASET / "workloads" / cat_dir.name / f"{task}.jsonl"
        if d.exists() and w.exists():
            return d, w, cat_dir.name
    raise FileNotFoundError(f"task {task} not found under {DATASET}/definitions/*/")


def extract_ksearch_solution(json_path: Path, out_dir: Path):
    """K-Search TaskSolution JSON -> 展开 sources 文件；返回 entry 函数名。"""
    obj = json.loads(json_path.read_text())
    spec = obj.get("spec", obj)
    entry_point = str(spec.get("entry_point", "") or "")
    # entry_point 形如 "solution.py::run"
    entry_file = entry_point.split("::")[0] if "::" in entry_point else "solution.py"
    entry_fn = entry_point.split("::")[-1] if "::" in entry_point else "run"
    srcs = obj.get("sources") or []
    if not srcs:
        raise ValueError(f"no sources in {json_path}")
    out_dir.mkdir(parents=True, exist_ok=True)
    for sf in srcs:
        p = out_dir / str(sf.get("path") or "solution.py")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(str(sf.get("content") or ""))
    main_py = out_dir / entry_file
    if not main_py.exists():
        # entry 指向的文件缺失时退回第一个源文件
        main_py = out_dir / str(srcs[0].get("path") or "solution.py")
    meta = {
        "ksearch_solution_name": obj.get("name"),
        "ksearch_description": obj.get("description"),
        "entry_fn": entry_fn,
    }
    return main_py, entry_fn, meta


def aggregate_usage(run_dir: Path):
    usage_path = run_dir / "usage.jsonl"
    if not usage_path.exists():
        return None
    calls, pin, pcached, pout, preasoning = 0, 0, 0, 0, 0
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
            # reasoning_tokens 是 output_tokens 的子集；旧记录无此字段时按 0 汇总。
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="definition 名，如 rmsnorm_h4096")
    ap.add_argument("--ksearch-json", default=None, help="K-Search solution JSON 路径")
    ap.add_argument("--solution", default=None, help="直接给定 solution.py 路径（DRTriton 等）")
    ap.add_argument("--entry", default="run", help="solution 入口函数名（ksearch-json 时自动解析）")
    ap.add_argument("--run-dir", required=True, help="run_seedN 目录")
    ap.add_argument("--method", default=None, help="ksearch / drtriton（写入 candidate.json）")
    ap.add_argument("--limit", type=int, default=0, help="只评测前 N 个 workload（smoke 用）")
    ap.add_argument("--correctness-only", action="store_true")
    ap.add_argument("--trials", type=int, default=5, help="计时的独立试验组数（透传给统一 evaluator）")
    ap.add_argument("--warmup", type=int, default=3, help="warmup 运行次数（透传）")
    ap.add_argument("--iters", type=int, default=100, help="计时迭代次数（透传）")
    ap.add_argument("--ref-cache", default=None, help="参考延迟缓存 JSON（跳过实时参考计时）")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    def_path, wl_path, cat = resolve_task(args.task)
    run_dir = Path(args.run_dir).resolve()
    out_dir = run_dir / "unified"
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {}
    if args.ksearch_json:
        sol_py, entry, meta = extract_ksearch_solution(Path(args.ksearch_json), out_dir)
    elif args.solution:
        sol_py = Path(args.solution)
        entry = args.entry
    else:
        ap.error("需要 --ksearch-json 或 --solution")

    cmd = [
        sys.executable, str(EVAL_SCRIPT),
        "--definition", str(def_path),
        "--workload", str(wl_path),
        "--solution", str(sol_py),
        "--entry", entry,
        "--dataset-root", str(DATASET),
        "--device", args.device,
        "--json", str(out_dir / "evaluation.json"),
    ]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    if args.correctness_only:
        cmd += ["--correctness-only"]
    cmd += ["--trials", str(args.trials), "--warmup", str(args.warmup), "--iters", str(args.iters)]
    if args.ref_cache:
        cmd += ["--ref-cache", args.ref_cache]

    print("[unified_eval]", " ".join(cmd))
    t0 = time.time()
    ret = subprocess.run(cmd, cwd=str(out_dir))
    wall = time.time() - t0

    usage = aggregate_usage(run_dir)
    candidate = {
        "method": args.method or ("ksearch" if args.ksearch_json else "drtriton"),
        "task": args.task,
        "category": cat,
        "run_dir": str(run_dir),
        "solution_py": str(sol_py),
        "entry_fn": entry,
        "eval_returncode": ret.returncode,
        "eval_wall_clock_s": round(wall, 2),
        "eval_cmd": " ".join(cmd),
        "dataset_root": str(DATASET),
        "definition_file": str(def_path),
        "workload_file": str(wl_path),
    }
    candidate.update(meta or {})
    if usage:
        candidate.update(usage)
    (out_dir / "candidate.json").write_text(json.dumps(candidate, indent=2, ensure_ascii=False))
    print(f"[unified_eval] done rc={ret.returncode} wall={wall:.1f}s -> {out_dir}")
    sys.exit(ret.returncode)


if __name__ == "__main__":
    main()
