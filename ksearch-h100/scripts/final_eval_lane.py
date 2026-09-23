#!/usr/bin/env python3
"""终评专用道（2026-09-21）：整卡专用，不再与搜索穿插。
用法: final_eval_lane.py <card> <fi|l1>
- 每 5 分钟扫描对应组：DONE 且无终评产物 → 守卫审计 → 干净则评测（评测器自带
  lease：计时窗口停持有器，测完回占）
- 违规解写 VIOLATION_SKIPPED 标记；全部落定后退出
"""
import glob, json, os, re, subprocess, sys, time

CARD, GROUP = sys.argv[1], sys.argv[2]
WS = "/home/ziming/ksearch_h100_portable"
ROOT = {"fi": WS + "/baseline/ksearch/experiments/formal_h100",
        "l1": WS + "/baseline/ksearch-sol-execbench/experiments/formal_sol_h100"}[GROUP]
ENV = dict(os.environ)
ENV["PATH"] = "/home/ziming/miniconda3/envs/ksearch/bin:" + ENV.get("PATH", "")
ENV["CUDA_VISIBLE_DEVICES"] = CARD
ENV["GPU_OCCUPANCY_ENABLED"] = "1"
BAN = re.compile(r"(torch\.matmul|torch\.mm\b|torch\.bmm|torch\.addmm|torch\.einsum|F\.linear|F\.conv\w*|torch\.fft\.\w+|torch\.cumsum|torch\.sort\b|torch\.topk|torch\.unique|@torch\.compile)")

def log(m): print(f"[lane-{GROUP}-gpu{CARD} {time.strftime('%m%d %H:%M:%S')}] {m}", flush=True)

while True:
    pending = []
    for d in sorted(glob.glob(ROOT + "/*/run_seed0")):
        if not os.path.exists(d + "/DONE"): continue
        if os.path.exists(d + "/unified/evaluation.json"): continue
        if os.path.exists(d + "/unified/performance.json"): continue
        if os.path.exists(d + "/unified/VIOLATION_SKIPPED"): continue
        pending.append((os.path.basename(os.path.dirname(d)), d))
    for t, d in pending:
        sols = sorted(glob.glob(f"{d}/ksearch-artifacts/{t}/solutions/{t}/*.json"), key=os.path.getmtime)
        if not sols:
            os.makedirs(d + "/unified", exist_ok=True)
            open(d + "/unified/VIOLATION_SKIPPED", "w").write("no-solution\n")
            log(f"SKIP no-solution: {t}"); continue
        src = "".join(s.get("content", "") for s in json.load(open(sols[-1])).get("sources", []))
        if BAN.search(src) or "@triton.jit" not in src:
            os.makedirs(d + "/unified", exist_ok=True)
            open(d + "/unified/VIOLATION_SKIPPED", "w").write(os.path.basename(sols[-1]) + "\n")
            log(f"SKIP violation: {t}"); continue
        task = f"L1/{t}" if GROUP == "l1" else t
        cmd = ["python3", WS + "/scripts/ksearch_final_eval.py", "--task", task, "--run-dir", d, "--iterations", "100"]
        cmd += ["--timeout", "7200"] if GROUP == "l1" else ["--trials", "1"]
        log(f"EVAL {t}")
        try:
            rc = subprocess.run(cmd, env=ENV, capture_output=True, text=True, timeout=7200).returncode
            log(f"EVAL {t} rc={rc}")
        except Exception as e:
            log(f"EVAL {t} error: {e}")
    ds = list(glob.glob(ROOT + "/*/run_seed0"))
    if all(os.path.exists(x+"/DONE") for x in ds) and all(
        os.path.exists(x+"/unified/evaluation.json") or os.path.exists(x+"/unified/performance.json")
        or os.path.exists(x+"/unified/VIOLATION_SKIPPED") for x in ds):
        log("ALL SETTLED — exiting"); break
    time.sleep(300)
