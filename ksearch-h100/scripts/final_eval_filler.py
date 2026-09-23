#!/usr/bin/env python3
"""空隙填充器（2026-09-21）：用卡 3 的空闲片段跑新完赛题的最终评测。
- 每 5 分钟扫描 FI+L1 任务：DONE 且无终评产物 → 守卫审计 → 干净则评测
- 评测期间持卡 3 的池锁（与 L1 窗口互斥，L1 会排队让位）
- 违规解写 VIOLATION_SKIPPED 标记，不评测（与 L2 口径一致）
- 全部 30 题有产物或标记后退出
"""
import fcntl, glob, json, os, re, subprocess, sys, time

WS = "/home/ziming/ksearch_h100_portable"
ENV = dict(os.environ)
ENV["PATH"] = "/home/ziming/miniconda3/envs/ksearch/bin:" + ENV.get("PATH", "")
ENV["CUDA_VISIBLE_DEVICES"] = "3"
ENV["GPU_OCCUPANCY_ENABLED"] = "1"

FI = WS + "/baseline/ksearch/experiments/formal_h100"
L1 = WS + "/baseline/ksearch-sol-execbench/experiments/formal_sol_h100"
BAN = re.compile(r"(torch\.matmul|torch\.mm\b|torch\.bmm|torch\.addmm|torch\.einsum|F\.linear|F\.conv\w*|torch\.fft\.\w+|torch\.cumsum|torch\.sort\b|torch\.topk|torch\.unique|@torch\.compile)")

def log(m): print(f"[filler {time.strftime('%m%d %H:%M:%S')}] {m}", flush=True)

def audit_exit_save(run_dir, tname):
    sols = sorted(glob.glob(f"{run_dir}/ksearch-artifacts/{tname}/solutions/{tname}/*.json"), key=os.path.getmtime)
    if not sols: return None, "no-solution"
    src = "".join(s.get("content", "") for s in json.load(open(sols[-1])).get("sources", []))
    return (not BAN.search(src) and "@triton.jit" in src), sols[-1]

def done_and_pending(root, sol_prefix):
    out = []
    for d in sorted(glob.glob(root + "/*/run_seed0")):
        t = os.path.basename(os.path.dirname(d))
        done = os.path.exists(d + "/DONE")
        res = os.path.exists(d + "/unified/evaluation.json") or os.path.exists(d + "/unified/performance.json")
        skip = os.path.exists(d + "/unified/VIOLATION_SKIPPED")
        if done and not res and not skip:
            out.append((t, d))
    return out

def run_eval(t, run_dir, is_sol):
    task = f"L1/{t}" if is_sol else t
    cmd = [sys.executable.replace("/bin/python3", "/bin/python3"), WS + "/scripts/ksearch_final_eval.py",
           "--task", task, "--run-dir", run_dir, "--iterations", "100"]
    if is_sol: cmd += ["--timeout", "7200"]
    else: cmd += ["--trials", "1"]
    return subprocess.run(cmd, env=ENV, capture_output=True, text=True, timeout=7200).returncode

lock_fh = open("/tmp/ksearch_gpu_pool/gpu3.lock", "a+")
while True:
    pending = done_and_pending(FI, "") + done_and_pending(L1, "")
    if not pending:
        fi_all = all(os.path.exists(d+"/unified/evaluation.json") or os.path.exists(d+"/unified/VIOLATION_SKIPPED") or not os.path.exists(d+"/DONE")
                     for d in glob.glob(FI + "/*/run_seed0"))
        log(f"idle: no pending DONE-without-result tasks")
    for t, d in pending:
        is_sol = t[0:3] in ("002","005","007","008","018","020","053","058","070","092")
        clean, info = audit_exit_save(d, t)
        if not clean:
            os.makedirs(d + "/unified", exist_ok=True)
            open(d + "/unified/VIOLATION_SKIPPED", "w").write(f"{info}\n")
            log(f"SKIP violation: {t} ({info})")
            continue
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
        lock_fh.seek(0); lock_fh.truncate(); lock_fh.write(f"pid={os.getpid()} gpu=3 task=FINAL_EVAL {t}\n"); lock_fh.flush()
        try:
            log(f"EVAL {t} on gpu3 (solution: {os.path.basename(info or '')[:50]})")
            rc = run_eval(t, d, is_sol)
            log(f"EVAL {t} rc={rc}")
        except Exception as e:
            log(f"EVAL {t} error: {e}")
        finally:
            lock_fh.seek(0); lock_fh.truncate(); lock_fh.write("free\n"); lock_fh.flush()
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
    fi = [d for d in glob.glob(FI+"/*/run_seed0")]
    l1 = [d for d in glob.glob(L1+"/*/run_seed0")]
    def settled(ds, name):
        return all(os.path.exists(d+"/DONE") for d in ds) and all(
            os.path.exists(d+"/unified/evaluation.json") or os.path.exists(d+"/unified/performance.json") or os.path.exists(d+"/unified/VIOLATION_SKIPPED") for d in ds)
    if settled(fi, "FI") and settled(l1, "L1"):
        log("ALL SETTLED — exiting")
        break
    time.sleep(300)
