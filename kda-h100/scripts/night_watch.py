#!/usr/bin/env python3
"""值夜巡检仪表盘：一轮输出全部关键信号，供 15 分钟周期的监控使用。"""
import json
import glob
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path("/home/ziming/kda-ops")
now = datetime.now().astimezone()
print(f"=== 巡检 {now.strftime('%H:%M:%S')} ===")

# 1) GPU 状态
out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                      "--format=csv,noheader,nounits", "-i", "0,1,2"], capture_output=True, text=True).stdout
gpu = {}
for line in out.strip().splitlines():
    idx, mem, util = [x.strip() for x in line.split(",")]
    gpu[int(idx)] = (int(mem), int(util))
for i, (mem, util) in sorted(gpu.items()):
    flag = " <-- 被外部占用?" if mem > 1000 and util > 50 else ""
    print(f"  卡{i}: {mem}MiB util={util}%{flag}")

# 2) 各题状态与最优成绩
sys.path.insert(0, str(ROOT / "kda-controller"))
print("\n--- 题目进度（有评测或终态的） ---")
for st in sorted(glob.glob(str(ROOT / "kda-control/formal-kda-h100-20260920*/state.json"))):
    run = Path(st).parent.name
    short = run.replace("formal-kda-h100-20260920--", "").replace("sol_execbench--", "").replace("flashinfer--", "")[:30]
    try:
        s = json.load(open(st))
    except Exception:
        continue
    n = s.get("candidate_evaluations", 0)
    if n == 0 and not any((ROOT / "kda-runs" / run / m).is_file()
                          for m in ("SEARCH_COMPLETE", "TOKEN_LIMIT_REACHED", "TIME_BUDGET_REACHED")):
        continue
    best = None
    for cid, r in s.get("candidates", {}).items():
        try:
            d = json.load(open(r["feedback_result"]))
            g = d.get("geomean_speedup")
            if d.get("valid") and g and (best is None or g > best[1]):
                best = (cid, g)
        except Exception:
            pass
    marks = [m for m in ("SEARCH_COMPLETE", "TOKEN_LIMIT_REACHED", "TIME_BUDGET_REACHED")
             if (ROOT / "kda-runs" / run / m).is_file()]
    mark = {"SEARCH_COMPLETE": "完成", "TOKEN_LIMIT_REACHED": "token尽", "TIME_BUDGET_REACHED": "时尽"}.get(marks[0], "") if marks else ""
    print(f"  {short:<32} 评{n:>2}次  best={best[1]:.1f}x({best[0]}) {mark}" if best else
          f"  {short:<32} 评{n:>2}次  best=无 {mark}")

# 3) 会话活性：只对"有 claude 进程驻留但最新 transcript >15 分钟无产出"的题报警
#    （无进程 = 窗口间/排队中；有进程且停滞 = 长回合生成或重试死循环，需关注）
print("\n--- 会话活性（有进程驻留的题） ---")
proc_cwds = set()
for pid_line in subprocess.run(["ps", "-eo", "pid"], capture_output=True, text=True).stdout.split():
    try:
        proc_cwds.add(subprocess.run(["readlink", f"/proc/{pid_line}/cwd"],
                                     capture_output=True, text=True).stdout.strip())
    except Exception:
        pass
stuck = 0
for d in sorted(glob.glob(str(ROOT / "kda-control/formal-kda-h100-20260920*/claude"))):
    run = Path(d).parent.name
    ws = str(ROOT / "kda-runs" / run)
    if any((Path(ws) / m).is_file() for m in ("SEARCH_COMPLETE", "TOKEN_LIMIT_REACHED", "TIME_BUDGET_REACHED")):
        continue
    if ws not in proc_cwds:
        continue  # 当前没有进程在跑这题（窗口间隙或排队）
    transcripts = sorted(Path(d).glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    if not transcripts:
        continue
    age = time.time() - transcripts[-1].stat().st_mtime
    tag = "运行中" if age <= 360 else f"停滞{age/60:.0f}m⚠"
    print(f"  {tag:<10} {run[-40:]}")
    if age > 900:
        stuck += 1
if stuck == 0:
    print("  （无 >15 分钟停滞）")

# 4) 中转站健康（最近 100 条状态码）
log = ROOT / "claude-opus-proxy/proxy.ruizhi-ubuntu-h100-01.log"
if log.is_file():
    lines = subprocess.run(["tail", "-100", str(log)], capture_output=True, text=True).stdout.splitlines()
    from collections import Counter
    c = Counter()
    for line in lines:
        try:
            c[json.loads(line).get("status")] += 1
        except Exception:
            pass
    total = sum(c.values()) or 1
    bad = c.get(504, 0) + c.get(502, 0) + c.get(503, 0)
    print(f"\n--- API 最近{total}条: {dict(c)} | 5xx={bad} ({100*bad/total:.0f}%) ---")

# 5) 全局进程 + 重复 driver 检测（同一任务被多个 driver 驱动 = 事故）
nclaude = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout.count("no-chrome")
drivers = []
for line in subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout.splitlines():
    if "timer_driver.py" in line and "bash -ic" not in line and "grep" not in line:
        for tok in line.split("--tasks ")[1].split(" ")[0].strip("'").split(",") if "--tasks " in line else []:
            drivers.append(tok.strip())
from collections import Counter as _C
task_pat_counts = _C(drivers)
dupes = {k: v for k, v in task_pat_counts.items() if v > 1 and not k.endswith("*") or v > 2}
print(f"claude 会话: {nclaude} | driver 任务模式数: {len(drivers)} | 重复嫌疑: {dupes or '无'}")
