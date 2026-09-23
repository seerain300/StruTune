#!/usr/bin/env python3
"""批量 final 精测：对全部有有效结果的最优候选执行权威评测（100 iters 成对计时）。

流程（手册 §4.5）：恢复最优候选源码 → 卡上独占执行 evaluate_candidate.py final
→ 结果落 runs/candidates/<cXXX>/final.json。rc=3（外部污染判废）自动重试一次
（必要时版本化上调 final 名额）。
"""
import json, glob, shutil, subprocess, sys, time
from pathlib import Path

ROOT = Path("/home/ziming/kda-ops")
sys.path.insert(0, str(ROOT / "kda-controller"))

TASKS = []
for st in sorted(glob.glob(str(ROOT / "kda-control/formal-kda-h100-20260920*/state.json"))):
    run = Path(st).parent.name
    s = json.load(open(st))
    best = None
    for cid, r in s.get("candidates", {}).items():
        try:
            d = json.load(open(r["feedback_result"])); g = d.get("geomean_speedup")
            if d.get("valid") and g and (best is None or g > best[1]): best = (cid, g)
        except Exception: pass
    if best:
        TASKS.append((run, best[0], best[1]))

print(f"待精测 {len(TASKS)} 题", flush=True)
results = []
for i, (run, cid, coarse) in enumerate(TASKS, 1):
    ws = ROOT / "kda-runs" / run
    ctl = ROOT / "kda-control" / run
    src = ctl / "candidates" / cid / "solution.py"
    shutil.copy2(src, ws / "solution" / "solution.py")
    out = ws / "runs" / "candidates" / cid / "final.json"
    print(f"[{i}/{len(TASKS)}] {run.split('--')[-1][:40]} {cid} (粗测 {coarse:.1f}x) ...", flush=True)
    for attempt in (1, 2):
        t0 = time.time()
        rc = subprocess.run(
            [sys.executable, str(ROOT / "kda-controller/evaluate_candidate.py"), "final",
             "--candidate", cid, "--workspace", str(ws), "--gpu-wait-timeout", "600"],
            capture_output=True, text=True)
        dt = time.time() - t0
        if rc.returncode == 0 and out.is_file():
            d = json.loads(out.read_text())
            g = d.get("geomean_speedup")
            print(f"    final={g:.2f}x valid={d.get('valid')} passed={d.get('passed')}/{d.get('total')} ({dt/60:.1f}m)", flush=True)
            results.append((run, cid, coarse, g, d.get("valid")))
            break
        print(f"    rc={rc.returncode} 尝试{attempt} ({dt/60:.1f}m): {rc.stdout[-200:] if rc.stdout else rc.stderr[-200:]}", flush=True)
        if attempt == 1:
            # 判废消耗 final 名额：版本化 +1 后重试
            tp = ctl / "task.json"
            t = json.loads(tp.read_text())
            t["final_evaluation_budget"] = int(t.get("final_evaluation_budget", 1)) + 1
            t["final_budget_note"] = f"rc={rc.returncode} invalidation retry at {time.strftime('%H:%M')}"
            tp.write_text(json.dumps(t, ensure_ascii=False, indent=2) + "\n")

print("\n=== 精测汇总 ===", flush=True)
ok = 0
for run, cid, coarse, g, valid in results:
    if valid and g: ok += 1
    print(f"{run.split('--')[-1][:42]:<44} {cid} 粗测{coarse:7.1f}x → 精测{g or float('nan'):7.1f}x {'✅' if valid else '❌'}", flush=True)
print(f"\n{ok}/{len(TASKS)} 精测有效", flush=True)
