#!/usr/bin/env python3
"""按 whr/实验产物整理与上传规范.md 构建 kda-h100/ 归档目录。

映射（KDA 结构 → 规范结构）：
  summary/results_table.{md,csv}   ← 从原始 final.json/feedback.json/state.json 生成（不手抄）
  summary/best_solutions/          ← 控制侧候选归档 solution.py（含来源头注释）
  summary/best_timing_detail.csv   ← 最优候选 final.json 的 per_workload 明细
  tasks/<题目>/formal-h100/        ← workspace(合同/产物/账本/评测明细) + 控制侧(状态/快照/transcript)
  batch_logs/                      ← campaign/timer/keeper/watchdog 日志与记账
  scripts/ evaluators/             ← 流水线脚本与评测器
  task_plan 等价物                 ← experiment_manifest.json + campaign json
不进归档：reference-cache（规范明令排除）、__pycache__、代理日志、GPU 锁。
"""
import json, glob, shutil, csv, io
from pathlib import Path

ROOT = Path("/home/ziming/kda-ops")
OUT = Path("/home/ziming/kda-h100")
RUN_PREFIX = "formal-kda-h100-20260920--"
TAG = "formal-kda-h100-20260920"

# ---------- 收集每题数据 ----------
tasks = []
for st in sorted(glob.glob(str(ROOT / "kda-control" / f"{RUN_PREFIX}*/state.json"))):
    run = Path(st).parent.name
    short = run.replace(RUN_PREFIX, "")
    bench, name = short.split("--", 1)
    ws = ROOT / "kda-runs" / run
    ctl = ROOT / "kda-control" / run
    s = json.load(open(st))
    best = None
    for cid, r in s.get("candidates", {}).items():
        try:
            d = json.load(open(r["feedback_result"]))
            g = d.get("geomean_speedup")
            if d.get("valid") and g and (best is None or g > best[1]):
                best = (cid, g)
        except Exception:
            pass
    final = None
    if best:
        fj = ws / "runs" / "candidates" / best[0] / "final.json"
        if fj.is_file():
            d = json.loads(fj.read_text())
            if d.get("valid") and d.get("geomean_speedup"):
                final = (best[0], d["geomean_speedup"])
    # token（observability 优先，缺失用 transcript 累计）
    obs = ctl / "observability.json"
    tk = 0
    if obs.is_file():
        u = json.load(open(obs)).get("usage") or {}
        tk = sum(u.get(k) or 0 for k in ("input_tokens", "cache_read_input_tokens",
                                          "cache_creation_input_tokens", "output_tokens"))
    else:
        for f in (ctl / "claude").glob("*.jsonl"):
            for line in open(f):
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                m = rec.get("message")
                if isinstance(m, dict) and m.get("role") == "assistant":
                    u = m.get("usage") or {}
                    tk += sum(u.get(k) or 0 for k in ("input_tokens", "cache_read_input_tokens",
                                                       "cache_creation_input_tokens", "output_tokens"))
    terminal = ""
    for mk, label in (("SEARCH_COMPLETE", "search_complete"), ("TOKEN_LIMIT_REACHED", "token_limit"),
                      ("TIME_BUDGET_REACHED", "time_budget")):
        if (ws / mk).is_file():
            terminal = label
            break
    tasks.append({
        "benchmark": bench, "task": name, "run": run, "ws": ws, "ctl": ctl,
        "evals": s.get("candidate_evaluations", 0),
        "best_fb": best, "final": final, "token": tk, "terminal": terminal,
    })

# ---------- summary ----------
sumdir = OUT / "summary"
bsdir = sumdir / "best_solutions"
bsdir.mkdir(parents=True, exist_ok=True)

def speedup_str(x):
    return f"{x:.2f}" if x else ""

rows = []
for t in tasks:
    fb = t["best_fb"]
    fin = t["final"]
    cid = fin[0] if fin else (fb[0] if fb else "")
    rows.append({
        "benchmark": t["benchmark"], "task": t["task"], "evaluations": t["evals"],
        "best_candidate": cid,
        "feedback_geomean": speedup_str(fb[1] if fb else None),
        "final_geomean": speedup_str(fin[1] if fin else None),
        "final_valid": bool(fin),
        "terminal_state": t["terminal"] or ("exhausted_windows" if t["evals"] >= 0 else ""),
        "token_budget_total": t["token"],
    })

# CSV
csv_path = sumdir / "results_table.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

# MD 总表
md = io.StringIO()
md.write("# KDA H100 首轮实验结果总表\n\n")
md.write("- 批次：`formal-kda-h100-20260920`（30 题，每题 1 小时含 draft/plan，反馈全量粗测 w2/i10）\n")
md.write("- **feedback_geomean**：搜索内粗测（warmup 2 / 10 iterations，排序信号）\n")
md.write("- **final_geomean**：终局精测（全量 workload / warmup 3–10 / **100 iterations** / 参考实现同进程成对实时计时，无缓存）——**权威口径**\n")
md.write("- 硬件：H100 80GB (sm_90)；token 为 budget 口径（未缓存输入+缓存写+缓存读+输出）累计\n")
md.write("- 17/30 精测有效；最优候选回退审计 17/17 干净（零 torch 回退）\n\n")
md.write("| 组 | 题 | 评次 | 最优候选 | 反馈粗测 | **终局精测** | 终态 | token(万) |\n")
md.write("|---|---|---:|---|---:|---:|---|---:|\n")
for r, t in zip(rows, tasks):
    md.write(f"| {r['benchmark']} | {r['task']} | {r['evaluations']} | {r['best_candidate'] or '—'} | "
             f"{r['feedback_geomean'] or '—'}× | {('**' + r['final_geomean'] + '×**') if r['final_valid'] else '—'} | "
             f"{r['terminal_state']} | {r['token_budget_total']/10000:.0f} |\n")
(sumdir / "results_table.md").write_text(md.getvalue())

# best_solutions + timing detail
detail = io.StringIO()
detail.write("uuid,status,ref_ms,sol_ms,speedup,axes\n")
for r, t in zip(rows, tasks):
    if not r["final_valid"]:
        continue
    cid = r["best_candidate"]
    src = t["ctl"] / "candidates" / cid / "solution.py"
    dst = bsdir / f"{t['task']}.py"
    header = (f"# {t['task']} — best candidate {cid} (batch {TAG})\n"
              f"# feedback (w2/i10 coarse): {r['feedback_geomean']}x | final (100 iters, paired timing): "
              f"{r['final_geomean']}x, valid\n"
              f"# source: kda-control/{t['run']}/candidates/{cid}/solution.py (sha256-locked snapshot)\n")
    dst.write_text(header + src.read_text())
    fj = t["ws"] / "runs" / "candidates" / cid / "final.json"
    d = json.loads(fj.read_text())
    for pw in d.get("per_workload", []):
        axes = json.dumps(pw.get("axes") or {}, ensure_ascii=False).replace(",", ";")
        detail.write(f"{t['task']},{pw['uuid'][:8]},{pw.get('status')},{pw.get('ref_ms')},"
                     f"{pw.get('sol_ms')},{pw.get('speedup')},\"{axes}\"\n")
(sumdir / "best_timing_detail.csv").write_text(detail.getvalue())

# ---------- tasks/<题>/<批次>/ ----------
for t in tasks:
    d = OUT / "tasks" / t["task"] / TAG
    d.mkdir(parents=True, exist_ok=True)
    # workspace 侧：合同与产物
    for item in ("CLAUDE.md", "TASK.md", "README.md", "candidates.jsonl"):
        src = t["ws"] / item
        if src.is_file():
            shutil.copy2(src, d / item)
    if (t["ws"] / "docs").is_dir():
        shutil.copytree(t["ws"] / "docs", d / "docs", dirs_exist_ok=True)
    if (t["ws"] / "task").is_dir():
        shutil.copytree(t["ws"] / "task", d / "task", dirs_exist_ok=True)
    for mk in ("SEARCH_COMPLETE", "TOKEN_LIMIT_REACHED", "TIME_BUDGET_REACHED"):
        src = t["ws"] / mk
        if src.is_file():
            shutil.copy2(src, d / mk)
    # 评测明细（feedback/final json+log）
    runs_dir = t["ws"] / "runs" / "candidates"
    if runs_dir.is_dir():
        shutil.copytree(runs_dir, d / "runs", dirs_exist_ok=True)
    # 控制侧：状态、配置、快照、transcript
    for item in ("state.json", "task.json", "last_claude_stage.json", "observability.json",
                 "pool-driver.log", "feedback_workloads.jsonl"):
        src = t["ctl"] / item
        if src.is_file():
            shutil.copy2(src, d / ("control_" + item))
    if (t["ctl"] / "candidates").is_dir():
        shutil.copytree(t["ctl"] / "candidates", d / "control_candidates", dirs_exist_ok=True)
    if (t["ctl"] / "claude").is_dir():
        shutil.copytree(t["ctl"] / "claude", d / "control_transcripts", dirs_exist_ok=True)
    for ar in t["ctl"].glob("claude-superseded-*"):
        shutil.copytree(ar, d / ("control_" + ar.name), dirs_exist_ok=True)

# ---------- batch_logs/ ----------
bl = OUT / "batch_logs"
bl.mkdir(exist_ok=True)
cdir = ROOT / "kda-control" / "campaigns"
for f in cdir.glob(f"{TAG}*"):
    if f.is_file():
        shutil.copy2(f, bl / f.name)

# ---------- scripts/ evaluators/ ----------
shutil.copytree(ROOT / "kda-controller", OUT / "scripts", dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("__pycache__"))
shutil.copytree(ROOT / "evaluators", OUT / "evaluators", dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("__pycache__"))

# ---------- 配置清单 ----------
shutil.copy2(ROOT / "experiment_manifest.json", OUT / "experiment_manifest.json")
shutil.copy2(cdir / f"{TAG}.json", OUT / "campaign.json")

# 清理 __pycache__ / 大日志
for p in OUT.rglob("__pycache__"):
    shutil.rmtree(p)

print("归档构建完成:", OUT)
n = len(list((OUT / "tasks").iterdir()))
print(f"tasks 题目数: {n}")
print(f"best_solutions: {len(list(bsdir.glob('*.py')))}")
