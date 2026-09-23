#!/usr/bin/env python3
"""生成瓶颈诊断：区分「模型在真实生成」vs「在等 API」vs「会话闲置」。

对每道在跑的题，解析其最新 transcript：
  - 输出 token 速率（按 assistant 消息的 output_tokens / 消息跨度时间）
  - 相邻 assistant 消息的间隔分布（>60s 的间隔 = 疑似 API 等待/重试）
  - 最近一条消息距今（会话是否还活着）
判定：
  healthy     — 速率正常（>100 tok/min）且间隔正常
  api-wait    — 有输出但长间隔占比高（等 API/重试）
  idle/stuck  — 长时间无新消息
"""
import glob, json, subprocess, sys, time
from datetime import datetime
from pathlib import Path

ROOT = Path("/home/ziming/kda-ops")

def analyze(run_glob: str) -> None:
    for d in sorted(glob.glob(f"{ROOT}/kda-control/formal-kda-h100-20260920*{run_glob}*/claude")):
        run = Path(d).parent.name
        ws = ROOT / "kda-runs" / run
        if any((ws / m).is_file() for m in ("SEARCH_COMPLETE", "TOKEN_LIMIT_REACHED", "TIME_BUDGET_REACHED")):
            print(f"{run.split('--')[-1][:40]:<42} [终态，跳过]")
            continue
        transcripts = sorted(Path(d).glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
        if not transcripts:
            continue
        latest = transcripts[-1]
        msgs = []  # (timestamp, output_tokens)
        for line in open(latest):
            try:
                rec = json.loads(line)
            except Exception:
                continue
            m = rec.get("message")
            if isinstance(m, dict) and m.get("role") == "assistant":
                u = m.get("usage") or {}
                ts = rec.get("timestamp")
                out = (u.get("output_tokens") or 0) + (u.get("input_tokens") or 0) * 0  # 只算输出
                if ts:
                    msgs.append((datetime.fromisoformat(ts.replace("Z", "+00:00")), out))
        if len(msgs) < 2:
            print(f"{run.split('--')[-1][:40]:<42} 最新会话消息太少（{len(msgs)}），刚起步")
            continue
        span = (msgs[-1][0] - msgs[0][0]).total_seconds()
        total_out = sum(o for _, o in msgs)
        rate = total_out / (span / 60) if span > 0 else 0
        gaps = [(msgs[i+1][0] - msgs[i][0]).total_seconds() for i in range(len(msgs)-1)]
        long_gaps = [g for g in gaps if g > 60]
        wait_ratio = sum(long_gaps) / span if span > 0 else 0
        idle = (datetime.now().astimezone() - msgs[-1][0].astimezone()).total_seconds()
        verdict = "idle/stuck" if idle > 300 else ("api-wait" if wait_ratio > 0.4 else "healthy")
        print(f"{run.split('--')[-1][:40]:<42} {verdict:<10} 输出{total_out}tok/{span/60:.0f}m = {rate:.0f} tok/min | "
              f"长间隔(>60s) {len(long_gaps)} 个占 {wait_ratio*100:.0f}% | 最后消息 {idle/60:.1f}m 前")

if __name__ == "__main__":
    pattern = sys.argv[1] if len(sys.argv) > 1 else ""
    analyze(pattern)
