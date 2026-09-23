#!/usr/bin/env bash
# 每 10 分钟向 /tmp/ksearch_progress.log 追加进度快照（只读监控，不碰 GPU）
while true; do
  for BASE in \
    /home/ziming/ksearch_h100_portable/baseline/ksearch/experiments/formal_h100 \
    /home/ziming/ksearch_h100_portable/baseline/ksearch-sol-execbench/experiments/formal_sol_h100 \
    /home/ziming/ksearch_h100_portable/baseline/ksearch-sol-execbench/experiments/formal_solL2_h100; do
    BATCH=$(basename "$(dirname "$BASE")")_$(basename "$BASE")
    {
      echo "===== $(date '+%m-%d %H:%M') [$BATCH] ====="
      for d in "${BASE}"/*/run_seed0; do
        [ -d "$d" ] || continue
        n=$(basename "$(dirname "$d")")
        # FI 打印 "Round summary"，SOL 打印 "[sol-task] ... workloads passed"，两种都计
        done_evals=$(grep -cE "Round summary|feedback workloads passed" "$d/campaign_stdout.log" 2>/dev/null)
        best=$(grep -oE "mean_speedup=[0-9.]+x|speedup=[0-9.]+x" "$d/campaign_stdout.log" 2>/dev/null | sed 's/.*speedup=//;s/x$//' | sort -g | tail -1)
        tok=$(/home/ziming/miniconda3/envs/ksearch/bin/python3 -c "
import json,sys
t=o=0
for line in open('$d/usage.jsonl'):
    try:
        r=json.loads(line); t+=int(r.get('input_tokens',r.get('prompt_tokens')) or 0); o+=int(r.get('output_tokens',r.get('completion_tokens')) or 0)
    except Exception: pass
print(f'{t/1000:.0f}k_in/{o/1000:.0f}k_out')" 2>/dev/null)
        echo "$n: ${done_evals:-0}/100 evals | best_speedup=${best:--}x | tok ${tok:-0}"
      done
    } >> /tmp/ksearch_progress.log 2>&1
  done
  sleep 600
done
