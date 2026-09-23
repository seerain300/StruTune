#!/usr/bin/env bash
# Independent 20-minute monitor for the full8 STTS campaign.
# Writes timestamped snapshots to monitor.log regardless of any session.
ROOT=/data1/workspace/weihongren/baseline/drtriton/kbstyle_stts_full8_20260919
LOG="$ROOT/monitor.log"
INTERVAL=1200
while true; do
  {
    echo "===== $(date '+%F %H:%M:%S') ====="
    done_n=$(cat "$ROOT/results.jsonl" 2>/dev/null | wc -l)
    echo "completed_tasks: ${done_n}/30"
    tok=$(python3 - <<'EOF' 2>/dev/null
import json
p=c=n=0
try:
    for line in open('/data1/workspace/weihongren/baseline/drtriton/kbstyle_stts_full8_20260919/tokens.jsonl'):
        r=json.loads(line); p+=r.get('prompt_tokens',0); c+=r.get('completion_tokens',0); n+=1
    print(f'{n} requests, prompt={p/1e6:.2f}M, completion={c/1e6:.2f}M')
except FileNotFoundError:
    print('no tokens yet')
EOF
)
    echo "tokens: $tok"
    echo "-- campaign tail:"
    tail -4 "$ROOT/campaign.log" 2>/dev/null | sed 's/^/   /'
    alive=$(pgrep -fc "kbstyle_stts.py" || true)
    echo "runner_alive: ${alive}"
  } >> "$LOG" 2>&1
  sleep "$INTERVAL"
done
