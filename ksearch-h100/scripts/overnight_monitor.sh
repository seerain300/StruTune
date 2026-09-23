#!/usr/bin/env bash
# 夜间值守 v2（2026-09-22）：40 分钟无调用完成即触发小+中型网关探针，
# 依探针结果决定 续跑/重启/暂停（用户裁定，替代 v1 的 90 分钟干等）。
WS=/home/ziming/ksearch_h100_portable
LOG=$WS/baseline/ksearch/experiments/formal_h100/gdn_prefill_qk4_v8_d128_k_last/run_seed0
USAGE=$LOG/usage.jsonl
MLOG=/tmp/overnight_monitor.log
KEY=$(grep -oP '(?<=K_SEARCH_KEY=).*' ~/.bashrc | head -1 | tr -d '"' | tr -d "'")
export KEY
log(){ echo "[mon $(date '+%m%d %H:%M:%S')] $*" >> $MLOG; }
evals(){ grep -c "Round summary" $LOG/campaign_stdout.log 2>/dev/null || echo 0; }
last_llm_age(){ python3 -c "
import json,datetime,time
j=None
for l in open('$USAGE'): j=json.loads(l)
ts=datetime.datetime.fromisoformat(j['ts_utc']).timestamp()
print(int(time.time()-ts))" 2>/dev/null || echo 999999; }
probe_small(){ timeout 60 curl -s -o /dev/null -w "%{http_code} %{time_total}" https://llmapi.isrc.ac.cn/v1/chat/completions -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" -d '{"model":"GPT-5.6-Sol","messages":[{"role":"user","content":"ok"}],"max_tokens":5}' 2>/dev/null || echo "FAIL -"; }
probe_medium(){ python3 - <<'PYEOF'
import json,subprocess,time,os
body=json.dumps({"model":"GPT-5.6-Sol","messages":[{"role":"user","content":"Count from 1 to 300 as a continuous list of words."}],"max_tokens":1000})
open('/tmp/_probe_med.json','w').write(body)
t0=time.time()
r=subprocess.run(["curl","-s","-m","290","-o","/tmp/_probe_med_out.json","-w","%{http_code} %{time_total}","https://llmapi.isrc.ac.cn/v1/chat/completions","-H",f"Authorization: Bearer {os.environ['KEY']}","-H","Content-Type: application/json","-d","@/tmp/_probe_med.json"],capture_output=True,text=True)
dur=time.time()-t0
try:
    c=json.load(open('/tmp/_probe_med_out.json')).get('usage',{}).get('completion_tokens',0) or 0
except Exception: c=0
code=r.stdout.split()[0] if r.stdout else "FAIL"
print(f"{code} {c/dur:.1f}")
PYEOF
}
gw_ok(){ M=$(probe_medium); C=$(echo $M | awk '{print $1}'); T=$(echo $M | awk '{print $2}'); echo "$M"; }
main_mb(){ nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -F', ' -v p=$(pgrep -f "definition gdn_prefill" | head -1) '$1==p {print $2}' | head -1; }
launch(){
  cd $WS && setsid nohup env KSEARCH_STRICT_NO_LIB=1 KSEARCH_OWNER_TAG=ksearch-campaign-formal_h100-overnight KSEARCH_GPU_POOL=5 KSEARCH_TASK_GPU=5 KSEARCH_RUN_TAG=formal_h100 KSEARCH_MAX_ROUNDS=$((100-$(evals))) KSEARCH_WM_MAX_ACTION_NODES=20 KSEARCH_WM_MAX_ATTEMPTS_PER_NODE=5 KSEARCH_POOL_TASK=gdn_prefill_qk4_v8_d128_k_last bash ./ksearch-run.sh gdn_prefill_qk4_v8_d128_k_last 0 --wm --continue-from-world-model auto >> $LOG/campaign_stdout.log 2>&1 < /dev/null &
  disown; log "relaunched (budget=$((100-$(evals))))"
}
killtask(){ for p in $(pgrep -f "definition gdn_prefill"); do kill -9 $p 2>/dev/null; done; }
decision(){ # 输入: 中型探针串; 输出 1=网关可用(≥20tok/s且200)
  python3 -c "
import sys
c,t=sys.argv[1].split()
print(1 if c=='200' and float(t)>=20 else 0)" "$1" 2>/dev/null || echo 0; }
paused=0; last_evals=$(evals); stalled=0
while true; do
  E=$(evals)
  if [ "$E" -ge 100 ]; then log "DONE (evals=100) — stop holder, release card5"; GPU_OCCUPANCY_ENABLED=1 python3 /home/ziming/MTMC-baseline/agent-generation/scripts/gpu_occupancy.py stop --gpu 5; exit 0; fi
  PID=$(pgrep -f "definition gdn_prefill" | head -1)
  AGE=$(last_llm_age); MB=$(main_mb)
  log "evals=$E pid=${PID:-dead} llm_age=${AGE}s main=${MB:-0}MB paused=$paused"
  if [ -n "$MB" ] && [ "$MB" -gt 20000 ]; then log "memory ratchet ${MB}MB -> restart task"; killtask; sleep 5; launch; sleep 1200; continue; fi
  if [ "$AGE" -gt 2400 ]; then
    S=$(probe_small); M=$(probe_medium); D=$(decision "$M")
    log "GATEWAY-TRIAGE llm_age=${AGE}s small=[$S] medium=[$M tok/s] decision_ok=$D"
    if [ "$D" = "1" ]; then
      if [ -n "$PID" ]; then log "gateway OK but task stalled -> restart task (fresh calls)"; killtask; sleep 5; else log "gateway OK, task dead -> launch"; fi
      launch; paused=0
    else
      if [ -n "$PID" ]; then log "gateway degraded -> PAUSE task (stop token burn)"; killtask; fi
      paused=1
    fi
    sleep 1200; continue
  fi
  if [ -z "$PID" ] && [ "$paused" = "0" ]; then log "task dead (llm_age=${AGE}s) -> launch"; launch; sleep 1200; continue; fi
  if [ "$paused" = "1" ]; then
    M=$(probe_medium); D=$(decision "$M")
    log "paused: medium=[$M] ok=$D"
    [ "$D" = "1" ] && { log "gateway recovered -> relaunch"; launch; paused=0; }
    sleep 1200; continue
  fi
  if [ "$E" -eq "$last_evals" ]; then stalled=$((stalled+1)); else stalled=0; fi
  [ "$stalled" -ge 3 ] && log "note: no new evals for 3 cycles (evals=$E llm_age=${AGE}s)"
  last_evals=$E
  sleep 1200
done
