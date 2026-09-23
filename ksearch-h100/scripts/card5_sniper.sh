#!/usr/bin/env bash
# 卡5 狙击手（2026-09-21，用户指令）：卡5 一空（同事 72.9GB 清退），立即把
# FI 四个重负载题挪上去（gqa_paged_decode/mla_paged_decode/gqa_paged_prefill/gemm），
# 重启 campaign 池扩为 1,2,5。流程=标准维护：停 supervisor → 改映射 → 停 campaign
# → 重启 → 重挂 supervisor。
WS=/home/ziming/ksearch_h100_portable
log(){ echo "[sniper $(date '+%m%d %H:%M:%S')] $*" >> /tmp/card5_sniper.log; }
while true; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 5 | tr -d ' ')
  [ -z "$used" ] && { sleep 60; continue; }
  if [ "$used" -lt 2000 ]; then
    log "card5 FREE (${used}MiB) — executing takeover"
    SP=$(pgrep -f "campaign[_]supervisor[.]sh" | head -1); [ -n "$SP" ] && kill "$SP" && log "supervisor stopped"
    python3 - <<'PYEOF'
import re
p='/home/ziming/ksearch_h100_portable/scripts/ksearch_pool_campaign.sh'
s=open(p).read()
for t in ['gqa_paged_decode_h32_kv8_d128_ps1','mla_paged_decode_h16_ckv512_kpe64_ps1','gqa_paged_prefill_causal_h32_kv8_d128_ps1','gemm_n4096_k4096']:
    s=re.sub(r'\["'+t+r'"\]="[0-9]"', f'["{t}"]="5"', s)
open(p,'w').write(s)
PYEOF
    log "map updated: 4 tasks -> gpu5"
    pkill -f "ksearch_pool_campaign[.]sh --pool 1,2"; sleep 2
    for t in dsa_ gdn_ gemm gqa_ mla_ rmsnorm; do pkill -f "ksearch[-]run[.]sh ${t}"; done; sleep 1
    pkill -f "task-source[ ]flashinfer"; sleep 2; pkill -9 -f "task-source[ ]flashinfer" 2>/dev/null
    setsid nohup env KSEARCH_STRICT_NO_LIB=1 bash $WS/scripts/ksearch_pool_campaign.sh \
      --pool 1,2,5 --concurrency 10 --tag formal_h100 --tasks \
      dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64 gdn_decode_qk4_v8_d128_k_last \
      gdn_prefill_qk4_v8_d128_k_last gemm_n4096_k4096 gqa_paged_decode_h32_kv8_d128_ps1 \
      gqa_paged_prefill_causal_h32_kv8_d128_ps1 gqa_ragged_prefill_causal_h32_kv8_d128 \
      mla_paged_decode_h16_ckv512_kpe64_ps1 mla_paged_prefill_causal_h16_ckv512_kpe64_ps1 \
      rmsnorm_h4096 >> /tmp/par6_campaign.log 2>&1 < /dev/null &
    sleep 5; log "campaign relaunched pool=1,2,5"
    setsid nohup bash $WS/scripts/campaign_supervisor.sh >> /tmp/ksearch_supervisor_stdout.log 2>&1 < /dev/null &
    sleep 3; log "supervisor re-armed; sniper exiting"
    exit 0
  fi
  log "card5 busy (${used}MiB), keep watching"
  sleep 60
done
