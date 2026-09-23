#!/usr/bin/env bash
# 卡0/1 占位看护（2026-09-22，用户指令）：一旦某卡显存 <2GB（持有器消失/被清），
# 用管理器持有器补位（管理器登记版，评测 lease 可感知）。
GPU_OCCUPANCY_PYTHON=/home/ziming/miniconda3/envs/ksearch/bin/python
export GPU_OCCUPANCY_PYTHON
while true; do
  for g in 0 1; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g | tr -d ' ')
    if [ -n "$used" ] && [ "$used" -lt 2000 ]; then
      echo "[hold01 $(date '+%m%d %H:%M:%S')] gpu$g free (${used}MiB) -> starting holder" >> /tmp/hold01_watcher.log
      GPU_OCCUPANCY_ENABLED=1 python3 /home/ziming/MTMC-baseline/agent-generation/scripts/gpu_occupancy.py start --gpu $g >> /tmp/hold01_watcher.log 2>&1
    fi
  done
  sleep 60
done
