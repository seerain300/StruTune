#!/usr/bin/env bash
# L2 十题最终评测链（卡5，串行，先快后慢）：全量 wl、100 iter、trials 1、无缓存
WS=/home/ziming/ksearch_h100_portable
RD=$WS/baseline/ksearch-sol-execbench/experiments/formal_solL2_h100
export CUDA_VISIBLE_DEVICES=5
export GPU_OCCUPANCY_ENABLED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for t in 040_altup_predict_correction_cycle_backward \
         049_group_limited_topk_routing \
         030_flux_concatenated_sequence_processing_with_split; do
  echo "===== FINAL_EVAL L2/$t start $(date '+%m%d %H:%M:%S') ====="
  python3 $WS/scripts/ksearch_final_eval.py --task "L2/$t" \
    --run-dir "$RD/$t/run_seed0" --iterations 100 --timeout 7200
  echo "===== FINAL_EVAL L2/$t done rc=$? $(date '+%m%d %H:%M:%S') ====="
done
echo "ALL L2 FINAL EVALS COMPLETE $(date '+%m%d %H:%M:%S')"
