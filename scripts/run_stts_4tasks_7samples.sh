#!/usr/bin/env bash
# 4 previously-failed tasks x 7 samples, eval cards 2+3 only.
# Separate RUN_ROOT so the completed 1-sample baseline stays untouched.
set -euo pipefail
cd "$(dirname "$0")"

RUN_PY=/home/ziming/miniconda3/envs/fib/bin/python
export STTS_RUN_ROOT=/home/ziming/whr/run/baseline/drtriton/kbstyle_stts_4tasks_7samples_20260922
mkdir -p "$STTS_RUN_ROOT"

# vLLM must be up (persistent on GPU 0)
until $RUN_PY -c "from openai import OpenAI; OpenAI(api_key='EMPTY', base_url='http://127.0.0.1:8001/v1').models.list()" 2>/dev/null; do
  echo "waiting for vLLM..."; sleep 30
done
echo "vLLM is up"

exec $RUN_PY kbstyle_stts.py \
  --samples 7 --iterations 10 --patience 0 \
  --task flashinfer/mla_paged_decode_h16_ckv512_kpe64_ps1 \
  --task SOL/L1/053_gaussian_topk_sparse_activation \
  --task SOL/L1/058_moe_expert_token_radix_sort_with_prefix_sum \
  --task SOL/L2/030_flux_concatenated_sequence_processing_with_split \
  --servers http://127.0.0.1:8001/v1 \
  --gpus 2,3,4 --max-parallel 3 --concurrent-tasks 4 \
  2>&1 | tee -a "$STTS_RUN_ROOT/campaign.log"
