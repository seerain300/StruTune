#!/usr/bin/env bash
# Probe run: 7 selected tasks x 1 sample, single vLLM server on GPU 0 (persistent).
# Launch only when GPUs are available. Everything is resumable if interrupted.
set -euo pipefail
cd "$(dirname "$0")"

RUN_PY=/home/ziming/miniconda3/envs/fib/bin/python   # runner env (openai client)

# 1) single vLLM server on GPU 0 (persistent, port 8001); start only if not already up
if $RUN_PY -c "from openai import OpenAI; OpenAI(api_key='EMPTY', base_url='http://127.0.0.1:8001/v1').models.list()" 2>/dev/null; then
  echo "vLLM already up, skipping server start"
else
  bash start_drkernel_gpu0_kbstyle.sh
fi

# 2) wait until the endpoint answers (model load takes a few minutes)
until $RUN_PY -c "from openai import OpenAI; OpenAI(api_key='EMPTY', base_url='http://127.0.0.1:8001/v1').models.list()" 2>/dev/null; do
  echo "waiting for vLLM to come up..."; sleep 30
done
echo "vLLM is up"

mkdir -p ../baseline/drtriton/kbstyle_stts_full8_20260922

# 3) STTS: 7 tasks, 1 sample, patience=0 (no early stop, full 10 iterations)
#    (strict protocol alignment, same as README full run)
exec $RUN_PY kbstyle_stts.py \
  --samples 1 --iterations 10 --patience 0 \
  --task flashinfer/gemm_n4096_k4096 \
  --task flashinfer/mla_paged_decode_h16_ckv512_kpe64_ps1 \
  --task flashinfer/rmsnorm_h4096 \
  --task SOL/L1/008_expert_output_weighted_index_add_accumulation \
  --task SOL/L1/053_gaussian_topk_sparse_activation \
  --task SOL/L1/058_moe_expert_token_radix_sort_with_prefix_sum \
  --task SOL/L2/030_flux_concatenated_sequence_processing_with_split \
  --servers http://127.0.0.1:8001/v1 \
  --gpus 2,5 --max-parallel 2 --concurrent-tasks 4 \
  2>&1 | tee -a "$(dirname "$0")/../baseline/drtriton/kbstyle_stts_full8_20260922/campaign.log"
