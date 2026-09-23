#!/usr/bin/env bash
# K-Search 批量编排骨本：自动探测全空 GPU 并行调度，动态补位，断点续跑。
#
# 用法:
#   ./scripts/ksearch_campaign.sh --all20 [--gpus auto] [--rounds 100] [--seed 0] [--wm] [--tag TAG] [--force]
#   ./scripts/ksearch_campaign.sh --gpus 1,2,4 --tasks rmsnorm_h4096 L1/094_time_decay_exponential_stabilization
#
# GPU 调度:
#   --gpus auto（默认）: 启动时探测全部"全空"卡（显存占用 < 阈值），运行期间定期
#   复扫，新腾空的卡自动加入池子；每次启动任务前复查该卡仍为空，被外部占用则移出。
#   --gpus 1,2,4       : 手动指定（同样做启动前复查）。
#   空卡阈值: GPU_EMPTY_THRESH_MIB（默认 200 MiB）；复扫间隔: GPU_RESCAN_SEC（默认 300）。
#
# 其他选项:
#   --tasks <defs...> / --flashinfer10 / --sol10 / --all20
#   --rounds <n>   KSEARCH_MAX_ROUNDS（默认 20）
#   --seed <n>     search seed（默认 0）
#   --wm / --no-wm world-model 开关（默认开）
#   --tag <TAG>    KSEARCH_RUN_TAG 实验隔离标签
#   --force        忽略 DONE 标记强制重跑
#
# 预算环境变量原样透传: KSEARCH_WM_MAX_ACTION_NODES / KSEARCH_WM_MAX_ATTEMPTS_PER_NODE /
# KSEARCH_WM_STAGNATION_WINDOW / KSEARCH_FINAL_EVAL 等。
# 注: 不用 set -u —— 本机 bash 对空关联数组 + set -u 会误报 unbound variable。
set -o pipefail

WS=/home/ziming/ksearch_h100_portable
RUN="${WS}/ksearch-run.sh"

GPUS="auto"
TASKS=()
ROUNDS=20
SEED=0
WM=1
TAG=""
FORCE=0
EMPTY_THRESH="${GPU_EMPTY_THRESH_MIB:-200}"
RESCAN_SEC="${GPU_RESCAN_SEC:-300}"

FLASHINFER_TASKS=(
  dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64
  gdn_decode_qk4_v8_d128_k_last
  gdn_prefill_qk4_v8_d128_k_last
  gemm_n4096_k4096
  gqa_paged_decode_h32_kv8_d128_ps1
  gqa_paged_prefill_causal_h32_kv8_d128_ps1
  gqa_ragged_prefill_causal_h32_kv8_d128
  mla_paged_decode_h16_ckv512_kpe64_ps1
  mla_paged_prefill_causal_h16_ckv512_kpe64_ps1
  rmsnorm_h4096
)
SOL_TASKS=(
  L1/002_vae_conv3x3_groupnorm_silu_residual_fused
  L1/005_conv_gated_projection_with_causal_conv
  L1/007_hyena_fft_size_padding_rfft
  L1/008_expert_output_weighted_index_add_accumulation
  L1/018_fused_rope_with_qk_norm_and_kv_cache_update
  L1/020_vision_patch_merger_spatial_shuffle_mlp
  L1/053_gaussian_topk_sparse_activation
  L1/058_moe_expert_token_radix_sort_with_prefix_sum
  L1/070_mamba2_fused_intra_chunk_diagonal_computation
  L1/094_time_decay_exponential_stabilization
)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus) GPUS="${2:?--gpus 需要参数}"; shift 2 ;;
    --tasks) shift; while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do TASKS+=("$1"); shift; done ;;
    --flashinfer10) TASKS+=("${FLASHINFER_TASKS[@]}"); shift ;;
    --sol10) TASKS+=("${SOL_TASKS[@]}"); shift ;;
    --all20) TASKS+=("${FLASHINFER_TASKS[@]}" "${SOL_TASKS[@]}"); shift ;;
    --rounds) ROUNDS="${2:?}"; shift 2 ;;
    --seed) SEED="${2:?}"; shift 2 ;;
    --wm) WM=1; shift ;;
    --no-wm) WM=0; shift ;;
    --tag) TAG="${2:?}"; shift 2 ;;
    --force) FORCE=1; shift ;;
    *) echo "未知参数: $1" >&2; exit 2 ;;
  esac
done

if [[ ${#TASKS[@]} -eq 0 ]]; then echo "错误: 需要 --tasks / --flashinfer10 / --sol10 / --all20" >&2; exit 2; fi

# 与 ksearch-run.sh 保持一致的 run 目录推导，用于 DONE 标记判断
run_dir_for() {
  local def="$1" base def_key
  if [[ "${def}" =~ ^L[0-9]+/ ]]; then
    base="${WS}/baseline/ksearch-sol-execbench"; def_key="${def##*/}"
  else
    base="${WS}/baseline/ksearch"; def_key="${def}"
  fi
  if [[ -n "${TAG}" ]]; then
    echo "${base}/experiments/${TAG}/${def_key}/run_seed${SEED}"
  else
    echo "${base}/${def_key}/run_seed${SEED}"
  fi
}

# ---------- GPU 探测 ----------
list_empty_gpus() {  # 输出: 每行一个空卡编号
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
    | awk -F', *' -v th="${EMPTY_THRESH}" '$2+0 < th+0 {print $1+0}'
}

gpu_is_empty() { # $1=gpu id
  local used
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" 2>/dev/null | tr -d ' ')
  [[ "${used}" =~ ^[0-9]+$ ]] && [[ "${used}" -lt "${EMPTY_THRESH}" ]]
}

declare -A IN_POOL=()   # gpu -> 1（在池中）
declare -A BUSY=()      # gpu -> 1（正在跑任务）

rescan_pool() {  # 把新出现的空卡加入池子（不移除 busy 的）
  local g added=0
  while read -r g; do
    [[ -z "${g}" ]] && continue
    if [[ -z "${IN_POOL[${g}]:-}" ]]; then
      IN_POOL[${g}]=1
      added=$((added+1))
      echo "[campaign] RESCAN gpu${g} 加入池子"
    fi
  done < <(list_empty_gpus)
  return 0
}

if [[ "${GPUS}" == "auto" ]]; then
  while read -r g; do IN_POOL[${g}]=1; done < <(list_empty_gpus)
  if [[ ${#IN_POOL[@]} -eq 0 ]]; then
    echo "错误: auto 模式下没有探测到全空 GPU（阈值 ${EMPTY_THRESH} MiB）" >&2
    exit 2
  fi
else
  IFS=',' read -r -a MANUAL_ARR <<< "${GPUS}"
  for g in "${MANUAL_ARR[@]}"; do IN_POOL[${g}]=1; done
fi

WM_FLAG=""
[[ ${WM} -eq 1 ]] && WM_FLAG="--wm"
TAG_ENV=()
[[ -n "${TAG}" ]] && TAG_ENV=(KSEARCH_RUN_TAG="${TAG}")

echo "[campaign] gpu_pool=${!IN_POOL[*]} tasks=${#TASKS[@]} rounds=${ROUNDS} seed=${SEED} wm=${WM} tag=${TAG:-none} force=${FORCE}"
echo "[campaign] wm_env: MAX_ACTION_NODES=${KSEARCH_WM_MAX_ACTION_NODES:-unbounded} MAX_ATTEMPTS_PER_NODE=${KSEARCH_WM_MAX_ATTEMPTS_PER_NODE:-unbounded} STAGNATION=${KSEARCH_WM_STAGNATION_WINDOW:-5}"

declare -A PID2TASK PID2GPU PID2T0 PID2RUNDIR
queued=("${TASKS[@]}")
results_log="$(mktemp /tmp/ksearch_campaign_XXXXXX.log)"
LAST_RESCAN=$(date +%s)

launch() { # $1=task $2=gpu
  local task="$1" gpu="$2" rd
  rd="$(run_dir_for "${task}")"
  mkdir -p "${rd}"
  BUSY[${gpu}]=1
  local t0=$(date +%s)
  rm -f "${rd}/exit_code"
  (
    env CUDA_VISIBLE_DEVICES="${gpu}" \
      KSEARCH_MAX_ROUNDS="${ROUNDS}" \
      "${TAG_ENV[@]}" \
      bash "${RUN}" "${task}" "${SEED}" ${WM_FLAG} \
      > "${rd}/campaign_stdout.log" 2>&1
    echo $? > "${rd}/exit_code"
  ) &
  PID2TASK[$!]="${task}"
  PID2GPU[$!]="${gpu}"
  PID2T0[$!]="${t0}"
  PID2RUNDIR[$!]="${rd}"
  echo "[campaign] LAUNCH ${task} on gpu${gpu} -> ${rd}"
}

reap() { # 等任一子进程结束并回收 GPU 槽位
  local pid rc dur task gpu rd
  wait -n || true
  sleep 1  # 给 exit_code 文件落盘留时间
  for pid in "${!PID2TASK[@]}"; do
    if ! kill -0 "${pid}" 2>/dev/null; then
      wait "${pid}" 2>/dev/null || true
      task="${PID2TASK[${pid}]}"
      gpu="${PID2GPU[${pid}]}"
      rd="${PID2RUNDIR[${pid}]}"
      dur=$(( $(date +%s) - ${PID2T0[${pid}]} ))
      rc=127
      [[ -f "${rd}/exit_code" ]] && rc="$(cat "${rd}/exit_code")"
      if [[ ${rc} -eq 0 ]]; then
        touch "${rd}/DONE"
        echo "[campaign] OK    ${task} rc=0 wall=${dur}s" | tee -a "${results_log}"
      else
        echo "[campaign] FAIL  ${task} rc=${rc} wall=${dur}s (log: ${rd}/campaign_stdout.log)" | tee -a "${results_log}"
      fi
      unset BUSY[${gpu}] PID2TASK[${pid}] PID2GPU[${pid}] PID2T0[${pid}] PID2RUNDIR[${pid}]
    fi
  done
}

maybe_rescan() {
  # 复扫仅在 auto 模式启用：手动 --gpus 指定的是"只准用这些卡"的硬约束，
  # 不应把别的租户腾出的卡自动吸进池子（2026-09-16 事故：--gpus 1,2 被扩到 6 卡）。
  [[ "${GPUS}" != "auto" ]] && return 0
  local now=$(date +%s)
  if [[ $((now - LAST_RESCAN)) -ge ${RESCAN_SEC} ]]; then
    rescan_pool
    LAST_RESCAN=${now}
  fi
}

acquire_gpu() { # 输出一个可用的空 gpu 编号（启动前复查仍为空）；无则返回 1
  local g
  for g in "${!IN_POOL[@]}"; do
    [[ -n "${BUSY[${g}]:-}" ]] && continue
    if gpu_is_empty "${g}"; then
      echo "${g}"
      return 0
    else
      echo "[campaign] WARN gpu${g} 被外部占用，移出池子"
      unset IN_POOL[${g}]
    fi
  done
  return 1
}

for task in "${queued[@]}"; do
  rd="$(run_dir_for "${task}")"
  if [[ ${FORCE} -eq 0 && -f "${rd}/DONE" ]]; then
    echo "[campaign] SKIP  ${task##*/} (DONE marker exists)"
    continue
  fi
  # 等待空闲 GPU（期间回收已结束的任务并复扫新空卡）
  while ! acquire_gpu > /dev/null; do
    if [[ ${#PID2TASK[@]} -eq 0 ]]; then
      maybe_rescan
      acquire_gpu > /dev/null || { echo "[campaign] 没有可用 GPU，等待 ${RESCAN_SEC}s 后复扫..." >&2; sleep "${RESCAN_SEC}"; }
    else
      reap
      maybe_rescan
    fi
  done
  gpu="$(acquire_gpu)"
  launch "${task}" "${gpu}"
done

while [[ ${#PID2TASK[@]} -gt 0 ]]; do reap; done

echo "================ campaign summary ================"
cat "${results_log}"
echo "ok=$(grep -c '^\[campaign\] OK' "${results_log}" || true) fail=$(grep -c '^\[campaign\] FAIL' "${results_log}" || true)"
rm -f "${results_log}"
