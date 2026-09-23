#!/usr/bin/env bash
# 池化并发搜索编排：N 个任务并发（LLM 阶段不占卡），benchmark 经 KSEARCH_GPU_POOL 排队。
# 续跑感知（有 WM → 剩余轮数预算 + --continue-from-world-model auto）；成功打 DONE。
#
# 用法: bash ksearch_pool_campaign.sh --pool "5,6" --concurrency 4 [--tag TAG] --tasks <t1> <t2> ...
#   环境: KSEARCH_WM_MAX_ACTION_NODES 等照常透传；不设 CUDA_VISIBLE_DEVICES（由池动态分配）
set -o pipefail
WS=/home/ziming/ksearch_h100_portable
POOL=""; CONC=4; TAG=""; FORCE=0; TASKS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --pool) POOL="${2:?}"; shift 2 ;;
    --concurrency) CONC="${2:?}"; shift 2 ;;
    --tag) TAG="${2:?}"; shift 2 ;;
    --tasks) shift; while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do TASKS+=("$1"); shift; done ;;
    --force) FORCE=1; shift ;;
    *) echo "未知参数 $1"; exit 2 ;;
  esac
done
[[ -z "$POOL" || ${#TASKS[@]} -eq 0 ]] && { echo "需要 --pool 与 --tasks"; exit 2; }
[[ -z "$TAG" ]] && TAG="pool_$(date +%Y%m%d_%H%M)"

run_dir_for() {
  local def="$1" name="${1##*/}"
  # 目录约定同 ksearch-run.sh 的自动判定：L1/L2 前缀为 SOL 任务，其余为 FlashInfer 任务
  if [[ "${def}" =~ ^L[0-9]+/ ]]; then
    echo "${WS}/baseline/ksearch-sol-execbench/experiments/${TAG}/${name}/run_seed0"
  else
    echo "${WS}/baseline/ksearch/experiments/${TAG}/${name}/run_seed0"
  fi
}

echo "[pool-campaign] pool=${POOL} concurrency=${CONC} tag=${TAG} tasks=${#TASKS[@]}"
# 同 campaign 共享 owner 戳：各任务的 gpu_pool 身份识别互认"自己人"，
# 否则彼此留在卡上的父进程上下文会被误判为陌生租户，造成卡池独占/饿死
export KSEARCH_OWNER_TAG="ksearch-campaign-${TAG}-$(date +%Y%m%d%H%M%S)-$$"
declare -A PID2TASK PID2RD
declare -A RETRIES   # 自愈：异常退出的任务重新排队（每题最多 3 次）
MAX_RETRY=3

launch_one() { # $1=task
  local t="$1" name="${1##*/}" rd
  rd="$(run_dir_for "$t")"
  if [[ ${FORCE} -eq 0 && -f "${rd}/DONE" ]]; then echo "[pool] SKIP ${t} (DONE)"; return 1; fi
  mkdir -p "${rd}"; touch "${rd}/campaign_stdout.log"
  local rounds=100 extra=()
  if [[ -f "${rd}/ksearch-artifacts/${name}/world_model/world_model.json" ]]; then
    # 预算按"完成评测数"计（FI=Round summary 行，SOL=feedback workloads passed 行）。
    # 旧版数"轮次启动横幅"——横幅在生成前打印，任务被杀会灌水横幅而不产生评测，
    # 曾致预算提前耗尽、题目在 21-75 次真实评测时即被标 DONE（2026-09-21 修正）
    local done_r; done_r=$(grep -cE "Round summary|feedback workloads passed" "${rd}/campaign_stdout.log" 2>/dev/null); done_r=${done_r:-0}
    rounds=$((100 - done_r)); [[ $rounds -lt 5 ]] && rounds=5
    extra=(--continue-from-world-model auto)
    echo "[pool] $(date '+%m%d %H:%M') RESUME ${t} (done_evals=${done_r} budget=${rounds})"
  else
    echo "[pool] $(date '+%m%d %H:%M') FRESH ${t}"
  fi
  # 任务-卡终身绑定（sticky affinity）：按任务序号轮转分配一张卡，整个生命周期
  # 不变。父进程 CUDA_VISIBLE_DEVICES 出生即钉死在自己的卡上——杜绝跨卡张量
  # 错位、跨卡缓存池累计、以及一切"漂移"类竞态（2026-09-20 六次故障的共同根因）
  local pool_arr=(${POOL//,/ })
  # 显式映射优先（2026-09-21 承接式调整：按剩余工作量装箱，未列出的任务回落轮转）。
  # 映射中的卡号必须属于本 campaign 的 --pool，否则同样回落轮转。
  local sticky=""
  if [[ -n "${TASK_GPU_MAP[${t}]}" ]]; then
    local want="${TASK_GPU_MAP[${t}]}"
    for c in "${pool_arr[@]}"; do [[ "${c}" == "${want}" ]] && sticky="${want}" && break; done
  fi
  if [[ -z "${sticky}" ]]; then
    sticky="${pool_arr[$(( LAUNCH_SEQ % ${#pool_arr[@]} ))]}"
  fi
  LAUNCH_SEQ=$((LAUNCH_SEQ + 1))
  echo "[pool] $(date '+%m%d %H:%M') AFFINITY ${t} -> gpu${sticky}"
  (
    env KSEARCH_GPU_POOL="${sticky}" KSEARCH_TASK_GPU="${sticky}" \
        KSEARCH_RUN_TAG="${TAG}" \
        KSEARCH_MAX_ROUNDS="${rounds}" \
        KSEARCH_WM_MAX_ACTION_NODES="${KSEARCH_WM_MAX_ACTION_NODES:-20}" \
        KSEARCH_WM_MAX_ATTEMPTS_PER_NODE="${KSEARCH_WM_MAX_ATTEMPTS_PER_NODE:-5}" \
        KSEARCH_POOL_TASK="${t}" \
      bash "${WS}/ksearch-run.sh" "${t}" 0 --wm "${extra[@]}" \
        >> "${rd}/campaign_stdout.log" 2>&1
    echo $? > "${rd}/exit_code"
  ) &
  PID2TASK[$!]="${t}"; PID2RD[$!]="${rd}"
  return 0
}
LAUNCH_SEQ=0
# 显式任务→卡映射（2026-09-21 14:50 布局：卡0=dsa+gemm+gqa_ragged+mla_paged_prefill，
# 卡1=gdn_decode+gqa_paged_decode+gqa_paged_prefill+mla_paged_decode，卡2=gdn_prefill 独占。
# gqa_ragged 原进程 LLM 调用挂死 26min，随本次重启一并换新进程）
declare -A TASK_GPU_MAP=(
  ["dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64"]="1"
  ["gemm_n4096_k4096"]="1"
  ["gqa_ragged_prefill_causal_h32_kv8_d128"]="1"
  ["gdn_decode_qk4_v8_d128_k_last"]="1"
  ["gqa_paged_decode_h32_kv8_d128_ps1"]="5"
  ["gqa_paged_prefill_causal_h32_kv8_d128_ps1"]="4"
  ["mla_paged_decode_h16_ckv512_kpe64_ps1"]="4"
  ["mla_paged_prefill_causal_h16_ckv512_kpe64_ps1"]="0"
  ["gdn_prefill_qk4_v8_d128_k_last"]="5"
  ["rmsnorm_h4096"]="1"
  # L1 收回单卡4（2026-09-21 19:25：生成是瓶颈单卡不减速，卡3/5整卡划给终评专用道）
  ["L1/002_vae_conv3x3_groupnorm_silu_residual_fused"]="4"
  ["L1/018_fused_rope_with_qk_norm_and_kv_cache_update"]="4"
  ["L1/053_gaussian_topk_sparse_activation"]="4"
  ["L1/020_vision_patch_merger_spatial_shuffle_mlp"]="4"
  ["L1/058_moe_expert_token_radix_sort_with_prefix_sum"]="4"
  ["L1/092_gqa_attention_with_qk_norm"]="5"
  ["L1/070_mamba2_fused_intra_chunk_diagonal_computation"]="4"
)

queued=("${TASKS[@]}")
results_log="$(mktemp /tmp/pool_campaign_XXXX.log)"
while true; do
  # 回收结束的进程
  for pid in "${!PID2TASK[@]}"; do
    if ! kill -0 "${pid}" 2>/dev/null; then
      wait "${pid}" 2>/dev/null || true
      rd="${PID2RD[${pid}]}"; t="${PID2TASK[${pid}]}"
      rc=127; [[ -f "${rd}/exit_code" ]] && rc="$(cat "${rd}/exit_code")"
      if [[ ${rc} -eq 0 ]]; then
        touch "${rd}/DONE"; echo "[pool] OK    ${t}" | tee -a "${results_log}"
      else
        r=${RETRIES[${t}]:-0}; r=$((r+1)); RETRIES[${t}]=$r
        if [[ ${r} -le ${MAX_RETRY} ]]; then
          queued+=("${t}")
          echo "[pool] FAIL  ${t} rc=${rc} -> requeue (retry ${r}/${MAX_RETRY})" | tee -a "${results_log}"
        else
          echo "[pool] FAIL  ${t} rc=${rc} (log: ${rd}/campaign_stdout.log) retries exhausted" | tee -a "${results_log}"
        fi
      fi
      unset PID2TASK[${pid}] PID2RD[${pid}]
    fi
  done
  # 补位
  while [[ ${#PID2TASK[@]} -lt ${CONC} && ${#queued[@]} -gt 0 ]]; do
    next="${queued[0]}"; queued=("${queued[@]:1}")
    launch_one "${next}" || continue
  done
  [[ ${#PID2TASK[@]} -eq 0 && ${#queued[@]} -eq 0 ]] && break
  sleep 60
done
echo "================ pool campaign summary ================"
cat "${results_log}"; rm -f "${results_log}"
