#!/usr/bin/env bash
# 池化并发搜索编排：N 个任务并发（LLM 阶段不占卡），benchmark 经 KSEARCH_GPU_POOL 排队。
# 续跑感知（有 WM → 剩余轮数预算 + --continue-from-world-model auto）；成功打 DONE。
#
# 用法: bash ksearch_pool_campaign.sh --pool "5,6" --concurrency 4 [--tag TAG] --tasks <t1> <t2> ...
#   环境: KSEARCH_WM_MAX_ACTION_NODES 等照常透传；不设 CUDA_VISIBLE_DEVICES（由池动态分配）
set -o pipefail
WS=/data1/workspace/weihongren
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
  # SOL 任务（L1/L2 前缀）与 FlashInfer 任务目录约定同 ksearch_campaign.sh；
  # 本脚本当前仅用于 SOL 批次，FlashInfer 池化需先做亲和改造（见 GPU_POOL_DESIGN.md）。
  echo "${WS}/baseline/ksearch-sol-execbench/experiments/${TAG}/${name}/run_seed0"
}

echo "[pool-campaign] pool=${POOL} concurrency=${CONC} tag=${TAG} tasks=${#TASKS[@]}"
declare -A PID2TASK PID2RD

launch_one() { # $1=task
  local t="$1" name="${1##*/}" rd
  rd="$(run_dir_for "$t")"
  if [[ ${FORCE} -eq 0 && -f "${rd}/DONE" ]]; then echo "[pool] SKIP ${t} (DONE)"; return 1; fi
  mkdir -p "${rd}"; touch "${rd}/campaign_stdout.log"
  local rounds=100 extra=()
  if [[ -f "${rd}/ksearch-artifacts/${name}/world_model/world_model.json" ]]; then
    local done_r; done_r=$(grep -c "Optimization Round" "${rd}/campaign_stdout.log" 2>/dev/null); done_r=${done_r:-0}
    rounds=$((100 - done_r)); [[ $rounds -lt 5 ]] && rounds=5
    extra=(--continue-from-world-model auto)
    echo "[pool] $(date '+%m%d %H:%M') RESUME ${t} (done=${done_r} budget=${rounds})"
  else
    echo "[pool] $(date '+%m%d %H:%M') FRESH ${t}"
  fi
  (
    env KSEARCH_GPU_POOL="${POOL}" KSEARCH_RUN_TAG="${TAG}" \
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
        echo "[pool] FAIL  ${t} rc=${rc} (log: ${rd}/campaign_stdout.log)" | tee -a "${results_log}"
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
