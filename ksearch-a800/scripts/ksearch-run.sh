#!/usr/bin/env bash
# K-Search baseline 统一入口
# 用法: ./ksearch-run.sh <definition名> <seed0|1|2> [--wm] [额外透传参数...]
#   --wm            启用 world-model（K-Search 完整版；不加则为普通迭代生成器 ablation）
# 环境变量:
#   KSEARCH_MAX_ROUNDS                 候选实现总数上限（默认 20）
#   KSEARCH_WM_MAX_ACTION_NODES        world-model 节点数上限（默认不另设限）
#   KSEARCH_WM_MAX_ATTEMPTS_PER_NODE   每个节点的候选生成/评测上限（默认不另设限）
#   KSEARCH_WM_STAGNATION_WINDOW       节点内连续无提升时的提前停止阈值（默认 5）
#   KSEARCH_RUN_TAG                    独立实验标签（设置后与 smoke/其他预实验隔离）
#   KSEARCH_FINAL_EVAL                 1=生成后立即全量评测；默认 0，留待统一评测
#   EXTRA: 其余参数原样透传给 generate_kernels_and_eval.py
set -euo pipefail

DEF="${1:?用法: ksearch-run.sh <definition> <seed> [--wm] [extra...]}"
SEED_ARG="${2:?缺少 seed (0/1/2 或 seed0/seed1/seed2)}"
SEED="${SEED_ARG#seed}"
if [[ ! "${SEED}" =~ ^[0-9]+$ ]]; then
  echo "错误: seed 必须是非负整数或 seedN，收到: ${SEED_ARG}" >&2
  exit 2
fi
shift 2

WM=0
if [[ "${1:-}" == "--wm" ]]; then WM=1; shift; fi

WS=/data1/workspace/weihongren
source "${WS}/activate-ksearch.sh"
# llm.env 提供默认端点/模型/key。Key 优先级：
#   1) 调用方预设 LLM_API_KEY（按批次临时换 key）
#   2) llm.env 的 K_SEARCH_KEY（K-Search 专用 key，全局默认）
#   3) llm.env 的 LLM_API_KEY（兜底）
_CALLER_KEY="${LLM_API_KEY:-}"
source "${WS}/llm.env"
if [[ -n "${K_SEARCH_KEY:-}" ]]; then export LLM_API_KEY="${K_SEARCH_KEY}"; fi
if [[ -n "${_CALLER_KEY}" ]]; then export LLM_API_KEY="${_CALLER_KEY}"; fi
# 反馈评测 -inf 哨兵补丁（经 sitecustomize 注入 isolated runner 的 spawn 子进程）
export PYTHONPATH="${WS}/ksearch_patchshim${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONHASHSEED="${SEED}"

# 题目来源自动判定：DEF 形如 L1/094_...（SOL-ExecBench）时切到 sol task source
TASK_SOURCE="flashinfer"
TASKSET="${WS}/dataset/flashinfer-test"
ART_BASE="${WS}/baseline/ksearch"
DEF_KEY="${DEF}"
if [[ "${DEF}" =~ ^L[0-9]+/ ]]; then
  TASK_SOURCE="sol"
  TASKSET="/data1/workspace/ziming/dataset/SOL-ExecBench"
  ART_BASE="${WS}/baseline/ksearch-sol-execbench"
  DEF_KEY="${DEF##*/}"
fi
RUN_TAG="${KSEARCH_RUN_TAG:-}"
if [[ -n "${RUN_TAG}" ]]; then
  if [[ ! "${RUN_TAG}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "错误: KSEARCH_RUN_TAG 只能包含字母、数字、点、下划线和连字符，收到: ${RUN_TAG}" >&2
    exit 2
  fi
  ART="${ART_BASE}/experiments/${RUN_TAG}/${DEF_KEY}/run_seed${SEED}"
else
  ART="${ART_BASE}/${DEF_KEY}/run_seed${SEED}"
fi
mkdir -p "${ART}"

ROUNDS="${KSEARCH_MAX_ROUNDS:-20}"

# 评测计时参数与 ziming 统一评测器对齐（协议 v0.4）：
#   FlashInfer（evaluate.py 口径）: warmup 3 / iterations 100 / trials 5
#   SOL（evaluate_sol.py 口径）   : warmup 10（SOL 官方默认，脚本不覆写）/ iterations 100
# 与统一评测唯一的差别是 workload 数：搜索反馈只跑固定 seed 抽样的 5 个。
if [[ "${TASK_SOURCE}" == "sol" ]]; then
  BENCH_ARGS=(--warmup-runs 10 --iterations 100 --num-trials 1)
else
  BENCH_ARGS=(--warmup-runs 3 --iterations 100 --num-trials 5)
fi

WM_ARGS=()
FINAL_EVAL_ARGS=()
RUN_VARIANT="plain"
if [[ ${WM} -eq 1 ]]; then
  WM_ARGS=(--world-model --wm-stagnation-window "${KSEARCH_WM_STAGNATION_WINDOW:-5}")
  if [[ -n "${KSEARCH_WM_MAX_ACTION_NODES:-}" ]]; then
    WM_ARGS+=(--wm-max-action-nodes "${KSEARCH_WM_MAX_ACTION_NODES}")
  fi
  if [[ -n "${KSEARCH_WM_MAX_ATTEMPTS_PER_NODE:-}" ]]; then
    WM_ARGS+=(--wm-max-attempts-per-node "${KSEARCH_WM_MAX_ATTEMPTS_PER_NODE}")
  fi
  RUN_VARIANT="world-model"
fi

if [[ "${KSEARCH_FINAL_EVAL:-0}" != "1" ]]; then
  FINAL_EVAL_ARGS=(--skip-final-eval)
fi

echo "[ksearch-run] def=${DEF} seed=${SEED} wm=${WM} rounds=${ROUNDS} model=${LLM_MODEL}"
echo "[ksearch-run] task_source=${TASK_SOURCE} taskset=${TASKSET}"
echo "[ksearch-run] run_tag=${RUN_TAG:-default}"
echo "[ksearch-run] final_eval=${KSEARCH_FINAL_EVAL:-0}"
if [[ ${WM} -eq 1 ]]; then
  echo "[ksearch-run] wm_nodes=${KSEARCH_WM_MAX_ACTION_NODES:-unbounded} wm_attempts_per_node=${KSEARCH_WM_MAX_ATTEMPTS_PER_NODE:-unbounded} wm_stagnation=${KSEARCH_WM_STAGNATION_WINDOW:-5}"
fi
echo "[ksearch-run] artifacts -> ${ART}"

KSEARCH_USAGE_LOG="${ART}/usage.jsonl" \
LLM_USAGE_METHOD="ksearch" \
LLM_USAGE_RUN_ID="${DEF}/seed${SEED}/${RUN_VARIANT}" \
python -u "${WS}/ksearch-token-run.py" \
  --task-source "${TASK_SOURCE}" \
  --task-path "${TASKSET}" \
  --definition "${DEF}" \
  --model-name "${LLM_MODEL}" \
  --base-url "${LLM_BASE_URL}" \
  --language triton \
  --target-gpu A800 \
  --max-opt-rounds "${ROUNDS}" \
  --seed "${SEED}" \
  "${BENCH_ARGS[@]}" \
  --use-isolated-runner \
  --save-solutions \
  --no-save-results \
  --artifacts-dir "${ART}/ksearch-artifacts" \
  "${WM_ARGS[@]}" "${FINAL_EVAL_ARGS[@]}" "$@" 2>&1 | tee "${ART}/stdout_$(date +%Y%m%d_%H%M%S).log"
