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

WS=/home/ziming/ksearch_h100_portable
source "${WS}/activate-ksearch.sh"
# Key 来源：bashrc 的 K_SEARCH_KEY（本机约定，不用 llm.env）。Key 优先级：
#   1) 调用方预设 LLM_API_KEY（按批次临时换 key）
#   2) bashrc 的 K_SEARCH_KEY（K-Search 专用 key，全局默认）
_CALLER_KEY="${LLM_API_KEY:-}"
if [[ -z "${_CALLER_KEY}" && -z "${K_SEARCH_KEY:-}" ]]; then
  # 非交互 shell 下 bashrc 的 interactive 早退会跳过 K_SEARCH_KEY，直接从文件提取
  K_SEARCH_KEY="$(grep -E '^export K_SEARCH_KEY=' "${HOME}/.bashrc" | tail -1 | cut -d= -f2- | tr -d '"')"
  export K_SEARCH_KEY
fi
if [[ -n "${K_SEARCH_KEY:-}" ]]; then export LLM_API_KEY="${K_SEARCH_KEY}"; fi
if [[ -n "${_CALLER_KEY}" ]]; then export LLM_API_KEY="${_CALLER_KEY}"; fi
# 端点/模型默认值（可被调用方环境变量覆盖）
export LLM_BASE_URL="${LLM_BASE_URL:-https://llmapi.isrc.ac.cn/v1}"
export LLM_MODEL="${LLM_MODEL:-GPT-5.6-Sol}"
# 反馈评测 -inf 哨兵补丁（经 sitecustomize 注入 isolated runner 的 spawn 子进程）
export PYTHONPATH="${WS}/ksearch_patchshim${PYTHONPATH:+:${PYTHONPATH}}"
# 所有者戳：优先继承 campaign 层的共享戳（同 campaign 各任务互认"自己人"，
# 避免彼此的父进程上下文/评测子进程被误判为陌生租户导致卡池饿死）；
# 单任务直跑时才自动生成。gpu_pool 据此把"我们的 GPU 进程"与同事/陌生租户区分开
export KSEARCH_OWNER_TAG="${KSEARCH_OWNER_TAG:-ksearch-${USER}-$(date +%Y%m%d%H%M%S)-$$}"
export PYTHONHASHSEED="${SEED}"

# 题目来源自动判定：DEF 形如 L1/094_...（SOL-ExecBench）时切到 sol task source
TASK_SOURCE="flashinfer"
TASKSET="/home/ziming/dataset/flashinfer-test"
ART_BASE="${WS}/baseline/ksearch"
DEF_KEY="${DEF}"
if [[ "${DEF}" =~ ^L[0-9]+/ ]]; then
  TASK_SOURCE="sol"
  TASKSET="/home/ziming/dataset/SOL-ExecBench"
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

# GPU 可见性约束（终版：任务-卡终身绑定）：
#   1) CUDA_DEVICE_ORDER=PCI_BUS_ID：CUDA 序号 = nvidia-smi 物理序号
#   2) KSEARCH_TASK_GPU（campaign 轮转分配，终身不变）：进程出生即钉死在自己的卡，
#      父/子进程都只可能碰这张卡——跨卡张量错位、缓存池累计、漂移竞态从根上不存在
export CUDA_DEVICE_ORDER=PCI_BUS_ID
if [[ -n "${KSEARCH_TASK_GPU:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${KSEARCH_TASK_GPU}"
  export KSEARCH_GPU_POOL="${KSEARCH_TASK_GPU}"
  echo "[ksearch-run] sticky affinity -> gpu${KSEARCH_TASK_GPU}"
elif [[ -n "${KSEARCH_GPU_POOL:-}" && -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${KSEARCH_GPU_POOL}"
  echo "[ksearch-run] pool mode: parent CUDA_VISIBLE_DEVICES default -> ${CUDA_VISIBLE_DEVICES}"
fi

# 反馈 ref 延迟缓存开关（协议 v0.6：仅"不缓存时单轮 ref 成本 >30s"的题开启；
# A800 实测类比 H100，清单见 notes/REF_LATENCY_CACHE_STRATEGY.md §5）。
# 最终精确评测不受影响（始终实时成对计时）。
case "${DEF}" in
  gdn_prefill_qk4_v8_d128_k_last|mla_paged_prefill_causal_h16_ckv512_kpe64_ps1|\
  gqa_paged_prefill_causal_h32_kv8_d128_ps1|gqa_paged_decode_h32_kv8_d128_ps1|\
  gdn_decode_qk4_v8_d128_k_last|L2/036_convnextv2_layer_with_nhwc_persistence_backward)
    export KSEARCH_REF_CACHE=1 ;;
  *)
    export KSEARCH_REF_CACHE=0 ;;
esac

# 反馈评测参数（协议 v0.6）：全量 workload（--num-feedback-workloads 默认全量）；
# iterations 默认 20，慢 reference 题降到 10（首轮 ref 计时之后有磁盘缓存，后续轮只计时候选）。
# 最终精确评测不受此影响：仍为全量 workload + 100 iterations（ksearch_final_eval.py 口径）。
case "${DEF}" in
  gdn_prefill_qk4_v8_d128_k_last|mla_paged_prefill_causal_h16_ckv512_kpe64_ps1|gqa_paged_prefill_causal_h32_kv8_d128_ps1|gqa_paged_decode_h32_kv8_d128_ps1|L2/036_convnextv2_layer_with_nhwc_persistence_backward)
    FEEDBACK_ITERS=10 ;;
  *)
    FEEDBACK_ITERS=20 ;;
esac
if [[ "${TASK_SOURCE}" == "sol" ]]; then
  BENCH_ARGS=(--warmup-runs 10 --iterations "${FEEDBACK_ITERS}" --num-trials 1)
else
  BENCH_ARGS=(--warmup-runs 3 --iterations "${FEEDBACK_ITERS}" --num-trials 5)
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

cd /home/ziming/K-Search
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
  --target-gpu H100 \
  --max-opt-rounds "${ROUNDS}" \
  --seed "${SEED}" \
  "${BENCH_ARGS[@]}" \
  --use-isolated-runner \
  --save-solutions \
  --no-save-results \
  --artifacts-dir "${ART}/ksearch-artifacts" \
  "${WM_ARGS[@]}" "${FINAL_EVAL_ARGS[@]}" "$@" 2>&1 | tee "${ART}/stdout_$(date +%Y%m%d_%H%M%S).log"
