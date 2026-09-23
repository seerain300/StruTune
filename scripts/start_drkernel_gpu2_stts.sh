#!/usr/bin/env bash
# vLLM server for the kbstyle pass@1 experiment: same weights, eos fixed server-side.
set -euo pipefail

WS=/home/ziming/whr/run
PY=/home/qirui/miniconda3/envs/llama/bin/python
MODEL=/home/ziming/models/drkernel-8b
HOST=127.0.0.1
PORT=8002
SERVED_MODEL=drkernel-stop-8b
LOG_DIR="${WS}/baseline/drtriton/server"
LOG="${LOG_DIR}/vllm_gpu6_kbstyle_20260918.log"
PID_FILE="${LOG_DIR}/vllm_gpu6_kbstyle_20260918.pid"

mkdir -p "${LOG_DIR}"

if ss -lnt "( sport = :${PORT} )" | tail -n +2 | grep -q .; then
  echo "port ${PORT} is already in use (server may be running)" >&2
  exit 1
fi

while true; do
  used_mib=$(nvidia-smi -i 2 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  if [[ "${used_mib}" =~ ^[0-9]+$ ]] && (( used_mib <= 200 )); then
    break
  fi
  echo "GPU 2 busy (${used_mib} MiB used); waiting 30 seconds..."
  sleep 30
done

export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost"
export no_proxy="${no_proxy:+${NO_PROXY},}127.0.0.1,localhost"

SESSION=drkernel-gpu6-kbstyle
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session ${SESSION} already exists" >&2
  exit 1
fi

printf -v server_command \
  'exec env CUDA_VISIBLE_DEVICES=2 %q -m vllm.entrypoints.openai.api_server --model %q --host %q --port %q --served-model-name %q --max-model-len 32768 --max-num-seqs 16 --generation-config vllm --override-generation-config %q >>%q 2>&1' \
  "${PY}" "${MODEL}" "${HOST}" "${PORT}" "${SERVED_MODEL}" '{"eos_token_id":[151643,151645]}' "${LOG}"
tmux new-session -d -s "${SESSION}" "${server_command}"

server_pid=$(tmux display-message -p -t "${SESSION}" '#{pane_pid}')
echo "${server_pid}" >"${PID_FILE}"
echo "started tmux=${SESSION} PID=${server_pid}; log=${LOG}"
