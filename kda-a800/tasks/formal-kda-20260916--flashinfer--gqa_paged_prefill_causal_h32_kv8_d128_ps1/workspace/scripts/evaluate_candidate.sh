#!/usr/bin/env bash
set -euo pipefail

stage=${1:?usage: evaluate_candidate.sh feedback|final cNNN}
candidate=${2:?usage: evaluate_candidate.sh feedback|final cNNN}

export KDA_WORKSPACE=/data1/workspace/weihongren/kda-runs/formal-kda-20260916--flashinfer--gqa_paged_prefill_causal_h32_kv8_d128_ps1
exec /data1/workspace/weihongren/bin/kda-eval "$stage" --candidate "$candidate"
