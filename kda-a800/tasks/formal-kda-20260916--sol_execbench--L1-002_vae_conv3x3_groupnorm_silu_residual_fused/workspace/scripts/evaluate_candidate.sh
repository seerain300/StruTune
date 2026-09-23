#!/usr/bin/env bash
set -euo pipefail

stage=${1:?usage: evaluate_candidate.sh feedback|final cNNN}
candidate=${2:?usage: evaluate_candidate.sh feedback|final cNNN}

export KDA_WORKSPACE=/data1/workspace/weihongren/kda-runs/formal-kda-20260916--sol_execbench--L1-002_vae_conv3x3_groupnorm_silu_residual_fused
exec /data1/workspace/weihongren/bin/kda-eval "$stage" --candidate "$candidate"
