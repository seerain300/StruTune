#!/usr/bin/env bash
set -euo pipefail

stage=${1:?usage: evaluate_candidate.sh feedback|final cNNN}
candidate=${2:?usage: evaluate_candidate.sh feedback|final cNNN}

export KDA_WORKSPACE=/data1/workspace/weihongren/kda-runs/formal-kda-20260916--sol_execbench--L1-020_vision_patch_merger_spatial_shuffle_mlp
exec /data1/workspace/weihongren/bin/kda-eval "$stage" --candidate "$candidate"
