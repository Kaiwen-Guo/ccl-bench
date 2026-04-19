#!/usr/bin/env bash
set -euo pipefail

TRACE_ROOT="${TRACE_ROOT:-/pscratch/sd/k/kg597/ccl-bench-traces/llama-deepseek-kineto-mscclpp}"
mkdir -p "$TRACE_ROOT"

salloc \
  --nodes 1 \
  --qos interactive \
  --time "${SALLOC_TIME:-02:00:00}" \
  --constraint gpu \
  --gpus 4 \
  --account m4999 \
  srun --nodes 1 --ntasks 1 --gpus 4 --gpu-bind=none bash -lc '
    set -euo pipefail
    module load conda
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate /pscratch/sd/k/kg597/session1/cvllm
    cd "$HOME/ccl-bench"
    mkdir -p /pscratch/sd/k/kg597/ccl-bench-traces/llama-deepseek-kineto-mscclpp
    bash scripts/run_llama_deepseek_kineto_mscclpp_perlmutter.sh
    bash scripts/make_llama_deepseek_kineto_mscclpp_bundles_perlmutter.sh
  '
