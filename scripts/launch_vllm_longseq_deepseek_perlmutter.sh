#!/usr/bin/env bash
set -euo pipefail

TRACE_ROOT="${TRACE_ROOT:-$PSCRATCH/ccl-bench-traces/vllm-longseq-kineto}"
mkdir -p "$TRACE_ROOT"

salloc --nodes 1 --qos interactive --time 01:00:00 --constraint gpu --gpus 4 --account m4999 \
  srun --nodes 1 --ntasks 1 --gpus 4 --cpus-per-task 64 --gpu-bind=none bash -lc '
    set -euo pipefail
    source ~/.bashrc >/dev/null 2>&1 || true
    module load conda >/dev/null 2>&1
    conda activate /pscratch/sd/k/kg597/session1/cvllm
    cd ~/ccl-bench
    export TRACE_ROOT="'"$TRACE_ROOT"'"
    export BUNDLE_ROOT="${PSCRATCH}/ccl-bench-traces/vllm-longseq-kineto-bundles"
    bash scripts/run_vllm_longseq_deepseek_kineto_perlmutter.sh
    bash scripts/make_vllm_longseq_deepseek_bundles_perlmutter.sh
  '
