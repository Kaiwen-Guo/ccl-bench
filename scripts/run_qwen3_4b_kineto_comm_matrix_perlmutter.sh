#!/usr/bin/env bash
set -uo pipefail

# Qwen3-4B Perlmutter matrix for CCL-Bench website rows.
#
# Produces one directory per TP x communication library x batch size:
#   - vLLM bench latency JSON
#   - PyTorch Kineto trace JSON from vLLM's built-in profiler
#   - NSYS .nsys-rep and .sqlite around the profiled latency run
#
# Default matrix:
#   TP: 2, 4
#   comm: nccl, mscclpp
#   batch: 8, 128
#   input/output: 1024/128

TRACE_ROOT="${TRACE_ROOT:-$PSCRATCH/ccl-bench-traces/qwen3-4b-kineto-comm-matrix}"
HF_HOME="${HF_HOME:-$PSCRATCH/huggingface}"
MODEL_ID="${MODEL_ID:-$PSCRATCH/models/Qwen3-4B}"
INPUT_LEN="${INPUT_LEN:-1024}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
BATCH_SIZES="${BATCH_SIZES:-8 128}"
TP_SIZES="${TP_SIZES:-2 4}"
COMM_VARIANTS="${COMM_VARIANTS:-nccl mscclpp}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1152}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
MAX_NUM_BATCHED_TOKENS_CAP="${MAX_NUM_BATCHED_TOKENS_CAP:-16384}"
NUM_ITERS_WARMUP="${NUM_ITERS_WARMUP:-3}"
NUM_ITERS="${NUM_ITERS:-5}"
PROFILE_WARMUP="${PROFILE_WARMUP:-2}"
MSCCLPP_PRELOAD="${MSCCLPP_PRELOAD:-$HOME/mscclpp/build/lib/libmscclpp_nccl.so}"

export HF_HOME
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$TRACE_ROOT"

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "missing required command: $1" >&2
    exit 1
  }
}

scheduled_batched_tokens() {
  local batch_size=$1
  local full_tokens=$((batch_size * (INPUT_LEN + OUTPUT_LEN)))

  if [[ "$MAX_NUM_BATCHED_TOKENS_CAP" =~ ^[0-9]+$ ]] && \
     (( MAX_NUM_BATCHED_TOKENS_CAP > 0 )) && \
     (( full_tokens > MAX_NUM_BATCHED_TOKENS_CAP )); then
    echo "$MAX_NUM_BATCHED_TOKENS_CAP"
  else
    echo "$full_tokens"
  fi
}

run_one() {
  local tp=$1
  local comm=$2
  local batch_size=$3

  if [[ ! -d "$MODEL_ID" ]]; then
    echo "missing model directory: $MODEL_ID" >&2
    return 1
  fi

  if [[ "$comm" == "mscclpp" && ! -f "$MSCCLPP_PRELOAD" ]]; then
    echo "missing MSCCL++ preload library: $MSCCLPP_PRELOAD" >&2
    return 1
  fi

  local batched_tokens
  batched_tokens=$(scheduled_batched_tokens "$batch_size")

  local name="qwen3-4b-vllm-tp${tp}-batch${batch_size}-${comm}-perlmutter"
  local out_dir="$TRACE_ROOT/$name"
  local kineto_dir="$out_dir/kineto"
  local nsys_prefix="$out_dir/${name}"
  local latency_json="$out_dir/${name}.latency.json"
  local bench_log="$out_dir/${name}.bench.log"
  local profile_log="$out_dir/${name}.profile.log"

  rm -rf "$out_dir"
  mkdir -p "$kineto_dir"

  echo
  echo "============================================================"
  echo "RUN $name"
  echo "  model: $MODEL_ID"
  echo "  tp=$tp comm=$comm batch=$batch_size input=$INPUT_LEN output=$OUTPUT_LEN"
  echo "  max_num_batched_tokens=$batched_tokens"
  echo "  out: $out_dir"
  echo "============================================================"

  local common_args=(
    --model "$MODEL_ID"
    --input-len "$INPUT_LEN"
    --output-len "$OUTPUT_LEN"
    --batch-size "$batch_size"
    --max-model-len "$MAX_MODEL_LEN"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --max-num-seqs "$batch_size"
    --max-num-batched-tokens "$batched_tokens"
    --enforce-eager
    --disable-detokenize
    --tensor-parallel-size "$tp"
  )

  local env_cmd=(env)
  if [[ "$comm" == "mscclpp" ]]; then
    # MSCCL++ rejects PyTorch expandable-segment allocations on A100 because
    # they use cuMemMap without NVLS support.
    env_cmd=(env -u PYTORCH_CUDA_ALLOC_CONF LD_PRELOAD="$MSCCLPP_PRELOAD")
  fi

  echo "[1/2] latency run..."
  "${env_cmd[@]}" python -m vllm.entrypoints.cli.main bench latency \
    "${common_args[@]}" \
    --num-iters-warmup "$NUM_ITERS_WARMUP" \
    --num-iters "$NUM_ITERS" \
    --output-json "$latency_json" \
    > "$bench_log" 2>&1
  local latency_rc=$?
  if [[ "$latency_rc" != 0 ]]; then
    echo "  ! latency run failed rc=$latency_rc; see $bench_log"
    return "$latency_rc"
  fi

  echo "[2/2] kineto + nsys profile run..."
  nsys profile \
    -t cuda,nvtx,osrt \
    -s none --cpuctxsw=none \
    --trace-fork-before-exec=true \
    --force-overwrite=true \
    --stats=true \
    --export=sqlite \
    -o "$nsys_prefix" \
    "${env_cmd[@]}" python -m vllm.entrypoints.cli.main bench latency \
      "${common_args[@]}" \
      --num-iters-warmup "$PROFILE_WARMUP" \
      --num-iters 1 \
      --profile \
      --profiler-config.profiler torch \
      --profiler-config.torch_profiler_dir "$kineto_dir" \
      --profiler-config.torch_profiler_with_stack false \
      --profiler-config.torch_profiler_record_shapes true \
      --profiler-config.torch_profiler_with_memory true \
      --profiler-config.torch_profiler_with_flops true \
      --profiler-config.torch_profiler_use_gzip false \
      > "$profile_log" 2>&1
  local profile_rc=$?
  if [[ "$profile_rc" != 0 ]]; then
    echo "  ! profile run failed rc=$profile_rc; see $profile_log"
    return "$profile_rc"
  fi

  echo "  produced:"
  find "$out_dir" -maxdepth 2 -type f | sed "s#^#    #"
}

main() {
  require_cmd python
  require_cmd nsys
  require_cmd nvidia-smi

  echo "Host: $(hostname)"
  echo "GPU count: $(nvidia-smi -L | wc -l)"
  echo "Trace root: $TRACE_ROOT"
  echo "Model: $MODEL_ID"
  echo "MSCCL++ preload: $MSCCLPP_PRELOAD"
  echo "Batch sizes: $BATCH_SIZES"
  echo "TP sizes: $TP_SIZES"
  echo "Comm variants: $COMM_VARIANTS"

  for tp in $TP_SIZES; do
    for comm in $COMM_VARIANTS; do
      for batch_size in $BATCH_SIZES; do
        run_one "$tp" "$comm" "$batch_size" || echo "FAILED: tp=$tp comm=$comm batch=$batch_size"
      done
    done
  done

  echo
  echo "==== DONE ===="
  find "$TRACE_ROOT" -maxdepth 2 -type f | sed "s#^#  #"
}

main "$@"
