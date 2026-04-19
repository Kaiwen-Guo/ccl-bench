#!/usr/bin/env bash
set -uo pipefail

# Collect missing NCCL+MSCCL++ vLLM Kineto/NSYS rows for CCL-Bench.
#
# Produces one directory per model x batch size:
#   - vLLM bench latency JSON
#   - PyTorch Kineto trace JSON from vLLM's built-in profiler
#   - NSYS .nsys-rep and .sqlite around the profiled latency run
#
# Default matrix:
#   models: Llama-3.1-8B, DeepSeek-MoE-16B
#   comm: NCCL+MSCCL++ via LD_PRELOAD
#   batch: 8, 128
#   input/output: 1024/128

TRACE_ROOT="${TRACE_ROOT:-$PSCRATCH/ccl-bench-traces/llama-deepseek-kineto-mscclpp}"
HF_HOME="${HF_HOME:-$PSCRATCH/huggingface}"
INPUT_LEN="${INPUT_LEN:-1024}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
BATCH_SIZES="${BATCH_SIZES:-8 128}"
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

model_args() {
  local model_key=$1
  case "$model_key" in
    llama3.1-8b)
      MODEL_ID="${LLAMA_MODEL_ID:-meta-llama/Llama-3.1-8B}"
      MODEL_FAMILY="llama-3.1-8b"
      TP="4"
      EP="1"
      EXTRA_ARGS=(--tensor-parallel-size 4 --trust-remote-code)
      ;;
    deepseek-moe-16b)
      MODEL_ID="${DEEPSEEK_MODEL_ID:-deepseek-ai/deepseek-moe-16b-base}"
      MODEL_FAMILY="deepseek-moe-16b"
      TP="4"
      EP="4"
      EXTRA_ARGS=(--tensor-parallel-size 4 --enable-expert-parallel --trust-remote-code)
      ;;
    *)
      echo "unknown model key: $model_key" >&2
      return 1
      ;;
  esac
}

run_one() {
  local model_key=$1
  local batch_size=$2
  model_args "$model_key" || return 1

  if [[ ! -f "$MSCCLPP_PRELOAD" ]]; then
    echo "missing MSCCL++ preload library: $MSCCLPP_PRELOAD" >&2
    return 1
  fi

  local batched_tokens
  batched_tokens=$(scheduled_batched_tokens "$batch_size")

  local name="${MODEL_FAMILY}-vllm-tp${TP}"
  if [[ "$EP" != "1" ]]; then
    name="${name}-ep${EP}"
  fi
  name="${name}-batch${batch_size}-mscclpp-perlmutter"

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
  echo "  TP=$TP EP=$EP comm=NCCL+MSCCL++ batch=$batch_size input=$INPUT_LEN output=$OUTPUT_LEN"
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
    "${EXTRA_ARGS[@]}"
  )

  local env_cmd=(env -u PYTORCH_CUDA_ALLOC_CONF LD_PRELOAD="$MSCCLPP_PRELOAD")

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
  echo "HF_HOME: $HF_HOME"
  echo "MSCCL++ preload: $MSCCLPP_PRELOAD"
  echo "Batch sizes: $BATCH_SIZES"

  local models
  models=(${MODELS:-llama3.1-8b deepseek-moe-16b})

  for model_key in "${models[@]}"; do
    for batch_size in $BATCH_SIZES; do
      run_one "$model_key" "$batch_size" || echo "FAILED: $model_key batch=$batch_size"
    done
  done

  echo
  echo "==== DONE ===="
  find "$TRACE_ROOT" -maxdepth 2 -type f | sed "s#^#  #"
}

main "$@"
