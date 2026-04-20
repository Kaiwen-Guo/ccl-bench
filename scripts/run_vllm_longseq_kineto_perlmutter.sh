#!/usr/bin/env bash
set -uo pipefail

# Larger-sequence vLLM Kineto/NSYS runs for CCL-Bench.
#
# Chosen subset:
#   - Qwen3-4B TP4, batch64, input4096, output128: NCCL and pure MSCCL++
#   - Llama-3.1-8B TP4, batch64, input4096, output128: NCCL and pure MSCCL++
#   - DeepSeek-MoE-16B TP4/EP4, batch32, input4096, output128: NCCL and pure MSCCL++
#
# Pure MSCCL++ uses VLLM_NCCL_SO_PATH plus LD_PRELOAD and disables vLLM's
# custom all-reduce so collective kernels hit MSCCL++ directly.

TRACE_ROOT="${TRACE_ROOT:-$PSCRATCH/ccl-bench-traces/vllm-longseq-kineto}"
HF_HOME="${HF_HOME:-$PSCRATCH/huggingface}"
QWEN_MODEL_ID="${QWEN_MODEL_ID:-$PSCRATCH/models/Qwen3-4B}"
LLAMA_MODEL_ID="${LLAMA_MODEL_ID:-meta-llama/Llama-3.1-8B}"
DEEPSEEK_MODEL_ID="${DEEPSEEK_MODEL_ID:-deepseek-ai/deepseek-moe-16b-base}"

INPUT_LEN="${INPUT_LEN:-4096}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4224}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
MAX_NUM_BATCHED_TOKENS_CAP="${MAX_NUM_BATCHED_TOKENS_CAP:-32768}"
NUM_ITERS_WARMUP="${NUM_ITERS_WARMUP:-2}"
NUM_ITERS="${NUM_ITERS:-3}"
PROFILE_WARMUP="${PROFILE_WARMUP:-1}"
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
  local name=$1
  local model_id=$2
  local tp=$3
  local batch_size=$4
  local comm=$5
  shift 5
  local extra_args=("$@")

  if [[ "$model_id" == /* && ! -d "$model_id" ]]; then
    echo "missing model directory: $model_id" >&2
    return 1
  fi

  if [[ "$comm" == "puremscclpp" && ! -f "$MSCCLPP_PRELOAD" ]]; then
    echo "missing MSCCL++ preload library: $MSCCLPP_PRELOAD" >&2
    return 1
  fi

  local batched_tokens
  batched_tokens=$(scheduled_batched_tokens "$batch_size")

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
  echo "  model: $model_id"
  echo "  tp=$tp comm=$comm batch=$batch_size input=$INPUT_LEN output=$OUTPUT_LEN"
  echo "  max_model_len=$MAX_MODEL_LEN max_num_batched_tokens=$batched_tokens"
  echo "  out: $out_dir"
  echo "============================================================"

  local common_args=(
    --model "$model_id"
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
    "${extra_args[@]}"
  )

  local env_cmd=(env)
  if [[ "$comm" == "puremscclpp" ]]; then
    common_args+=(--disable-custom-all-reduce)
    env_cmd=(env -u PYTORCH_CUDA_ALLOC_CONF -u MSCCLPP_NCCL_LIB_PATH \
      VLLM_NCCL_SO_PATH="$MSCCLPP_PRELOAD" LD_PRELOAD="$MSCCLPP_PRELOAD")
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
  echo "Input/output: $INPUT_LEN/$OUTPUT_LEN"
  echo "MSCCL++ preload: $MSCCLPP_PRELOAD"

  run_one "qwen3-4b-vllm-tp4-batch64-in4096-out128-nccl-perlmutter" \
    "$QWEN_MODEL_ID" 4 64 nccl || echo "FAILED: qwen nccl"
  run_one "qwen3-4b-vllm-tp4-batch64-in4096-out128-puremscclpp-perlmutter" \
    "$QWEN_MODEL_ID" 4 64 puremscclpp || echo "FAILED: qwen puremscclpp"

  run_one "llama-3.1-8b-vllm-tp4-batch64-in4096-out128-nccl-perlmutter" \
    "$LLAMA_MODEL_ID" 4 64 nccl --trust-remote-code || echo "FAILED: llama nccl"
  run_one "llama-3.1-8b-vllm-tp4-batch64-in4096-out128-puremscclpp-perlmutter" \
    "$LLAMA_MODEL_ID" 4 64 puremscclpp --trust-remote-code || echo "FAILED: llama puremscclpp"

  run_one "deepseek-moe-16b-vllm-tp4-ep4-batch32-in4096-out128-nccl-perlmutter" \
    "$DEEPSEEK_MODEL_ID" 4 32 nccl --enable-expert-parallel --trust-remote-code \
    || echo "FAILED: deepseek nccl"
  run_one "deepseek-moe-16b-vllm-tp4-ep4-batch32-in4096-out128-puremscclpp-perlmutter" \
    "$DEEPSEEK_MODEL_ID" 4 32 puremscclpp --enable-expert-parallel --trust-remote-code \
    || echo "FAILED: deepseek puremscclpp"

  echo
  echo "==== DONE ===="
  find "$TRACE_ROOT" -maxdepth 2 -type f | sed "s#^#  #"
}

main "$@"
