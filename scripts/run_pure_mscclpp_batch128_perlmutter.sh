#!/usr/bin/env bash
set -uo pipefail

# Pure MSCCL++ vLLM Kineto/NSYS runs for batch=128 variants.
#
# Differences vs prior "mscclpp" runs:
#   - --disable-custom-all-reduce so AllReduce goes through torch.distributed
#     and hits the LD_PRELOAD'd MSCCL++ shim (selector picks native
#     default_allreduce_fullmesh on A100 because MSCCLPP_NCCL_LIB_PATH unset
#     -> mscclppNcclDlopenSharedLib=false).
#   - MSCCLPP_NCCL_LIB_PATH is explicitly unset to prevent MSCCL++ from
#     dlopen'ing system NCCL as a fallback.
#   - VLLM_NCCL_SO_PATH points vLLM's pynccl wrapper at the MSCCL++ shim
#     directly. LD_PRELOAD alone is not enough because pynccl uses ctypes.CDLL.
#
# Variants (4 total):
#   qwen3-4b    tp2    batch128
#   qwen3-4b    tp4    batch128
#   llama-3.1-8b tp4   batch128
#   deepseek-moe-16b tp4 ep4 batch128

TRACE_ROOT="${TRACE_ROOT:-$PSCRATCH/ccl-bench-traces/pure-mscclpp-batch128}"
HF_HOME="${HF_HOME:-$PSCRATCH/huggingface}"
QWEN_MODEL_ID="${QWEN_MODEL_ID:-$PSCRATCH/models/Qwen3-4B}"
LLAMA_MODEL_ID="${LLAMA_MODEL_ID:-meta-llama/Llama-3.1-8B}"
DEEPSEEK_MODEL_ID="${DEEPSEEK_MODEL_ID:-deepseek-ai/deepseek-moe-16b-base}"

INPUT_LEN="${INPUT_LEN:-1024}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
BATCH_SIZE="${BATCH_SIZE:-128}"
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
  local full_tokens=$((BATCH_SIZE * (INPUT_LEN + OUTPUT_LEN)))
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
  shift 3
  local extra_args=("$@")

  if [[ ! -f "$MSCCLPP_PRELOAD" ]]; then
    echo "missing MSCCL++ preload library: $MSCCLPP_PRELOAD" >&2
    return 1
  fi

  local batched_tokens
  batched_tokens=$(scheduled_batched_tokens)

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
  echo "RUN $name  (pure MSCCL++ via LD_PRELOAD, custom_all_reduce disabled)"
  echo "  model: $model_id  tp=$tp  batch=$BATCH_SIZE"
  echo "  max_num_batched_tokens=$batched_tokens"
  echo "  out: $out_dir"
  echo "============================================================"

  local common_args=(
    --model "$model_id"
    --input-len "$INPUT_LEN"
    --output-len "$OUTPUT_LEN"
    --batch-size "$BATCH_SIZE"
    --max-model-len "$MAX_MODEL_LEN"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --max-num-seqs "$BATCH_SIZE"
    --max-num-batched-tokens "$batched_tokens"
    --enforce-eager
    --disable-detokenize
    --disable-custom-all-reduce
    --tensor-parallel-size "$tp"
    "${extra_args[@]}"
  )

  # -u PYTORCH_CUDA_ALLOC_CONF: MSCCL++ rejects expandable-segments on A100.
  # -u MSCCLPP_NCCL_LIB_PATH: prevent shim from dlopen'ing libnccl as fallback,
  #   forcing native MSCCL++ fullmesh path on A100 without NVLS.
  local env_cmd=(env -u PYTORCH_CUDA_ALLOC_CONF -u MSCCLPP_NCCL_LIB_PATH VLLM_NCCL_SO_PATH="$MSCCLPP_PRELOAD" LD_PRELOAD="$MSCCLPP_PRELOAD")

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
  echo "MSCCL++ preload: $MSCCLPP_PRELOAD"

  if [[ ! -d "$QWEN_MODEL_ID" ]]; then
    echo "WARN: Qwen model dir not found: $QWEN_MODEL_ID (will try HF download)" >&2
  fi

  run_one "qwen3-4b-vllm-tp2-batch${BATCH_SIZE}-puremscclpp-perlmutter" \
    "$QWEN_MODEL_ID" 2 || echo "FAILED: qwen tp2"

  run_one "qwen3-4b-vllm-tp4-batch${BATCH_SIZE}-puremscclpp-perlmutter" \
    "$QWEN_MODEL_ID" 4 || echo "FAILED: qwen tp4"

  run_one "llama-3.1-8b-vllm-tp4-batch${BATCH_SIZE}-puremscclpp-perlmutter" \
    "$LLAMA_MODEL_ID" 4 --trust-remote-code || echo "FAILED: llama tp4"

  run_one "deepseek-moe-16b-vllm-tp4-ep4-batch${BATCH_SIZE}-puremscclpp-perlmutter" \
    "$DEEPSEEK_MODEL_ID" 4 --enable-expert-parallel --trust-remote-code \
    || echo "FAILED: deepseek tp4-ep4"

  echo
  echo "==== DONE ===="
  find "$TRACE_ROOT" -maxdepth 2 -type f | sed "s#^#  #"
}

main "$@"
