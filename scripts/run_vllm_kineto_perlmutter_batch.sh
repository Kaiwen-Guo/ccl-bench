#!/usr/bin/env bash
set -uo pipefail

# Batch runner for vLLM latency/profiling traces on Perlmutter.
#
# Produces, per model x batch-size:
#   - latency JSON from `vllm bench latency`
#   - Kineto/PyTorch profiler JSON from vLLM's built-in torch profiler
#   - NSYS .nsys-rep + .sqlite wrapping the profiled batch
#
# Intended default matrix:
#   models: Qwen3-4B, Llama-3.1-8B, DeepSeek-MoE-16B
#   batch sizes: 8, 128
#   input/output: 1024/128

TRACE_ROOT="${TRACE_ROOT:-$PSCRATCH/ccl-bench-traces/vllm-kineto-perlmutter}"
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
NUM_PROMPTS="${NUM_PROMPTS:-200}"
REQUEST_RATE="${REQUEST_RATE:-inf}"
PORT="${PORT:-8000}"
READY_TIMEOUT="${READY_TIMEOUT:-600}"
NSYS_DRAIN="${NSYS_DRAIN:-300}"

export HF_HOME
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"

mkdir -p "$TRACE_ROOT"

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "missing required command: $1" >&2
    exit 1
  }
}

maybe_download_models() {
  # Qwen3-4B already exists on Kaiwen's Perlmutter scratch from prior runs.
  # Llama and DeepSeek are loaded by HF id; HF_HOME points to PSCRATCH.
  require_cmd huggingface-cli

  if [[ ! -d "$PSCRATCH/models/Qwen3-4B" ]]; then
    echo "Downloading Qwen/Qwen3-4B into HF cache..."
    huggingface-cli download Qwen/Qwen3-4B >/dev/null || return 1
  fi

  echo "Ensuring Llama-3.1-8B and DeepSeek-MoE-16B are present in HF_HOME=$HF_HOME"
  huggingface-cli download meta-llama/Llama-3.1-8B >/dev/null || return 1
  huggingface-cli download deepseek-ai/deepseek-moe-16b-base >/dev/null || return 1
}

model_args() {
  local model_key=$1
  case "$model_key" in
    qwen3-4b)
      MODEL_ID="$PSCRATCH/models/Qwen3-4B"
      MODEL_FAMILY="qwen3-4b"
      MOE="false"
      TP="4"
      DP="1"
      EP="1"
      RUN_MODE="latency"
      EXTRA_ARGS=(--tensor-parallel-size 4)
      ;;
    llama3.1-8b)
      MODEL_ID="meta-llama/Llama-3.1-8B"
      MODEL_FAMILY="llama-3.1-8b"
      MOE="false"
      TP="4"
      DP="1"
      EP="1"
      RUN_MODE="latency"
      EXTRA_ARGS=(--tensor-parallel-size 4 --trust-remote-code)
      ;;
    deepseek-moe-16b)
      MODEL_ID="deepseek-ai/deepseek-moe-16b-base"
      MODEL_FAMILY="deepseek-moe-16b"
      MOE="true"
      TP="4"
      DP="1"
      EP="4"
      RUN_MODE="latency"
      EXTRA_ARGS=(--tensor-parallel-size 4 --enable-expert-parallel --trust-remote-code)
      ;;
    *)
      echo "unknown model key: $model_key" >&2
      return 1
      ;;
  esac
}

scheduled_batched_tokens() {
  local batch_size=$1
  local full_tokens=$((batch_size * (INPUT_LEN + OUTPUT_LEN)))

  if [[ "$MAX_NUM_BATCHED_TOKENS_CAP" =~ ^[0-9]+$ ]] && (( MAX_NUM_BATCHED_TOKENS_CAP > 0 )) && (( full_tokens > MAX_NUM_BATCHED_TOKENS_CAP )); then
    echo "$MAX_NUM_BATCHED_TOKENS_CAP"
  else
    echo "$full_tokens"
  fi
}

wait_for_server() {
  local port=$1
  local pid=${2:-}
  local t=0
  while (( t < READY_TIMEOUT )); do
    if curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      return 0
    fi
    if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
      return 1
    fi
    sleep 2
    t=$((t+2))
  done
  return 1
}

stop_server() {
  local pid=$1
  if kill -0 "$pid" 2>/dev/null; then
    kill -INT "$pid" || true
  fi

  local t=0
  while kill -0 "$pid" 2>/dev/null && (( t < NSYS_DRAIN )); do
    sleep 1
    t=$((t+1))
  done

  if kill -0 "$pid" 2>/dev/null; then
    echo "  ! server still alive after ${NSYS_DRAIN}s, SIGKILL"
    kill -9 "$pid" 2>/dev/null || true
  fi
}

run_latency_one() {
  local model_key=$1
  local batch_size=$2
  model_args "$model_key" || return 1

  local batched_tokens
  batched_tokens=$(scheduled_batched_tokens "$batch_size")
  local name="${MODEL_FAMILY}-vllm-batch${batch_size}-perlmutter"
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
  echo "  batch: $batch_size input=$INPUT_LEN output=$OUTPUT_LEN"
  echo "  TP=$TP DP=$DP EP=$EP mode=latency max_num_batched_tokens=$batched_tokens"
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

  echo "[1/2] latency run..."
  python -m vllm.entrypoints.cli.main bench latency \
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
    python -m vllm.entrypoints.cli.main bench latency \
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

run_server_one() {
  local model_key=$1
  local batch_size=$2
  model_args "$model_key" || return 1

  local batched_tokens
  batched_tokens=$(scheduled_batched_tokens "$batch_size")
  local name="${MODEL_FAMILY}-vllm-batch${batch_size}-perlmutter"
  local out_dir="$TRACE_ROOT/$name"
  local kineto_dir="$out_dir/kineto"
  local nsys_prefix="$out_dir/${name}"
  local bench_json="$out_dir/${name}.serve.json"
  local client_log="$out_dir/${name}.bench.log"
  local server_log="$out_dir/${name}.server.log"
  local port=$((PORT + batch_size % 1000))

  rm -rf "$out_dir"
  mkdir -p "$kineto_dir"

  echo
  echo "============================================================"
  echo "RUN $name"
  echo "  model: $MODEL_ID"
  echo "  batch/max_concurrency: $batch_size input=$INPUT_LEN output=$OUTPUT_LEN prompts=$NUM_PROMPTS"
  echo "  TP=$TP DP=$DP EP=$EP mode=server max_num_batched_tokens=$batched_tokens"
  echo "  out: $out_dir"
  echo "============================================================"

  local server_args=(
    --model "$MODEL_ID"
    --served-model-name "$MODEL_FAMILY"
    --host 127.0.0.1
    --port "$port"
    --max-model-len "$MAX_MODEL_LEN"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --max-num-seqs "$batch_size"
    --max-num-batched-tokens "$batched_tokens"
    --enforce-eager
    --no-enable-log-requests
    --profiler-config.profiler torch
    --profiler-config.torch_profiler_dir "$kineto_dir"
    --profiler-config.torch_profiler_with_stack false
    --profiler-config.torch_profiler_record_shapes true
    --profiler-config.torch_profiler_with_memory true
    --profiler-config.torch_profiler_with_flops true
    --profiler-config.torch_profiler_use_gzip false
    "${EXTRA_ARGS[@]}"
  )

  nsys profile \
    -t cuda,nvtx,osrt \
    -s none --cpuctxsw=none \
    --trace-fork-before-exec=true \
    --force-overwrite=true \
    --stats=true \
    --export=sqlite \
    -o "$nsys_prefix" \
    python -m vllm.entrypoints.openai.api_server \
      "${server_args[@]}" \
      > "$server_log" 2>&1 &
  local server_pid=$!

  echo "  server pid: $server_pid, waiting up to ${READY_TIMEOUT}s..."
  if ! wait_for_server "$port" "$server_pid"; then
    echo "  ! server never became ready; see $server_log"
    stop_server "$server_pid"
    return 1
  fi
  echo "  server ready"

  python -m vllm.entrypoints.cli.main bench serve \
    --backend vllm \
    --base-url "http://127.0.0.1:${port}" \
    --model "$MODEL_FAMILY" \
    --tokenizer "$MODEL_ID" \
    --trust-remote-code \
    --dataset-name random \
    --random-input-len "$INPUT_LEN" \
    --random-output-len "$OUTPUT_LEN" \
    --num-prompts "$NUM_PROMPTS" \
    --request-rate "$REQUEST_RATE" \
    --max-concurrency "$batch_size" \
    --profile \
    --save-result \
    --result-filename "$bench_json" \
    > "$client_log" 2>&1
  local client_rc=$?

  echo "  client rc=$client_rc, stopping server..."
  stop_server "$server_pid"

  if [[ "$client_rc" != 0 ]]; then
    echo "  ! serve benchmark failed rc=$client_rc; see $client_log"
    return "$client_rc"
  fi

  echo "  produced:"
  find "$out_dir" -maxdepth 2 -type f | sed "s#^#    #"
}

run_one() {
  local model_key=$1
  local batch_size=$2
  model_args "$model_key" || return 1

  if [[ "$RUN_MODE" == "server" ]]; then
    run_server_one "$model_key" "$batch_size"
  else
    run_latency_one "$model_key" "$batch_size"
  fi
}

main() {
  require_cmd python
  require_cmd nsys

  echo "Host: $(hostname)"
  echo "GPU count: $(nvidia-smi -L | wc -l)"
  echo "Trace root: $TRACE_ROOT"
  echo "HF_HOME: $HF_HOME"
  echo "Max scheduled tokens cap: $MAX_NUM_BATCHED_TOKENS_CAP"

  if [[ "${SKIP_DOWNLOAD:-0}" != "1" ]]; then
    maybe_download_models || {
      echo "model download/check failed" >&2
      return 1
    }
  fi

  local models
  models=(${MODELS:-qwen3-4b llama3.1-8b deepseek-moe-16b})

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
