#!/usr/bin/env bash
set -euo pipefail

SRC_ROOT="${SRC_ROOT:-$PSCRATCH/ccl-bench-traces/pure-mscclpp-batch128}"
BUNDLE_ROOT="${BUNDLE_ROOT:-$PSCRATCH/ccl-bench-traces/pure-mscclpp-batch128-bundles}"
TRACE_URL_ROOT="${TRACE_URL_ROOT:-/data/ccl-bench_trace_collection}"

INPUT_LEN="${INPUT_LEN:-1024}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
SEQ_LEN="${SEQ_LEN:-1152}"
BATCH="${BATCH:-128}"
MSCCLPP_PRELOAD="${MSCCLPP_PRELOAD:-$HOME/mscclpp/build/lib/libmscclpp_nccl.so}"

mkdir -p "$BUNDLE_ROOT"

link_or_copy() {
  local src=$1 dst=$2
  ln -f "$src" "$dst" 2>/dev/null || cp -f "$src" "$dst"
}

write_yaml() {
  local out=$1 name=$2 desc=$3 hf_url=$4 moe=$5 family=$6
  local num_params=$7 num_params_embedding=$8 layers=$9 heads=${10} head_dim=${11}
  local batch=${12} tp=${13} ep=${14} num_params_active=${15:-}
  local active_yaml=""
  if [[ -n "$num_params_active" ]]; then
    active_yaml="      num_params_active: ${num_params_active}"$'\n'
  fi
  cat > "$out" <<YAML
version: 1

description: >
  ${desc}
  Traces were collected on Perlmutter A100 with vLLM bench latency, PyTorch
  Kineto profiler, and Nsight Systems. Input length is ${INPUT_LEN} and output
  length is ${OUTPUT_LEN}. Pure MSCCL++ via LD_PRELOAD and
  VLLM_NCCL_SO_PATH with --disable-custom-all-reduce and
  MSCCLPP_NCCL_LIB_PATH unset, so AllReduce and AllGather both run through
  MSCCL++ native kernels (fullmesh / packet) rather than vLLM's
  custom_all_reduce or the NCCL fallback.

hf_url: ${hf_url}
trace_url: ${TRACE_URL_ROOT}/${name}
contributor: Kaiwen Guo
contact: kg597@cornell.edu

workload:
  model:
    phase: inference
    moe: ${moe}
    granularity: model_fwd
    model_family: ${family}
    precision: bf16
    epochs: 1
    iteration: 1
    model_arch:
      num_params: ${num_params}
${active_yaml}      num_params_embedding: ${num_params_embedding}
      num_layers: ${layers}
      num_heads: ${heads}
      head_dim: ${head_dim}
  data:
    batch_size: ${batch}
    seq_len: ${SEQ_LEN}
    dataset: random_${INPUT_LEN}_input_${OUTPUT_LEN}_output
  hardware:
    network_topo:
      topology: slingshot
      bandwidth_gbps:
        - 200
        - 2400
    xpu_spec:
      type: GPU
      model: nvidia_a100
      total_count: ${tp}
      count_per_node: ${tp}
    driver_version: cuda_12.8

Model-executor:
  framework:
    name: vllm-0.19.0
    compiler_tool_selection: plain_pytorch
  model_plan_parallelization:
    dp_replicate: 1
    dp_shard: 1
    tp: ${tp}
    pp: 1
    cp: 1
    ep: ${ep}
    pp_mb: 1
  communication_library:
    name: MSCCL++
    version: "0.8.0"
    env:
      NCCL_IB_DISABLE: "1"
      LD_PRELOAD: ${MSCCLPP_PRELOAD}
      VLLM_NCCL_SO_PATH: ${MSCCLPP_PRELOAD}
      MSCCLPP_NCCL_LIB_PATH: unset
      VLLM_DISABLE_CUSTOM_ALL_REDUCE: "1"
  protocol_selection:
    - slingshot
    - nvlink

metric_source:
  traces:
    - nsys
    - json
  metrics_specific_trace:
    - nsys
    - vllm_bench_latency
YAML
}

bundle_one() {
  local src_name=$1 bundle_name=$2 desc=$3 hf_url=$4 moe=$5 family=$6
  local num_params=$7 num_params_embedding=$8 layers=$9 heads=${10} head_dim=${11}
  local batch=${12} tp=${13} ep=${14} num_params_active=${15:-}

  local src="$SRC_ROOT/$src_name"
  local dst="$BUNDLE_ROOT/$bundle_name"
  local logs="$dst/logs"

  if [[ ! -d "$src" ]]; then
    echo "missing source directory: $src" >&2
    return 1
  fi

  rm -rf "$dst"
  mkdir -p "$dst" "$logs"

  write_yaml "$dst/$bundle_name.yaml" "$bundle_name" "$desc" "$hf_url" "$moe" "$family" \
    "$num_params" "$num_params_embedding" "$layers" "$heads" "$head_dim" "$batch" "$tp" "$ep" "$num_params_active"

  local i=0
  while IFS= read -r trace; do
    link_or_copy "$trace" "$dst/rank${i}_trace.json"
    i=$((i+1))
  done < <(find "$src/kineto" -maxdepth 1 -type f -name "*.pt.trace.json" 2>/dev/null | sort)

  if (( i == 0 )); then
    echo "no Kineto trace JSON files found in $src/kineto" >&2
    return 1
  fi

  for ext in sqlite nsys-rep latency.json bench.log profile.log; do
    while IFS= read -r f; do
      [[ -e "$f" ]] || continue
      case "$ext" in
        sqlite|nsys-rep) link_or_copy "$f" "$dst/$(basename "$f")" ;;
        *) link_or_copy "$f" "$logs/$(basename "$f")" ;;
      esac
    done < <(find "$src" -maxdepth 1 -type f -name "*.${ext}" 2>/dev/null | sort)
  done

  cat > "$dst/README.md" <<MD
# ${bundle_name}

Collected on Perlmutter A100 with \`vllm bench latency\`.

- Input length: ${INPUT_LEN}
- Output length: ${OUTPUT_LEN}
- Batch size: ${batch}
- Tensor parallelism: ${tp}
- Expert parallelism: ${ep}
- Communication library: **Pure MSCCL++** via \`LD_PRELOAD=${MSCCLPP_PRELOAD}\`
  and \`VLLM_NCCL_SO_PATH=${MSCCLPP_PRELOAD}\`, with
  \`--disable-custom-all-reduce\` and \`MSCCLPP_NCCL_LIB_PATH\` unset.
- Primary metric sources: NSYS SQLite plus PyTorch Kineto JSON
MD

  echo "created $dst with $i rank trace(s)"
}

bundle_one "qwen3-4b-vllm-tp2-batch${BATCH}-puremscclpp-perlmutter" \
  "qwen3-4b-vllm-tp2-batch${BATCH}-puremscclpp-perlmutter" \
  "Qwen3-4B inference, TP=2, batch size ${BATCH}, pure MSCCL++." \
  "https://huggingface.co/Qwen/Qwen3-4B" false qwen3-4b \
  4022468096 388956160 36 32 128 "$BATCH" 2 1

bundle_one "qwen3-4b-vllm-tp4-batch${BATCH}-puremscclpp-perlmutter" \
  "qwen3-4b-vllm-tp4-batch${BATCH}-puremscclpp-perlmutter" \
  "Qwen3-4B inference, TP=4, batch size ${BATCH}, pure MSCCL++." \
  "https://huggingface.co/Qwen/Qwen3-4B" false qwen3-4b \
  4022468096 388956160 36 32 128 "$BATCH" 4 1

bundle_one "llama-3.1-8b-vllm-tp4-batch${BATCH}-puremscclpp-perlmutter" \
  "llama-3.1-8b-vllm-tp4-batch${BATCH}-puremscclpp-perlmutter" \
  "Llama-3.1-8B inference, TP=4, batch size ${BATCH}, pure MSCCL++." \
  "https://huggingface.co/meta-llama/Llama-3.1-8B" false llama-3.1-8b \
  8030261248 1050673152 32 32 128 "$BATCH" 4 1

bundle_one "deepseek-moe-16b-vllm-tp4-ep4-batch${BATCH}-puremscclpp-perlmutter" \
  "deepseek-moe-16b-vllm-tp4-ep4-batch${BATCH}-puremscclpp-perlmutter" \
  "DeepSeek-MoE-16B inference, TP=4, EP=4, batch size ${BATCH}, pure MSCCL++." \
  "https://huggingface.co/deepseek-ai/deepseek-moe-16b-base" true deepseek-moe-16b \
  16375728128 419430400 28 16 128 "$BATCH" 4 4 2828650496

echo
echo "Bundle root: $BUNDLE_ROOT"
find "$BUNDLE_ROOT" -maxdepth 2 -type f | sort
