#!/usr/bin/env bash
set -euo pipefail

SRC_ROOT="${SRC_ROOT:-$PSCRATCH/ccl-bench-traces/llama-deepseek-kineto-mscclpp}"
BUNDLE_ROOT="${BUNDLE_ROOT:-$PSCRATCH/ccl-bench-traces/llama-deepseek-kineto-mscclpp-bundles}"
TRACE_URL_ROOT="${TRACE_URL_ROOT:-/data/ccl-bench_trace_collection}"

INPUT_LEN="${INPUT_LEN:-1024}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
SEQ_LEN="${SEQ_LEN:-1152}"
BATCH_SIZES="${BATCH_SIZES:-8 128}"
MSCCLPP_PRELOAD="${MSCCLPP_PRELOAD:-$HOME/mscclpp/build/lib/libmscclpp_nccl.so}"

mkdir -p "$BUNDLE_ROOT"

link_or_copy() {
  local src=$1
  local dst=$2
  ln -f "$src" "$dst" 2>/dev/null || cp -f "$src" "$dst"
}

write_yaml() {
  local out=$1
  local name=$2
  local desc=$3
  local hf_url=$4
  local moe=$5
  local family=$6
  local num_params=$7
  local num_params_embedding=$8
  local layers=$9
  local heads=${10}
  local head_dim=${11}
  local batch=${12}
  local tp=${13}
  local ep=${14}
  local num_params_active=${15:-}
  local active_params_yaml=""
  if [[ -n "$num_params_active" ]]; then
    active_params_yaml="      num_params_active: ${num_params_active}"$'\n'
  fi

  cat > "$out" <<YAML
version: 1

description: >
  ${desc}
  Global batch size is ${batch} requests. Traces were collected on Perlmutter with vLLM bench latency, PyTorch Kineto
  profiler, and Nsight Systems. Input length is ${INPUT_LEN} and output length
  is ${OUTPUT_LEN}; seq_len records input plus output tokens for MFU accounting. MSCCL++ was enabled through LD_PRELOAD; NCCL remains the
  base communication stack for collectives not intercepted by MSCCL++.

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
${active_params_yaml}
      num_params_embedding: ${num_params_embedding}
      num_layers: ${layers}
      num_heads: ${heads}
      head_dim: ${head_dim}
  data:
    batch_size: ${batch}
    batch_size_scope: global
    input_len: ${INPUT_LEN}
    output_len: ${OUTPUT_LEN}
    seq_len: ${SEQ_LEN} # input_len + output_len
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
      total_count: 4
      count_per_node: 4
    driver_version: cuda_12.8

Model-executor:
  framework:
    name: vllm
    version: "0.19.0"
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
    name: NCCL+MSCCL++
    version: "2.27.5"
    env:
      NCCL_IB_DISABLE: "1"
      LD_PRELOAD: ${MSCCLPP_PRELOAD}
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
  local src_name=$1
  local bundle_name=$2
  local desc=$3
  local hf_url=$4
  local moe=$5
  local family=$6
  local num_params=$7
  local num_params_embedding=$8
  local layers=$9
  local heads=${10}
  local head_dim=${11}
  local batch=${12}
  local tp=${13}
  local ep=${14}
  local num_params_active=${15:-}

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
  done < <(find "$src/kineto" -maxdepth 1 -type f -name "*.pt.trace.json" | sort)

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
- Communication library: NCCL+MSCCL++ via \`LD_PRELOAD=${MSCCLPP_PRELOAD}\`
- Primary metric sources: NSYS SQLite plus PyTorch Kineto JSON
- Additional artifacts: NSYS \`.nsys-rep\`, \`.sqlite\`, and raw vLLM logs under \`logs/\`
MD

  echo "created $dst with $i rank trace(s)"
}

for batch in $BATCH_SIZES; do
  bundle_one \
    "llama-3.1-8b-vllm-tp4-batch${batch}-mscclpp-perlmutter" \
    "llama-3.1-8b-vllm-tp4-batch${batch}-mscclpp-perlmutter" \
    "Llama-3.1-8B inference, TP=4, batch size ${batch}, communication=NCCL+MSCCL++." \
    "https://huggingface.co/meta-llama/Llama-3.1-8B" false llama-3.1-8b \
    8030261248 1050673152 32 32 128 "$batch" 4 1

  bundle_one \
    "deepseek-moe-16b-vllm-tp4-ep4-batch${batch}-mscclpp-perlmutter" \
    "deepseek-moe-16b-vllm-tp4-ep4-batch${batch}-mscclpp-perlmutter" \
    "DeepSeek-MoE-16B inference, TP=4, EP=4, batch size ${batch}, communication=NCCL+MSCCL++." \
    "https://huggingface.co/deepseek-ai/deepseek-moe-16b-base" true deepseek-moe-16b \
    16375728128 419430400 28 16 128 "$batch" 4 4 2828650496
done

echo
echo "Bundle root: $BUNDLE_ROOT"
find "$BUNDLE_ROOT" -maxdepth 2 -type f | sort
