#!/usr/bin/env bash
set -euo pipefail

SRC_ROOT="${SRC_ROOT:-$PSCRATCH/ccl-bench-traces/vllm-longseq-kineto}"
BUNDLE_ROOT="${BUNDLE_ROOT:-$PSCRATCH/ccl-bench-traces/vllm-longseq-kineto-bundles}"
TRACE_URL_ROOT="${TRACE_URL_ROOT:-/data/ccl-bench_trace_collection}"

INPUT_LEN="${INPUT_LEN:-3968}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
SEQ_LEN="${SEQ_LEN:-4096}"
BATCH="${BATCH:-32}"

mkdir -p "$BUNDLE_ROOT"

link_or_copy() {
  local src=$1 dst=$2
  ln -f "$src" "$dst" 2>/dev/null || cp -f "$src" "$dst"
}

write_yaml() {
  local out=$1 name=$2 comm_name=$3 comm_version=$4
  local comm_env=""
  if [[ "$comm_name" == "MSCCL++" ]]; then
    comm_env="      NCCL_IB_DISABLE: \"1\"
      LD_PRELOAD: /global/homes/k/kg597/mscclpp/build/lib/libmscclpp_nccl.so
      VLLM_NCCL_SO_PATH: /global/homes/k/kg597/mscclpp/build/lib/libmscclpp_nccl.so
      MSCCLPP_NCCL_LIB_PATH: unset
      VLLM_DISABLE_CUSTOM_ALL_REDUCE: \"1\""
  else
    comm_env="      NCCL_IB_DISABLE: \"1\""
  fi
  cat > "$out" <<YAML
version: 1

description: >
  DeepSeek-MoE-16B inference, TP=4, EP=4, batch size ${BATCH}, input length
  ${INPUT_LEN}, communication=${comm_name}. Global batch size is ${BATCH}
  requests. Traces were collected on Perlmutter A100 with vLLM bench latency,
  PyTorch Kineto profiler, and Nsight Systems. Input length is ${INPUT_LEN} and
  output length is ${OUTPUT_LEN}; seq_len records input plus output tokens for
  MFU accounting. DeepSeek-MoE max_position_embeddings is 4096, so this uses
  input3968 + output128 rather than input4096 + output128.

hf_url: https://huggingface.co/deepseek-ai/deepseek-moe-16b-base
trace_url: ${TRACE_URL_ROOT}/${name}
contributor: Kaiwen Guo
contact: kg597@cornell.edu

workload:
  model:
    phase: inference
    moe: true
    granularity: model_fwd
    model_family: deepseek-moe-16b
    precision: bf16
    epochs: 1
    iteration: 1
    model_arch:
      num_params: 16375728128
      num_params_active: 2828650496
      num_params_embedding: 419430400
      num_layers: 28
      num_heads: 16
      head_dim: 128
  data:
    batch_size: ${BATCH}
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
    tp: 4
    pp: 1
    cp: 1
    ep: 4
    pp_mb: 1
  communication_library:
    name: ${comm_name}
    version: "${comm_version}"
    env:
${comm_env}
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
  local comm=$1 comm_name=$2 comm_version=$3
  local name="deepseek-moe-16b-vllm-tp4-ep4-batch${BATCH}-in${INPUT_LEN}-out${OUTPUT_LEN}-${comm}-perlmutter"
  local src="$SRC_ROOT/$name"
  local dst="$BUNDLE_ROOT/$name"
  local logs="$dst/logs"
  if [[ ! -d "$src" ]]; then
    echo "missing source directory: $src" >&2
    return 1
  fi
  rm -rf "$dst"
  mkdir -p "$dst" "$logs"
  write_yaml "$dst/$name.yaml" "$name" "$comm_name" "$comm_version"
  local i=0
  while IFS= read -r trace; do
    link_or_copy "$trace" "$dst/rank${i}_trace.json"
    i=$((i+1))
  done < <(find "$src/kineto" -maxdepth 1 -type f -name "*.pt.trace.json" | sort)
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
# ${name}

- Input length: ${INPUT_LEN}
- Output length: ${OUTPUT_LEN}
- Sequence length recorded in YAML: ${SEQ_LEN}
- Global batch size: ${BATCH}
- Tensor parallelism: 4
- Expert parallelism: 4
- Communication library: ${comm_name}
- Primary metric sources: NSYS SQLite plus PyTorch Kineto JSON
MD
  echo "created $dst with $i rank trace(s)"
}

bundle_one nccl NCCL "2.27.5"
bundle_one puremscclpp "MSCCL++" "0.8.0"
