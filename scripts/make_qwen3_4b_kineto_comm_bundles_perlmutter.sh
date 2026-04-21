#!/usr/bin/env bash
set -euo pipefail

SRC_ROOT="${SRC_ROOT:-$PSCRATCH/ccl-bench-traces/qwen3-4b-kineto-comm-matrix}"
BUNDLE_ROOT="${BUNDLE_ROOT:-$PSCRATCH/ccl-bench-traces/qwen3-4b-kineto-comm-bundles}"
TRACE_URL_ROOT="${TRACE_URL_ROOT:-/data/ccl-bench_trace_collection}"

INPUT_LEN="${INPUT_LEN:-1024}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
SEQ_LEN="${SEQ_LEN:-1024}"
BATCH_SIZES="${BATCH_SIZES:-8 128}"
TP_SIZES="${TP_SIZES:-2 4}"
COMM_VARIANTS="${COMM_VARIANTS:-nccl mscclpp}"
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
  local tp=$3
  local batch=$4
  local comm=$5

  local comm_name="NCCL"
  local comm_env="      PYTORCH_CUDA_ALLOC_CONF: expandable_segments:True"$'\n'
  if [[ "$comm" == "mscclpp" ]]; then
    comm_name="NCCL+MSCCL++"
    comm_env="      LD_PRELOAD: ${MSCCLPP_PRELOAD}"$'\n'
  fi

  cat > "$out" <<YAML
version: 1

description: >
  Qwen3-4B inference, TP=${tp}, batch size ${batch}, communication=${comm_name}.
  Global batch size is ${batch} requests. Traces were collected on Perlmutter with vLLM bench latency, PyTorch Kineto
  profiler, and Nsight Systems. Input length is ${INPUT_LEN} and output length
  is ${OUTPUT_LEN}; seq_len records input tokens for current MFU/prefill accounting.

hf_url: https://huggingface.co/Qwen/Qwen3-4B
trace_url: ${TRACE_URL_ROOT}/${name}
contributor: Kaiwen Guo
contact: kg597@cornell.edu

workload:
  model:
    phase: inference
    moe: false
    granularity: model_fwd
    model_family: qwen3-4b
    precision: bf16
    epochs: 1
    iteration: 1
    model_arch:
      num_params: 4022468096
      num_params_embedding: 388956160
      num_layers: 36
      num_heads: 32
      head_dim: 128
  data:
    batch_size: ${batch}
    batch_size_scope: global
    input_len: ${INPUT_LEN}
    output_len: ${OUTPUT_LEN}
    seq_len: ${SEQ_LEN} # input_len for MFU/prefill FLOP accounting
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
    ep: 1
    pp_mb: 1
  communication_library:
    name: ${comm_name}
    version: "2.27.5"
    env:
      NCCL_IB_DISABLE: "1"
${comm_env}  protocol_selection:
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
  local tp=$1
  local batch=$2
  local comm=$3
  local name="qwen3-4b-vllm-tp${tp}-batch${batch}-${comm}-perlmutter"
  local src="$SRC_ROOT/$name"
  local dst="$BUNDLE_ROOT/$name"
  local logs="$dst/logs"

  if [[ ! -d "$src" ]]; then
    echo "missing source directory: $src" >&2
    return 1
  fi

  rm -rf "$dst"
  mkdir -p "$dst" "$logs"

  write_yaml "$dst/$name.yaml" "$name" "$tp" "$batch" "$comm"

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
# ${name}

Collected on Perlmutter A100 with \`vllm bench latency\`.

- Input length: ${INPUT_LEN}
- Output length: ${OUTPUT_LEN}
- Batch size: ${batch}
- Tensor parallelism: ${tp}
- Communication library: ${comm}
- Primary metric sources: NSYS SQLite plus PyTorch Kineto JSON
- Additional artifacts: NSYS \`.nsys-rep\`, \`.sqlite\`, and raw vLLM logs under \`logs/\`
MD

  echo "created $dst with $i rank trace(s)"
}

for tp in $TP_SIZES; do
  for comm in $COMM_VARIANTS; do
    for batch in $BATCH_SIZES; do
      bundle_one "$tp" "$batch" "$comm"
    done
  done
done

echo
echo "Bundle root: $BUNDLE_ROOT"
find "$BUNDLE_ROOT" -maxdepth 2 -type f | sort
