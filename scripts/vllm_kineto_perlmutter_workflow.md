# Perlmutter vLLM Kineto Website Workflow

Date: 2026-04-18/2026-04-19

This note records the end-to-end workflow used to collect, validate, transfer,
and publish the Perlmutter vLLM Kineto benchmark rows for CCL-Bench PR #35.
It also records the earlier NSYS-only Qwen3-4B/MSCCL++ work, the metric fixes
added for Kineto vLLM traces, current caveats, and the recommended next steps.

## Goal

The immediate goal was to populate the CCL-Bench website with defensible vLLM
inference rows from Perlmutter A100 GPUs.

The original Qwen3-4B runs answered whether MSCCL++ was actually being loaded
and gave NCCL vs NCCL+MSCCL++ NSYS traces. Those rows were useful for kernel and
communication evidence, but they were NSYS-only and therefore did not populate
Step Time or MFU in the website.

The second collection pass generated PyTorch Kineto JSON traces for vLLM
latency runs so the website could populate JSON-based metrics such as
`avg_step_time` and `mfu`.

## Machines and Repositories

Local laptop:

- Repo: `/Users/kaiwen/Project/ccl-bench/ccl-bench`
- Branch: `kevin/memory-bw-tpu-fix`
- PR: <https://github.com/cornell-sysphotonics/ccl-bench/pull/35>

Perlmutter:

- Login alias: `perlmutter`
- Repo: `~/ccl-bench`
- Conda environment: `/pscratch/sd/k/kg597/session1/cvllm`
- Raw vLLM Kineto output:
  `/pscratch/sd/k/kg597/ccl-bench-traces/vllm-kineto-perlmutter`
- Bundled trace output:
  `/pscratch/sd/k/kg597/ccl-bench-traces/vllm-kineto-bundles`

Storage/website host:

- SSH host: `lambda7.cs.cornell.edu`
- Actual hostname observed: `singh-compute-07.cs.cornell.edu`
- Repo used for preview: `~/ccl-bench-main`
- Trace storage used by website generator:
  `/data/ccl-bench_trace_collection`
- Preview server:
  `python3 -m http.server 8081 --bind 127.0.0.1`
- Local tunnel:
  `ssh -fN -L 8081:localhost:8081 lambda7.cs.cornell.edu`
- Browser URL:
  `http://localhost:8081/`

## First Pass: Qwen3-4B NSYS NCCL vs MSCCL++

The first pass collected four Qwen3-4B vLLM serving traces with NSYS:

- `qwen3-4b-vllm-tp2-perlmutter[nccl]`
- `qwen3-4b-vllm-tp2-perlmutter[mscclpp]`
- `qwen3-4b-vllm-tp4-perlmutter[nccl]`
- `qwen3-4b-vllm-tp4-perlmutter[mscclpp]`

Parameters:

- Model: Qwen3-4B
- Input length: 1024
- Output length: 128
- Prompts: 200
- Request rate: 8 RPS
- Precision: BF16
- Hardware: 1 Perlmutter node, 4 x NVIDIA A100 40 GB
- TP variants: 2 and 4
- Communication variants: NCCL and NCCL+MSCCL++

The original script was:

- `scripts/run_qwen3_4b_perlmutter_batch.sh`

Important finding:

- `LD_PRELOAD=$HOME/mscclpp/build/lib/libmscclpp_nccl.so` was active in the
  MSCCL++ runs.
- NSYS kernel summaries showed `mscclpp::collective::allgatherFullmesh2` in
  the MSCCL++ runs.
- AllReduce remained NCCL, for example
  `ncclDevKernel_AllReduce_Sum_bf16_RING_LL`.
- Therefore this setup intercepts AllGather but not AllReduce.

Client results:

| Variant | Throughput tok/s | TTFT mean/median/P99 ms | TPOT mean/median/P99 ms |
|---|---:|---:|---:|
| tp2-nccl | 7887 | 1650 / 178 / 7258 | 38.5 / 34.1 / 102 |
| tp2-mscclpp | 7857 | 116 / 108 / 201 | 32.9 / 32.9 / 33.8 |
| tp4-nccl | 7864 | 107 / 105 / 167 | 34.0 / 33.5 / 36.0 |
| tp4-mscclpp | 7810 | 105 / 105 / 176 | 33.9 / 33.8 / 34.6 |

The high mean TTFT for `tp2-nccl` was due to cold-cache tail behavior. The
median and later runs are the relevant comparison.

Why those four rows still lack Step Time/MFU:

- Their workload cards use `metric_source.traces: [nsys]`.
- The website registry only assigns `avg_step_time` and `mfu` to JSON trace
  types (`json_tpu`, `json`), not NSYS.
- This is consistent with existing vLLM NSYS rows in the repository.

Clarification from Eric's profiling note:

- vLLM can collect torch-profiler JSON traces. In vLLM 0.19.0 the latency
  benchmark path calls `llm.start_profile()` / `llm.stop_profile()` when
  `--profile` is used with `--profiler-config.profiler=torch` and an absolute
  `--profiler-config.torch_profiler_dir=...`.
- That output is the PyTorch profiler / Kineto JSON artifact. In other words,
  "torch JSON trace" and "Kineto JSON trace" refer to the same class of trace
  for this workflow.
- The old four NSYS-only Qwen rows cannot be retrofitted with torch JSON after
  the fact. The same workload has to be rerun with torch profiling enabled.
- NSYS SQLite is still needed for the current kernel-level website metrics
  because it reliably exposes CUDA kernel names such as NCCL kernels and
  MSCCL++ `allreducePacket` / `allreduceFullmesh`.

Workload-card cleanup after Eric's template note:

- `batch_size` is treated as global batch size. The generated GPU vLLM cards
  now include `batch_size_scope: global` under `workload.data`.
- The actual run used `input_len=1024` and `output_len=128`. The cards now
  include those fields explicitly.
- `seq_len` was corrected to the input length (`1024`) for the inference cards
  after comparing against the current workload-card template and TPU vLLM rows,
  where sequence length is documented as fixed input sequence length. Output
  length remains recorded separately as `output_len`.
- The framework block now uses `name: vllm` plus `version: "0.19.0"` to better
  match the workload-card template style while preserving the exact vLLM
  version.
- For future TPU-vs-GPU comparability, keep `seq_len` as input length for
  vLLM inference cards unless the workload-card template changes explicitly.

## Qwen3-4B Kineto NCCL vs MSCCL++ Matrix

The four original Qwen3-4B rows could not be retrofitted with Step Time or MFU
because they only had NSYS traces and vLLM bench serve logs. The NSYS traces
did not contain vLLM engine-iteration NVTX ranges such as `execute_context_*`
or `ProfilerStep#N`, so any Step Time/MFU extraction from those rows would have
been a new heuristic rather than the same metric as the Kineto rows.

To get directly comparable Qwen data, the Qwen3-4B workload was recollected with
both Kineto JSON and NSYS for the full communication matrix:

- TP: 2 and 4
- Batch size: 8 and 128
- Communication library: NCCL and NCCL+MSCCL++
- Input length: 1024
- Output length: 128
- Sequence length recorded in YAML: 1024
- Precision: BF16
- Hardware: 1 Perlmutter node, 4 x NVIDIA A100 40 GB

Scripts:

- `scripts/run_qwen3_4b_kineto_comm_matrix_perlmutter.sh`
- `scripts/make_qwen3_4b_kineto_comm_bundles_perlmutter.sh`
- `scripts/launch_qwen3_4b_kineto_comm_matrix_perlmutter.sh`
- `scripts/sbatch_qwen3_4b_kineto_comm_matrix_perlmutter.sh`

The `sbatch` wrapper could not be used directly on Perlmutter's
`gpu_interactive_ss11` partition:

```text
Cannot submit batch jobs to gpu_interactive_ss11
```

The working launcher therefore uses `salloc` plus `srun`:

```bash
salloc --nodes 1 --qos interactive --time 00:30:00 \
  --constraint gpu --gpus 4 --account m4999 \
  srun --nodes 1 --ntasks 1 --gpus 4 --gpu-bind=none \
  bash scripts/run_qwen3_4b_kineto_comm_matrix_perlmutter.sh
```

Important MSCCL++ runtime fix:

- The NCCL runs use `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- MSCCL++ initially failed with:

```text
RuntimeError: cuMemMap is used in env without NVLS support
```

- The MSCCL++ runs were fixed by unsetting `PYTORCH_CUDA_ALLOC_CONF` only for
  the MSCCL++ server process while keeping:

```bash
LD_PRELOAD=/global/homes/k/kg597/mscclpp/build/lib/libmscclpp_nccl.so
```

This is recorded in the YAML cards. The NCCL rows include
`PYTORCH_CUDA_ALLOC_CONF`; the MSCCL++ rows include `LD_PRELOAD` and omit the
expandable-segments allocator setting.

The eight completed bundles are:

| Bundle | Size on `/data` |
|---|---:|
| `qwen3-4b-vllm-tp2-batch8-nccl-perlmutter` | 736M |
| `qwen3-4b-vllm-tp2-batch128-nccl-perlmutter` | 763M |
| `qwen3-4b-vllm-tp2-batch8-mscclpp-perlmutter` | 630M |
| `qwen3-4b-vllm-tp2-batch128-mscclpp-perlmutter` | 666M |
| `qwen3-4b-vllm-tp4-batch8-nccl-perlmutter` | 1.4G |
| `qwen3-4b-vllm-tp4-batch128-nccl-perlmutter` | 1.5G |
| `qwen3-4b-vllm-tp4-batch8-mscclpp-perlmutter` | 1.3G |
| `qwen3-4b-vllm-tp4-batch128-mscclpp-perlmutter` | 1.4G |

They were streamed from Perlmutter to the website storage host:

```bash
ssh perlmutter \
  'cd /pscratch/sd/k/kg597/ccl-bench-traces/qwen3-4b-kineto-comm-bundles && tar -czf - \
    qwen3-4b-vllm-tp2-batch8-nccl-perlmutter \
    qwen3-4b-vllm-tp2-batch128-nccl-perlmutter \
    qwen3-4b-vllm-tp2-batch8-mscclpp-perlmutter \
    qwen3-4b-vllm-tp2-batch128-mscclpp-perlmutter \
    qwen3-4b-vllm-tp4-batch8-nccl-perlmutter \
    qwen3-4b-vllm-tp4-batch128-nccl-perlmutter \
    qwen3-4b-vllm-tp4-batch8-mscclpp-perlmutter \
    qwen3-4b-vllm-tp4-batch128-mscclpp-perlmutter' \
| ssh lambda7.cs.cornell.edu \
  'cd /data/ccl-bench_trace_collection && tar -xzf -'
```

The website config was updated to add these eight rows with
`metrics: "auto"` and `required_update: true`. The two earlier generic Qwen
Kineto rows:

- `qwen3-4b-vllm-tp4-batch8-perlmutter`
- `qwen3-4b-vllm-tp4-batch128-perlmutter`

were removed from the preview config so the dashboard only shows the explicit
NCCL/MSCCL++ variants.

Final Qwen matrix values after regeneration:

| Row | Comm | TP | Batch | Step Time | MFU | Dom Kern | Mem Bound | Avg Mem BW | Mem Xfer OH | Comm Frac | Comm Time |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen3-4b-vllm-tp2-batch8-nccl-perlmutter` | NCCL | 2 | 8 | 0.0411 | 5.71 | 55.52 | 65.27 | 4.33 | 14.58 | 58.71 | 7.98 |
| `qwen3-4b-vllm-tp2-batch8-mscclpp-perlmutter` | NCCL+MSCCL++ | 2 | 8 | 0.0385 | 5.61 | 54.84 | 64.51 | 4.55 | 15.64 | 57.84 | 7.68 |
| `qwen3-4b-vllm-tp2-batch128-nccl-perlmutter` | NCCL | 2 | 128 | 0.0419 | 45.47 | 30.21 | 29.93 | 12.70 | 10.65 | 29.59 | 10.57 |
| `qwen3-4b-vllm-tp2-batch128-mscclpp-perlmutter` | NCCL+MSCCL++ | 2 | 128 | 0.0413 | 44.89 | 31.60 | 29.87 | 12.05 | 11.52 | 29.53 | 10.19 |
| `qwen3-4b-vllm-tp4-batch8-nccl-perlmutter` | NCCL | 4 | 8 | 0.0427 | 5.58 | 76.59 | 85.53 | 4.20 | 15.80 | 79.09 | 28.57 |
| `qwen3-4b-vllm-tp4-batch8-mscclpp-perlmutter` | NCCL+MSCCL++ | 4 | 8 | 0.0420 | 4.96 | 67.89 | 79.73 | 5.11 | 17.44 | 70.66 | 19.33 |
| `qwen3-4b-vllm-tp4-batch128-nccl-perlmutter` | NCCL | 4 | 128 | 0.0436 | 58.14 | 38.59 | 51.14 | 21.46 | 13.06 | 49.54 | 29.50 |
| `qwen3-4b-vllm-tp4-batch128-mscclpp-perlmutter` | NCCL+MSCCL++ | 4 | 128 | 0.0429 | 55.29 | 36.44 | 48.31 | 23.73 | 13.69 | 46.51 | 27.70 |

Interpretation notes:

- Batch 128 has much higher MFU than batch 8 because prefill is GEMM-heavy and
  the batch has far more token-equivalents per latency measurement.
- TP4 communication fraction is higher than TP2, as expected from more
  tensor-parallel collectives.
- The MSCCL++ rows show lower communication fraction/time than their NCCL
  counterparts, especially at TP4 batch 8. This is consistent with the earlier
  NSYS evidence that MSCCL++ intercepts AllGather; AllReduce still remains
  NCCL.
- `mean_sm_coverage` and bandwidth utilization metrics remain `None` for the
  same tool limitations described later in this note.

## Llama and DeepSeek-MoE NCCL+MSCCL++ Follow-up

After the Qwen communication matrix was validated, the same MSCCL++ preload
path was collected for the other two model families that already had NCCL
baselines in the dashboard:

- `llama-3.1-8b-vllm-tp4-batch8-mscclpp-perlmutter`
- `llama-3.1-8b-vllm-tp4-batch128-mscclpp-perlmutter`
- `deepseek-moe-16b-vllm-tp4-ep4-batch8-mscclpp-perlmutter`
- `deepseek-moe-16b-vllm-tp4-ep4-batch128-mscclpp-perlmutter`

Scripts:

- `scripts/run_llama_deepseek_kineto_mscclpp_perlmutter.sh`
- `scripts/make_llama_deepseek_kineto_mscclpp_bundles_perlmutter.sh`
- `scripts/launch_llama_deepseek_kineto_mscclpp_perlmutter.sh`

The launcher used the same Perlmutter interactive allocation pattern:

```bash
bash scripts/launch_llama_deepseek_kineto_mscclpp_perlmutter.sh
```

It produced four bundles under:

```text
/pscratch/sd/k/kg597/ccl-bench-traces/llama-deepseek-kineto-mscclpp-bundles
```

and the bundles were streamed to the website host:

```bash
ssh perlmutter \
  'cd /pscratch/sd/k/kg597/ccl-bench-traces/llama-deepseek-kineto-mscclpp-bundles && tar -czf - \
    llama-3.1-8b-vllm-tp4-batch8-mscclpp-perlmutter \
    llama-3.1-8b-vllm-tp4-batch128-mscclpp-perlmutter \
    deepseek-moe-16b-vllm-tp4-ep4-batch8-mscclpp-perlmutter \
    deepseek-moe-16b-vllm-tp4-ep4-batch128-mscclpp-perlmutter' \
| ssh lambda7.cs.cornell.edu \
  'cd /data/ccl-bench_trace_collection && tar -xzf -'
```

Verified bundle sizes on `/data`:

| Bundle | Size |
|---|---:|
| `llama-3.1-8b-vllm-tp4-batch8-mscclpp-perlmutter` | 1010M |
| `llama-3.1-8b-vllm-tp4-batch128-mscclpp-perlmutter` | 1.1G |
| `deepseek-moe-16b-vllm-tp4-ep4-batch8-mscclpp-perlmutter` | 1.6G |
| `deepseek-moe-16b-vllm-tp4-ep4-batch128-mscclpp-perlmutter` | 1.7G |

The website config was updated to add these four rows with `metrics: "auto"`.
At the same time, the old four Qwen NSYS-only rows were removed from the
preview config:

- `qwen3-4b-vllm-tp2-perlmutter[nccl]`
- `qwen3-4b-vllm-tp2-perlmutter[mscclpp]`
- `qwen3-4b-vllm-tp4-perlmutter[nccl]`
- `qwen3-4b-vllm-tp4-perlmutter[mscclpp]`

Those four rows remain useful as historical NSYS evidence, but the preview now
uses the Kineto+NSYS Qwen rows because they provide Step Time and MFU.

Final Llama/DeepSeek NCCL vs NCCL+MSCCL++ comparison after regeneration:

| Row | Comm | Batch | Step Time | MFU | Dom Kern | Mem Bound | Avg Mem BW | Mem Xfer OH | MoE | Comm Frac | Comm Time |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `llama-3.1-8b-vllm-tp4-batch8-perlmutter` | NCCL | 8 | 0.0352 | 10.50 | 64.37 | 73.85 | 2.97 | 31.54 | 0.00 | 68.35 | 18.13 |
| `llama-3.1-8b-vllm-tp4-batch8-mscclpp-perlmutter` | NCCL+MSCCL++ | 8 | 0.0379 | 9.71 | 65.65 | 74.05 | 3.10 | 31.41 | 0.00 | 68.42 | 19.71 |
| `llama-3.1-8b-vllm-tp4-batch128-perlmutter` | NCCL | 128 | 0.0366 | 85.26 | 29.75 | 37.29 | 14.63 | 21.01 | 0.00 | 37.01 | 24.43 |
| `llama-3.1-8b-vllm-tp4-batch128-mscclpp-perlmutter` | NCCL+MSCCL++ | 128 | 0.0384 | 84.06 | 28.40 | 38.36 | 15.45 | 21.91 | 0.00 | 38.05 | 25.71 |
| `deepseek-moe-16b-vllm-tp4-ep4-batch8-perlmutter` | NCCL | 8 | 0.0721 | 1.70 | 78.84 | 89.95 | 3.39 | 26.98 | 7.32 | 82.15 | 53.93 |
| `deepseek-moe-16b-vllm-tp4-ep4-batch8-mscclpp-perlmutter` | NCCL+MSCCL++ | 8 | 0.0708 | 1.72 | 61.02 | 79.70 | 3.60 | 38.97 | 14.83 | 63.73 | 21.28 |
| `deepseek-moe-16b-vllm-tp4-ep4-batch128-perlmutter` | NCCL | 128 | 0.0812 | 21.60 | 44.16 | 56.63 | 17.62 | 34.44 | 19.20 | 53.65 | 49.50 |
| `deepseek-moe-16b-vllm-tp4-ep4-batch128-mscclpp-perlmutter` | NCCL+MSCCL++ | 128 | 0.0760 | 20.35 | 36.58 | 49.77 | 18.25 | 35.81 | 22.27 | 46.41 | 38.14 |

Interpretation notes:

- Llama shows little benefit from the mixed MSCCL++ preload path on this
  workload. Step time and communication fraction are roughly flat or slightly
  worse.
- DeepSeek-MoE shows a large reduction in communication time and communication
  fraction under `NCCL+MSCCL++`, especially batch 8.
- This still should be described as `NCCL+MSCCL++`, not pure MSCCL++, because
  LD_PRELOAD only interposes selected NCCL calls and earlier NSYS evidence
  showed AllReduce remaining NCCL.

## Pure MSCCL++ Batch-128 Follow-up

To test pure MSCCL++ rather than the mixed `NCCL+MSCCL++` path, the batch-128
rows were collected again with:

- `LD_PRELOAD=$HOME/mscclpp/build/lib/libmscclpp_nccl.so`
- `VLLM_NCCL_SO_PATH=$HOME/mscclpp/build/lib/libmscclpp_nccl.so`
- `MSCCLPP_NCCL_LIB_PATH` unset
- `--disable-custom-all-reduce`

The `VLLM_NCCL_SO_PATH` setting is required because vLLM's pynccl wrapper
loads NCCL with `ctypes.CDLL(...)`; `LD_PRELOAD` alone does not force pynccl to
use the shim. With the shim loaded directly and NCCL fallback unset, NSYS shows
MSCCL++ native kernels (`allreducePacket`, `allreduceFullmesh`,
`allgatherFullmesh2`) and no `ncclDevKernel_*` kernels.

Scripts:

- `scripts/run_pure_mscclpp_batch128_perlmutter.sh`
- `scripts/make_pure_mscclpp_batch128_bundles_perlmutter.sh`
- `scripts/launch_pure_mscclpp_batch128_perlmutter.sh`

Bundles were created under:

```text
/pscratch/sd/k/kg597/ccl-bench-traces/pure-mscclpp-batch128-bundles
```

and transferred to:

```text
/data/ccl-bench_trace_collection
```

Verified bundle sizes on `/data`:

| Bundle | Size |
|---|---:|
| `qwen3-4b-vllm-tp2-batch128-puremscclpp-perlmutter` | 673M |
| `qwen3-4b-vllm-tp4-batch128-puremscclpp-perlmutter` | 1.4G |
| `llama-3.1-8b-vllm-tp4-batch128-puremscclpp-perlmutter` | 1.1G |
| `deepseek-moe-16b-vllm-tp4-ep4-batch128-puremscclpp-perlmutter` | 1.6G |

Kernel validation from NSYS SQLite:

| Row | allreducePacket | allreduceFullmesh | allgatherFullmesh2 | NCCL kernels |
|---|---:|---:|---:|---:|
| `qwen3-4b-vllm-tp2-batch128-puremscclpp-perlmutter` | 5.032s | 3.680s | 0.224s | 0 |
| `qwen3-4b-vllm-tp4-batch128-puremscclpp-perlmutter` | 21.269s | 5.326s | 0.331s | 0 |
| `llama-3.1-8b-vllm-tp4-batch128-puremscclpp-perlmutter` | 14.463s | 7.800s | 0.343s | 0 |
| `deepseek-moe-16b-vllm-tp4-ep4-batch128-puremscclpp-perlmutter` | 28.064s | 5.756s | 0.253s | 0 |

Batch-128 NCCL vs mixed NCCL+MSCCL++ vs pure MSCCL++ comparison after website
regeneration:

| Row | Comm | Step Time | MFU | Dom Kern | Mem Bound | Avg Mem BW | Mem Xfer OH | MoE | Comm Frac | Comm Time |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen3-4b-vllm-tp2-batch128-nccl-perlmutter` | NCCL | 0.0419 | 40.04 | 30.21 | 29.93 | 12.70 | 10.65 | 0.00 | 29.59 | 10.57 |
| `qwen3-4b-vllm-tp2-batch128-mscclpp-perlmutter` | NCCL+MSCCL++ | 0.0413 | 39.52 | 31.60 | 29.87 | 12.05 | 11.52 | 0.00 | 29.53 | 10.19 |
| `qwen3-4b-vllm-tp2-batch128-puremscclpp-perlmutter` | MSCCL++ | 0.0444 | 78.11 | 31.13 | 25.86 | 2.60 | 9.86 | 0.00 | 25.47 | 8.95 |
| `qwen3-4b-vllm-tp4-batch128-nccl-perlmutter` | NCCL | 0.0436 | 51.19 | 38.59 | 51.14 | 21.46 | 13.06 | 0.00 | 49.54 | 29.50 |
| `qwen3-4b-vllm-tp4-batch128-mscclpp-perlmutter` | NCCL+MSCCL++ | 0.0429 | 48.68 | 36.44 | 48.31 | 23.73 | 13.69 | 0.00 | 46.51 | 27.70 |
| `qwen3-4b-vllm-tp4-batch128-puremscclpp-perlmutter` | MSCCL++ | 0.0459 | 46.73 | 36.56 | 48.28 | 2.33 | 12.92 | 0.00 | 46.48 | 27.04 |
| `llama-3.1-8b-vllm-tp4-batch128-perlmutter` | NCCL | 0.0366 | 85.26 | 29.75 | 37.29 | 14.63 | 21.01 | 0.00 | 37.01 | 24.43 |
| `llama-3.1-8b-vllm-tp4-batch128-mscclpp-perlmutter` | NCCL+MSCCL++ | 0.0384 | 84.06 | 28.40 | 38.36 | 15.45 | 21.91 | 0.00 | 38.05 | 25.71 |
| `llama-3.1-8b-vllm-tp4-batch128-puremscclpp-perlmutter` | MSCCL++ | 0.0389 | 80.24 | 29.50 | 34.60 | 2.37 | 21.58 | 0.00 | 34.26 | 22.73 |
| `deepseek-moe-16b-vllm-tp4-ep4-batch128-perlmutter` | NCCL | 0.0812 | 21.60 | 44.16 | 56.63 | 17.62 | 34.44 | 19.20 | 53.65 | 49.50 |
| `deepseek-moe-16b-vllm-tp4-ep4-batch128-mscclpp-perlmutter` | NCCL+MSCCL++ | 0.0760 | 20.35 | 36.58 | 49.77 | 18.25 | 35.81 | 22.27 | 46.41 | 38.14 |
| `deepseek-moe-16b-vllm-tp4-ep4-batch128-puremscclpp-perlmutter` | MSCCL++ | 0.0767 | 18.69 | 36.18 | 47.38 | 14.73 | 37.31 | 23.35 | 43.93 | 34.07 |

Interpretation notes:

- Pure MSCCL++ is now confirmed by kernel names, not just environment
  variables.
- Pure MSCCL++ generally reduces communication time compared with NCCL and the
  mixed NCCL+MSCCL++ path, especially for DeepSeek-MoE batch 128.
- End-to-end step time does not uniformly improve. Qwen TP4 and Llama are
  slower than the mixed path despite lower communication time, so the pure path
  should be presented as a communication-kernel experiment, not a blanket
  throughput win.
- The Qwen TP2 MFU is higher than the adjacent rows because the MFU tool uses
  the vLLM latency JSON for end-to-end throughput; review this before making
  a performance claim from MFU alone.

## Larger-Sequence Follow-up

Eric's feedback was that the original `batch=1, seq_len=1048` style traces were
too small to drive GPU utilization, and that batch size should be interpreted as
global batch size. After the batch-128 short-sequence rows, a cherry-picked
larger-sequence pass was collected for the model families where the communication
experiments were most useful:

- `qwen3-4b-vllm-tp4-batch64-in4096-out128-nccl-perlmutter`
- `qwen3-4b-vllm-tp4-batch64-in4096-out128-puremscclpp-perlmutter`
- `llama-3.1-8b-vllm-tp4-batch64-in4096-out128-nccl-perlmutter`
- `llama-3.1-8b-vllm-tp4-batch64-in4096-out128-puremscclpp-perlmutter`
- `deepseek-moe-16b-vllm-tp4-ep4-batch32-in3968-out128-nccl-perlmutter`
- `deepseek-moe-16b-vllm-tp4-ep4-batch32-in3968-out128-puremscclpp-perlmutter`

Scripts:

- `scripts/run_vllm_longseq_kineto_perlmutter.sh`
- `scripts/make_vllm_longseq_bundles_perlmutter.sh`
- `scripts/launch_vllm_longseq_kineto_perlmutter.sh`
- `scripts/run_vllm_longseq_deepseek_kineto_perlmutter.sh`
- `scripts/make_vllm_longseq_deepseek_bundles_perlmutter.sh`
- `scripts/launch_vllm_longseq_deepseek_perlmutter.sh`

The first launch attempted DeepSeek with `input_len=4096, output_len=128`, but
DeepSeek-MoE-16B has `max_position_embeddings=4096`, so vLLM rejected
`max_model_len=4224`. The DeepSeek rows were retried with
`input_len=3968, output_len=128` without using `VLLM_ALLOW_LONG_MAX_MODEL_LEN`.
The YAML `seq_len` remains `3968` because the inference cards record input
length for MFU/prefill accounting, with `output_len` tracked separately.

Perlmutter source bundles:

```text
/pscratch/sd/k/kg597/ccl-bench-traces/vllm-longseq-kineto-bundles
```

The full source bundles on Perlmutter keep all four rank traces. The `/data`
copies intentionally keep the files required by the website only:

- workload YAML
- `README.md`
- `.sqlite`
- `.nsys-rep`
- `.latency.json`
- `rank0_trace.json`

This avoids copying multiple GB of rank1-rank3 JSON over the local
Perlmutter-to-lambda relay. The full traces remain available on Perlmutter.

Verified `/data` bundle sizes after the minimal copy:

| Bundle | Size |
|---|---:|
| `qwen3-4b-vllm-tp4-batch64-in4096-out128-nccl-perlmutter` | 580M |
| `qwen3-4b-vllm-tp4-batch64-in4096-out128-puremscclpp-perlmutter` | 401M |
| `llama-3.1-8b-vllm-tp4-batch64-in4096-out128-nccl-perlmutter` | 469M |
| `llama-3.1-8b-vllm-tp4-batch64-in4096-out128-puremscclpp-perlmutter` | 320M |
| `deepseek-moe-16b-vllm-tp4-ep4-batch32-in3968-out128-nccl-perlmutter` | 787M |
| `deepseek-moe-16b-vllm-tp4-ep4-batch32-in3968-out128-puremscclpp-perlmutter` | 532M |

Kernel validation from NSYS SQLite:

| Row | NCCL kernels | MSCCL++ kernels | vLLM custom reduce |
|---|---:|---:|---:|
| `qwen3-4b-vllm-tp4-batch64-in4096-out128-nccl-perlmutter` | 6.819s | 0.000s | 15.820s |
| `qwen3-4b-vllm-tp4-batch64-in4096-out128-puremscclpp-perlmutter` | 0.000s | 21.108s | 0.000s |
| `llama-3.1-8b-vllm-tp4-batch64-in4096-out128-nccl-perlmutter` | 9.786s | 0.000s | 9.688s |
| `llama-3.1-8b-vllm-tp4-batch64-in4096-out128-puremscclpp-perlmutter` | 0.000s | 16.872s | 0.000s |
| `deepseek-moe-16b-vllm-tp4-ep4-batch32-in3968-out128-nccl-perlmutter` | 4.245s | 0.000s | 34.510s |
| `deepseek-moe-16b-vllm-tp4-ep4-batch32-in3968-out128-puremscclpp-perlmutter` | 0.000s | 28.108s | 0.000s |

Final larger-sequence comparison after website regeneration:

| Row | Comm | Batch | Input | Output | Step Time | MFU | Dom Kern | Mem Bound | Avg Mem BW | Mem Xfer OH | MoE | Comm Frac | Comm Time |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen3-4b-vllm-tp4-batch64-in4096-out128-nccl-perlmutter` | NCCL | 64 | 4096 | 128 | 0.0448 | 88.15 | 27.56 | 40.94 | 7.94 | 12.70 | 0.00 | 39.43 | 22.64 |
| `qwen3-4b-vllm-tp4-batch64-in4096-out128-puremscclpp-perlmutter` | MSCCL++ | 64 | 4096 | 128 | 0.0477 | 83.87 | 25.09 | 39.48 | 2.42 | 13.50 | 0.00 | 37.99 | 21.11 |
| `llama-3.1-8b-vllm-tp4-batch64-in4096-out128-nccl-perlmutter` | NCCL | 64 | 4096 | 128 | 0.0353 | 122.43 | 39.33 | 28.96 | 6.11 | 21.40 | 0.00 | 28.88 | 19.47 |
| `llama-3.1-8b-vllm-tp4-batch64-in4096-out128-puremscclpp-perlmutter` | MSCCL++ | 64 | 4096 | 128 | 0.0412 | 117.19 | 40.99 | 26.32 | 2.35 | 22.41 | 0.00 | 26.23 | 16.87 |
| `deepseek-moe-16b-vllm-tp4-ep4-batch32-in3968-out128-nccl-perlmutter` | NCCL | 32 | 3968 | 128 | 0.0804 | 22.54 | 55.53 | 66.07 | 10.92 | 42.58 | 14.18 | 62.36 | 38.75 |
| `deepseek-moe-16b-vllm-tp4-ep4-batch32-in3968-out128-puremscclpp-perlmutter` | MSCCL++ | 32 | 3968 | 128 | 0.0824 | 22.94 | 46.47 | 59.23 | 10.60 | 45.70 | 17.00 | 54.82 | 28.11 |

Interpretation notes:

- The larger-sequence rows answer the feedback that the previous global batch
  and sequence length were too small.
- Pure MSCCL++ again reduces communication time/fraction, but end-to-end step
  time is slightly slower for all three cherry-picked comparisons.
- Llama MFU still exceeds 100% after correcting `seq_len` to input length. That
  is a metric-definition problem: the current MFU formula is training/prefill
  flavored and is not an inference-aware prefill/decode FLOP model. Treat MFU
  for vLLM inference rows as a caveat requiring tool work, not as a literal
  hardware-utilization claim.

## Second Pass: vLLM Kineto JSON Collection

The second pass collected Kineto JSON traces for three model families, two batch
sizes each:

- `qwen3-4b-vllm-tp4-batch8-perlmutter`
- `qwen3-4b-vllm-tp4-batch128-perlmutter`
- `llama-3.1-8b-vllm-tp4-batch8-perlmutter`
- `llama-3.1-8b-vllm-tp4-batch128-perlmutter`
- `deepseek-moe-16b-vllm-tp4-ep4-batch8-perlmutter`
- `deepseek-moe-16b-vllm-tp4-ep4-batch128-perlmutter`

Collection script:

- `scripts/run_vllm_kineto_perlmutter_batch.sh`

Bundle script:

- `scripts/make_vllm_kineto_bundles_perlmutter.sh`

Common workload parameters:

- Input length: 1024
- Output length: 128
- Sequence length recorded in YAML: 1024
- Batch sizes: 8 and 128
- Precision: BF16
- Hardware: 1 Perlmutter node, 4 x NVIDIA A100 40 GB
- Framework: `vllm-0.19.0`
- Compiler field: `plain_pytorch`
- Communication library: NCCL
- `NCCL_IB_DISABLE=1`
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- `--enforce-eager`
- `--disable-detokenize`

The collection command shape was:

```bash
module load conda
conda activate /pscratch/sd/k/kg597/session1/cvllm
cd ~/ccl-bench
bash scripts/run_vllm_kineto_perlmutter_batch.sh
bash scripts/make_vllm_kineto_bundles_perlmutter.sh
```

Each bundle contains:

- `<bundle>.yaml`
- `rank0_trace.json` through `rank3_trace.json`
- NSYS `.nsys-rep`
- NSYS `.sqlite`
- `logs/*.latency.json`
- raw bench/profile/server logs as available
- `README.md`

## Workload Card Decisions

The Kineto rows use `phase: inference`, not `serving`, because we want them
grouped with inference rows in the dashboard.

The compiler field is set to:

```yaml
compiler_tool_selection: plain_pytorch
```

This is because the vLLM runs used `--enforce-eager`, disabling CUDA graph
capture for the measured run. Earlier names like `eager`, `continuous`, or
`no_cudagraph` were avoided because they do not match the existing website
metadata pattern. The dashboard column is `Compiler`, and `plain_pytorch` is
the existing site convention for eager-mode PyTorch.

Batch is set in the YAML:

```yaml
workload:
  data:
    batch_size: 8
    seq_len: 1024
```

or:

```yaml
workload:
  data:
    batch_size: 128
    seq_len: 1024
```

The model architecture fields were added so MFU can be computed:

Qwen3-4B:

- `num_params: 4022468096`
- `num_params_embedding: 388956160`
- `num_layers: 36`
- `num_heads: 32`
- `head_dim: 128`

Llama-3.1-8B:

- `num_params: 8030261248`
- `num_params_embedding: 1050673152`
- `num_layers: 32`
- `num_heads: 32`
- `head_dim: 128`

DeepSeek-MoE-16B:

- `num_params: 16375728128`
- `num_params_embedding: 419430400`
- `num_params_active: 2828650496`
- `num_layers: 28`
- `num_heads: 16`
- `head_dim: 128`

For DeepSeek-MoE, `num_params_active` is used for MFU so inactive routed
experts are not charged on every token.

## Metric Fixes Added

Two tool changes were made locally, copied to Perlmutter, and copied to
`~/ccl-bench-main` on the website host.

### `tools/avg_step_time/avg_step_time.py`

Before this change, GPU JSON step time only supported PyTorch profiler
`ProfilerStep#N` events. The vLLM Kineto traces do not contain those markers.
They do contain `execute_context_*` user annotation ranges.

The new behavior:

- Keep existing `ProfilerStep#N` behavior for training/Kineto JSON traces.
- If no `ProfilerStep#N` ranges are found, fall back to vLLM
  `execute_context_*` user annotation ranges.
- Trim first and last range when more than two ranges are present, matching the
  existing warmup/tail trimming pattern.
- Convert Kineto durations from microseconds to seconds.

This avoids using `vllm bench latency` `avg_latency` as Step Time. That latency
is full-batch end-to-end latency, not a model step or engine iteration.

### `tools/mfu/mfu.py`

MFU was extended for vLLM bundles:

- Look for `*.latency.json` under the bundle.
- Use `avg_latency` from vLLM bench latency to compute tokens/second for MFU.
- Add `num_params_active` support for MoE models.
- Keep the original `ProfilerStep#N` fallback when no vLLM latency JSON exists.

This split is intentional:

- Step Time uses Kineto `execute_context_*` engine iteration ranges.
- MFU uses end-to-end batch latency to estimate achieved token throughput.

## Validated Metric Values

Validated with:

```bash
cd ~/ccl-bench
python tools/main.py --trace /pscratch/sd/k/kg597/ccl-bench-traces/vllm-kineto-bundles/<bundle> --metric avg_step_time
python tools/main.py --trace /pscratch/sd/k/kg597/ccl-bench-traces/vllm-kineto-bundles/<bundle> --metric mfu
```

Results:

| Bundle | Step Time s | MFU % |
|---|---:|---:|
| qwen3-4b-vllm-tp4-batch8-perlmutter | 0.0428635 | 5.2227 |
| qwen3-4b-vllm-tp4-batch128-perlmutter | 0.0479903 | 56.9764 |
| llama-3.1-8b-vllm-tp4-batch8-perlmutter | 0.0352361 | 11.8712 |
| llama-3.1-8b-vllm-tp4-batch128-perlmutter | 0.0365609 | 85.2567 |
| deepseek-moe-16b-vllm-tp4-ep4-batch8-perlmutter | 0.0721431 | 1.7038 |
| deepseek-moe-16b-vllm-tp4-ep4-batch128-perlmutter | 0.0812221 | 21.5991 |

The large MFU for Llama batch 128 should be treated as a number to review
before publication. It comes from the current repository MFU formula and vLLM
batch latency, not from hardware counters.

## Transfer to Website Storage

The six bundles were streamed directly from Perlmutter to the storage host:

```bash
ssh -o IdentitiesOnly=yes perlmutter \
  'cd /pscratch/sd/k/kg597/ccl-bench-traces/vllm-kineto-bundles && tar -czf - \
    qwen3-4b-vllm-tp4-batch8-perlmutter \
    qwen3-4b-vllm-tp4-batch128-perlmutter \
    llama-3.1-8b-vllm-tp4-batch8-perlmutter \
    llama-3.1-8b-vllm-tp4-batch128-perlmutter \
    deepseek-moe-16b-vllm-tp4-ep4-batch8-perlmutter \
    deepseek-moe-16b-vllm-tp4-ep4-batch128-perlmutter' \
| ssh lambda7.cs.cornell.edu \
  'cd /data/ccl-bench_trace_collection && tar -xzf -'
```

Verified sizes on `/data`:

| Bundle | Size |
|---|---:|
| qwen3-4b-vllm-tp4-batch8-perlmutter | 1.4G |
| qwen3-4b-vllm-tp4-batch128-perlmutter | 1.4G |
| llama-3.1-8b-vllm-tp4-batch8-perlmutter | 1.1G |
| llama-3.1-8b-vllm-tp4-batch128-perlmutter | 1.3G |
| deepseek-moe-16b-vllm-tp4-ep4-batch8-perlmutter | 1.8G |
| deepseek-moe-16b-vllm-tp4-ep4-batch128-perlmutter | 1.9G |

## Website Generation

The generator reads:

```text
website/benchmark_config.json
```

Each pair points to a trace directory under:

```text
/data/ccl-bench_trace_collection
```

The six Kineto entries were added to `pairs` as:

```json
{
  "trace": "/data/ccl-bench_trace_collection/qwen3-4b-vllm-tp4-batch8-perlmutter",
  "metrics": "auto",
  "required_update": true
}
```

and similarly for the other five bundles.

Regeneration command:

```bash
cd ~/ccl-bench-main
python3 website/generate_data.py
```

Preview server:

```bash
cd ~/ccl-bench-main
python3 -m http.server 8081 --bind 127.0.0.1
```

Local tunnel:

```bash
ssh -fN -L 8081:localhost:8081 lambda7.cs.cornell.edu
```

Open:

```text
http://localhost:8081/
```

## Dashboard State

After the first generation with explicit metrics, the six Kineto rows showed
only Step Time and MFU. That was because the config used:

```json
"metrics": ["avg_step_time", "mfu"]
```

The config was then changed to:

```json
"metrics": "auto"
```

and a hard regeneration was run so all JSON-supported metrics were attempted.

Expected JSON metric set:

- `avg_step_time`
- `mfu`
- `mean_sm_coverage`
- `dominant_kernel_concentration`
- `memory_bound_fraction`
- `moe_fraction`
- `average_memory_bandwidth`
- `memory_transfer_overhead`
- `communication_fraction`
- `total_communication_time`
- `bandwidth_utilization_allgather_group_6`
- `bandwidth_utilization_allreduce_group_6`

Some metrics may still return `None` because the corresponding tool is not
implemented for this trace shape, cannot find expected communication events, or
has a pre-existing parser limitation. That is acceptable for preview, but it
should be documented before final publication.

First-pass regeneration result (auto mode, YAML still declaring
`metric_source.traces: [json]`):

- `website/benchmark_data.json`: 47 rows, 14 metrics.
- The six new rows populated only `avg_step_time`, `mfu`, and
  `communication_fraction`.
- Every kernel-reading metric returned `None` (memory-bound, average memory
  bandwidth, memory transfer overhead, dominant kernel concentration, total
  communication time, mean SM coverage, MoE fraction, allgather/allreduce
  bandwidth utilization).

## Root Cause of the Empty Metrics

Each tool reports its own stderr. Every kernel-reading failure reduced to the
same root cause: the tools were fed Kineto JSON traces that contained **no GPU
kernel events**. Enumerating event categories in `rank0_trace.json` for
`qwen3-4b-vllm-tp4-batch128-perlmutter`:

```text
('X', 'cpu_op')             558025
('i', 'cpu_instant_event')  145955
('X', 'user_annotation')       392
('M', None)                      8
('X', 'Trace')                   1
```

There are no `('X', 'kernel')` events at all. The vLLM Kineto profiler was
configured with CPU-side activities only, so the Kineto JSON files capture CPU
operator events and user annotations (`ProfilerStep`, `execute_context_*`) but
not GPU kernel launches. That is why `avg_step_time` (ProfilerStep-based),
`mfu` (vLLM latency JSON), and `communication_fraction` (degrades gracefully)
succeeded, while every metric that requires per-kernel fields failed.

At the same time, each bundle also contains a full NSYS capture
(`<bundle>.nsys-rep` and `<bundle>.sqlite`) with GPU kernel data. The metric
tools that support both trace types (`mean_sm_coverage`,
`total_communication_time`, `dominant_kernel_concentration`,
`memory_bound_fraction`, `average_memory_bandwidth`, `memory_transfer_overhead`,
`moe_fraction`) already dispatch on the YAML `metric_source.traces` list via
`_get_trace_types(directory)`, checking `nsys` before `json`. The JSON-side was
selected only because the YAML did not advertise `nsys`.

## Fix Applied

The YAML `metric_source.traces` list was widened to advertise both trace
artifacts that actually ship in the bundle. For each of the six bundles on the
storage host:

```yaml
metric_source:
  traces:
    - nsys
    - json
  metrics_specific_trace:
    - nsys
    - vllm_bench_latency
```

This does not add any metric that was not already auto-selected under `[json]`
alone, because the auto-selection in `website/generate_data.py` intersects the
YAML `trace_types` set with each metric's registered `trace_types` set. It
does flip the per-metric dispatch inside each tool: with `nsys` present first,
the tool reads from the NSYS SQLite (which has kernel data) instead of the
Kineto JSON (which does not).

Steps executed on the website host (`singh-compute-07`, SSH alias `lambda7`):

```bash
# 1. Add nsys to metric_source.traces in all six YAMLs on /data.
for d in /data/ccl-bench_trace_collection/qwen3-4b-vllm-tp4-batch8-perlmutter \
         /data/ccl-bench_trace_collection/qwen3-4b-vllm-tp4-batch128-perlmutter \
         /data/ccl-bench_trace_collection/llama-3.1-8b-vllm-tp4-batch8-perlmutter \
         /data/ccl-bench_trace_collection/llama-3.1-8b-vllm-tp4-batch128-perlmutter \
         /data/ccl-bench_trace_collection/deepseek-moe-16b-vllm-tp4-ep4-batch8-perlmutter \
         /data/ccl-bench_trace_collection/deepseek-moe-16b-vllm-tp4-ep4-batch128-perlmutter; do
  python3 -c "import re,sys;p=sys.argv[1];s=open(p).read();open(p,'w').write(
    re.sub(r'(\n  traces:\n)(    - json\n)', r'\1    - nsys\n\2', s, count=1))" "$d"/*.yaml
done

# 2. Force recompute on the six pairs.
python3 -c "
import json
from pathlib import Path
p = Path('website/benchmark_config.json')
cfg = json.loads(p.read_text())
for pair in cfg['pairs']:
    if 'perlmutter' in pair['trace'] and 'batch' in pair['trace']:
        pair['required_update'] = True
p.write_text(json.dumps(cfg, indent=2))
"

# 3. Regenerate.
python3 website/generate_data.py
```

## Final Metric Coverage for the Six Rows

After the fix, ten of the twelve auto-selected metrics populate. Snapshot of
the values now stored in `website/benchmark_data.json`:

| Row | Step Time | MFU | Dom Kern | Mem Bound | Avg Mem BW | Mem Xfer OH | MoE | Comm Frac | Total Comm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3-4b-vllm-tp4-batch8-nccl-perlmutter | 0.0427 | 4.91 | 76.59 | 85.53 | 4.20 | 15.80 | 0.0 | 79.09 | 28.57 |
| qwen3-4b-vllm-tp4-batch128-nccl-perlmutter | 0.0436 | 51.19 | 38.59 | 51.14 | 21.46 | 13.06 | 0.0 | 49.54 | 29.50 |
| llama-3.1-8b-vllm-tp4-batch8-perlmutter | 0.0352 | 10.50 | 64.37 | 73.85 | 2.97 | 31.54 | 0.0 | 68.35 | 18.13 |
| llama-3.1-8b-vllm-tp4-batch128-perlmutter | 0.0366 | 85.26 | 29.75 | 37.29 | 14.63 | 21.01 | 0.0 | 37.01 | 24.43 |
| deepseek-moe-16b-vllm-tp4-ep4-batch8-perlmutter | 0.0721 | 1.70 | 78.84 | 89.95 | 3.39 | 26.98 | 7.32 | 82.15 | 53.93 |
| deepseek-moe-16b-vllm-tp4-ep4-batch128-perlmutter | 0.0812 | 21.60 | 44.16 | 56.63 | 17.62 | 34.44 | 19.20 | 53.65 | 49.50 |

Sanity check on `moe_fraction`: 0 for dense Qwen3-4B and Llama-3.1-8B, non-zero
for DeepSeek-MoE (batch 128 keeps more experts active than batch 8, which is
the expected direction).

## Follow-up Fixes After First Sanity Check

A first pass at the six populated rows revealed two more issues that needed
attention:

**`communication_fraction` silently routed to a stale tool.** `tools/main.py`
had two `elif metric_name == "communication_fraction":` blocks. The earlier
one (near line 151) routed to `utilization-group-21/utilization.py` with
`metric_type="comm_fraction"`, which reads a Chrome-trace JSON and does a
different calculation. Python's elif chain always hit that first block, so the
proper NSYS-aware implementation in `tools/communication_fraction/` was never
invoked. Direct NSYS aggregation for qwen3-4b-vllm-tp4-batch128 gives 46%
comm time; the stale path produced 0.003%. The fix was to delete the earlier
block. After deletion the NSYS values for the six new rows land in the
37–82% range, consistent with `total_communication_time` / total kernel time
in the SQLite.

**Compiler field.** The YAMLs initially declared
`compiler_tool_selection: disabled`. No other row in the site uses that
string. Two existing rows use `plain_pytorch` for eager PyTorch, which is the
closest match for `--enforce-eager` vLLM (no CUDA graphs, no torch.compile).
All six YAMLs were changed to `compiler_tool_selection: plain_pytorch`.

## Note on vLLM Inference Step Time and MFU

`mfu.py` computes
```
FLOP/token    = 6(N_active - N_emb) + 12·L·H·Q·S
tokens/second = (batch_size * seq_len) / avg_latency
MFU           = (FLOP/token · tokens/second) / (world_size · peak_TFLOPS)
```

For `llama-3.1-8b-vllm-tp4-batch128-perlmutter` this gives 85.26% after the
`seq_len=input_len` correction. The math checks out against the vLLM
`avg_latency` and the hardware peak (312 TFLOPS × 4 GPUs), but the metric
should still be reviewed before publication. The larger-sequence Llama rows
remain above 100%, which shows that the current formula is not defensible as a
literal inference utilization metric.

Two caveats should be raised in review:

- Step Time and MFU use different timing sources. `avg_step_time` uses Kineto
  `execute_context_*` ranges, which are vLLM engine execution intervals. MFU
  uses the vLLM benchmark latency JSON (`avg_latency`) to estimate tokens/sec.
  These are intentionally different clocks, but they are easy to misread as the
  same "step".
- MFU uses a training/prefill-style FLOP model. It charges
  `6 * (params - embeddings)` per token plus an attention term. For forward-only
  vLLM inference, prefill and decode should be modeled separately and the
  parameter term should not be assumed to match training-step accounting.

In other words, the >100% Llama values are not a Kineto parsing failure and not
a hardware result. They indicate that vLLM inference needs an inference-aware
MFU metric, or MFU should be omitted for these rows until that tool work lands.

## Remaining `None` Values

Two metric groups still return `None` and the root cause is tool-side, not
configuration:

- `mean_sm_coverage` — the NSYS path queries
  `SELECT gpuId, numMultiprocessors FROM TARGET_INFO_CUDA_DEVICE`. The sqlite
  produced by the NSYS version used on Perlmutter does not have a
  `numMultiprocessors` column, so the query errors out and the JSON fallback
  has no kernel data either. Fixing this requires either updating the tool to
  tolerate the newer/older NSYS schema or baking the per-device SM count into
  a constant for A100 (108 SMs). Out of scope for this change.
- `bandwidth_utilization_{allgather,allreduce,alltoall,peertopeer}_group_6` —
  these tools read Kineto JSON and search for `nccl:all_gather`,
  `nccl:all_reduce`, etc. kernel events. The JSON has no kernel events, so
  the tools legitimately find nothing. Implementing an NSYS backend for them
  is the proper fix, but it is outside the scope of the website update.

These `None` cells should be treated as honest gaps in tool coverage rather
than bugs in the bundles.

## Mem Bound Caveat

Current website description:

```text
Fraction of kernel time classified as memory-bound. Uses SM coverage for NSYS
traces and kernel-category heuristics for Kineto JSON traces
```

This is accurate for the current implementation, but it is not strong enough
for an academic claim. The advisor concern is valid: a defensible Mem Bound
definition should not be based only on kernel names or low SM coverage.

Recommended standard before using Mem Bound in paper claims:

```text
A kernel/window is memory-bound if achieved memory bandwidth exceeds X percent
of architecture peak sustained memory bandwidth while achieved compute
throughput is below Y percent of architecture peak compute throughput.
```

One possible formula:

```text
memory_bound_fraction =
  sum(duration_i for i where BW_i / BW_peak >= X and FLOPS_i / FLOPS_peak < Y)
  / sum(duration_i)
```

This requires reliable per-kernel or per-window memory bytes and FLOPs. Kineto
JSON may not always provide those fields in the same way across TPU/GPU and
training/inference traces. Until that is implemented, Mem Bound should be
treated as a dashboard heuristic, not as a research conclusion.

## Files That Should Be Considered for PR

Core metric code:

- `tools/avg_step_time/avg_step_time.py`
- `tools/mfu/mfu.py`

Website data/config after final regeneration:

- `website/benchmark_config.json`
- `website/benchmark_data.json`
- `website/data.js`

Collection/reproduction scripts:

- `scripts/run_vllm_kineto_perlmutter_batch.sh`
- `scripts/make_vllm_kineto_bundles_perlmutter.sh`
- `scripts/run_qwen3_4b_kineto_comm_matrix_perlmutter.sh`
- `scripts/make_qwen3_4b_kineto_comm_bundles_perlmutter.sh`
- `scripts/launch_qwen3_4b_kineto_comm_matrix_perlmutter.sh`
- `scripts/sbatch_qwen3_4b_kineto_comm_matrix_perlmutter.sh`
- `scripts/run_llama_deepseek_kineto_mscclpp_perlmutter.sh`
- `scripts/make_llama_deepseek_kineto_mscclpp_bundles_perlmutter.sh`
- `scripts/launch_llama_deepseek_kineto_mscclpp_perlmutter.sh`
- `scripts/run_pure_mscclpp_batch128_perlmutter.sh`
- `scripts/make_pure_mscclpp_batch128_bundles_perlmutter.sh`
- `scripts/launch_pure_mscclpp_batch128_perlmutter.sh`
- `scripts/run_vllm_longseq_kineto_perlmutter.sh`
- `scripts/make_vllm_longseq_bundles_perlmutter.sh`
- `scripts/launch_vllm_longseq_kineto_perlmutter.sh`
- `scripts/run_vllm_longseq_deepseek_kineto_perlmutter.sh`
- `scripts/make_vllm_longseq_deepseek_bundles_perlmutter.sh`
- `scripts/launch_vllm_longseq_deepseek_perlmutter.sh`
- `scripts/vllm_kineto_perlmutter_workflow.md`

Trace collection cards:

- The six model/batch bundle YAML files and eight Qwen comm-matrix YAML files
  plus the four Llama/DeepSeek MSCCL++ YAML files, four pure-MSCCL++ batch-128
  YAML files, and six larger-sequence YAML files should be committed under
  `trace_collection/` if the repository convention is to include workload cards
  locally.
- The large JSON, SQLite, and NSYS artifacts should not be committed to git.
  They should live in `/data/ccl-bench_trace_collection`.

Avoid committing unrelated local scratch files:

- `.claude/`
- `*.exp` helper files
- old qwen NSYS bundles unless they are explicitly part of the PR
- local dirty website files generated against the wrong filesystem

## Immediate Next Steps

1. Inspect the 63-row preview in `http://localhost:8081/`, especially the Qwen,
   Llama, and DeepSeek NCCL/MSCCL++/pure-MSCCL++ comparisons and the six
   larger-sequence rows.
2. Confirm whether the `None` values are acceptable for this PR.
3. Decide whether the new rows should keep `metrics: "auto"` or explicitly
   list only defensible metrics.
4. Decide whether the old four NSYS Qwen rows stay in the PR alongside the new
   Kineto rows. They answer different questions:
   - old four: NCCL vs MSCCL++ NSYS kernel/communication evidence
   - new eight Qwen rows: comparable NCCL/MSCCL++ Kineto+NSYS data with Step
     Time/MFU
   - six model/batch rows: Qwen, Llama, and DeepSeek batch-size comparison
   - four follow-up rows: Llama and DeepSeek NCCL+MSCCL++ comparison
   - four pure-MSCCL++ batch-128 rows
   - six larger-sequence rows
5. Commit only the necessary files to PR #35 after review.

## Recommended PR Message

Suggested summary:

```text
Add Perlmutter vLLM Kineto benchmark rows and extend JSON metrics for vLLM.

- Add vLLM Kineto fallback in avg_step_time using execute_context_* ranges.
- Extend MFU to use vLLM latency JSON and active MoE parameter counts.
- Add Perlmutter vLLM batch8/batch128 rows for Qwen3-4B, Llama-3.1-8B, and DeepSeek-MoE-16B.
- Add explicit Qwen3-4B NCCL vs NCCL+MSCCL++ rows for TP2/TP4 and batch8/batch128.
- Add Llama-3.1-8B and DeepSeek-MoE-16B NCCL+MSCCL++ comparison rows.
- Add pure-MSCCL++ batch-128 rows and larger-sequence rows for Qwen3-4B, Llama-3.1-8B, and DeepSeek-MoE-16B.
- Regenerate website benchmark data from /data/ccl-bench_trace_collection.
```

Suggested caveat:

```text
The memory-bound metric remains a heuristic in the current repository. It should
not be used as a paper-level claim until redefined using architecture-normalized
memory bandwidth and compute throughput.
```
