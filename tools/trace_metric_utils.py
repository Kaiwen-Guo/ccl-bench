import json
import os
import sys
from collections import defaultdict

from json_sampling import select_json_files


def load_yaml(directory: str) -> dict:
    for fn in os.listdir(directory):
        if fn.endswith(".yaml"):
            with open(os.path.join(directory, fn), encoding="utf-8", errors="replace") as f:
                import yaml

                return yaml.safe_load(f) or {}
    return {}


def get_trace_types(yaml_data: dict) -> list:
    return yaml_data.get("metric_source", {}).get("traces", [])


def find_sqlite_file(path: str) -> str | None:
    """Find an SQLite trace in a directory or return the file path directly."""
    path = os.path.abspath(path)

    if os.path.isfile(path) and path.endswith(".sqlite"):
        return path

    if os.path.isdir(path):
        sqlite_files = [f for f in os.listdir(path) if f.endswith(".sqlite")]
        if not sqlite_files:
            return None
        non_profiling = [f for f in sqlite_files if "profiling" not in f.lower()]
        selected = non_profiling[0] if non_profiling else sqlite_files[0]
        return os.path.abspath(os.path.join(path, selected))

    return None


def load_json_events(path: str) -> list:
    """Load traceEvents from a PyTorch-profiler JSON file; partial-parse fallback."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                content = f.read()
            idx = content.find('"traceEvents"')
            if idx == -1:
                return []
            bracket = content.find("[", idx)
            if bracket == -1:
                return []
            partial = content[bracket:]
            data = None
            for suffix in ("]}", "]}}}"):
                try:
                    data = json.loads(partial + suffix)
                    break
                except json.JSONDecodeError:
                    pass
            if data is None:
                return []
            if isinstance(data, list):
                return data
        except Exception:
            return []
    if isinstance(data, dict):
        return data.get("traceEvents", [])
    return []


def select_rank_json_files(directory: str) -> list[str]:
    files = [
        f
        for f in select_json_files(directory)
        if os.path.basename(f).startswith(("rank", "kineto"))
    ]
    return files


def classify_compute_kernel(name: str) -> str:
    n = name.lower()
    if "flash_fwd" in n or ("flash" in n and "attention" in n and "bwd" not in n and "backward" not in n):
        return "attention"
    if "flash_bwd" in n or ("flash" in n and ("bwd" in n or "backward" in n)):
        return "attention_backward"
    if any(p in n for p in ["gemm", "cutlass", "ampere_bf16", "sm80_xmma", "sm90_xmma", "matmul", "cublas"]):
        return "gemm"
    if any(p in n for p in ["elementwise", "vectorized", "silu", "gelu", "relu", "swish"]):
        return "elementwise"
    if any(p in n for p in ["layernorm", "layer_norm", "rmsnorm", "rms_norm"]):
        return "normalization"
    if "softmax" in n:
        return "softmax"
    if any(p in n for p in ["moe", "expert", "topk", "gating"]):
        return "moe"
    if any(p in n for p in ["memcpy", "memset"]) or "copy" in n:
        return "memory_transfer"
    if "reduce" in n:
        return "reduction"
    if any(p in n for p in ["adam", "sgd", "optimizer", "multi_tensor_apply"]):
        return "optimizer"
    return "other_compute"


def summarize_kineto_kernel_breakdown(directory: str) -> tuple[float, dict[str, float]] | None:
    """
    Summarize kernel durations across sampled rank/kineto JSON traces.

    Returns:
        (total_kernel_duration_us, categorized_non_nccl_duration_us)
        or None if no supported JSON kernel events are found.
    """
    rank_files = select_rank_json_files(directory)
    if not rank_files:
        print(f"[kineto] No rank/kineto JSON traces found in {directory}", file=sys.stderr)
        return None

    total_kernel_dur = 0.0
    breakdown = defaultdict(float)
    files_with_kernel_events = 0

    for path in rank_files:
        events = load_json_events(path)
        kernel_events = [
            e
            for e in events
            if isinstance(e, dict)
            and e.get("cat") == "kernel"
            and isinstance(e.get("dur"), (int, float))
        ]
        if not kernel_events:
            continue

        files_with_kernel_events += 1
        total_kernel_dur += sum(e["dur"] for e in kernel_events)

        for event in kernel_events:
            name = event.get("name", "")
            if "nccl" in name.lower():
                continue
            breakdown[classify_compute_kernel(name)] += event["dur"]

    if files_with_kernel_events == 0 or total_kernel_dur <= 0:
        print(f"[kineto] No kernel events found in sampled JSON traces for {directory}", file=sys.stderr)
        return None

    return total_kernel_dur, dict(breakdown)
