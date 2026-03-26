"""
Metric: memory_bound_fraction
Description: Percentage of kernel time classified as memory-bound.
             For NSYS traces, this uses an SM coverage heuristic.
             For Kineto JSON traces, this uses kernel-category heuristics.
Unit: Percentage (%)
Returns: Float between 0-100, or -1 if data unavailable
"""

import sqlite3
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trace_metric_utils import (
    find_sqlite_file,
    get_trace_types,
    load_yaml,
    summarize_kineto_kernel_breakdown,
)


def _calc_nsys(path: str) -> float:
    sqlite_path = find_sqlite_file(path)
    if sqlite_path is None:
        print(f"[memory_bound_fraction/nsys] No .sqlite file found in {path}", file=sys.stderr)
        return -1.0

    try:
        conn = sqlite3.connect(sqlite_path)
        cursor = conn.cursor()
        device_row = cursor.execute(
            """
            SELECT numMultiprocessors
            FROM TARGET_INFO_CUDA_DEVICE
            LIMIT 1
            """
        ).fetchone()
        if device_row is None or device_row[0] in (None, 0):
            conn.close()
            return -1.0

        num_sms = device_row[0]

        kernels = cursor.execute(
            """
            SELECT (end - start) as duration, gridX, gridY, gridZ
            FROM CUPTI_ACTIVITY_KIND_KERNEL
            """
        ).fetchall()
        conn.close()

        if not kernels:
            return -1.0

        total_time = 0.0
        memory_bound_time = 0.0
        for duration, grid_x, grid_y, grid_z in kernels:
            if duration is None:
                continue
            blocks_per_grid = (grid_x or 0) * (grid_y or 0) * (grid_z or 0)
            sm_coverage = min(blocks_per_grid / num_sms, 1.0) * 100 if num_sms else 0
            total_time += duration
            if sm_coverage < 50:
                memory_bound_time += duration
        if total_time <= 0:
            return -1.0

        return float((memory_bound_time / total_time) * 100)
    except Exception as e:
        print(f"[memory_bound_fraction/nsys] {e}", file=sys.stderr)
        return -1.0


def _calc_json(directory: str) -> float:
    summary = summarize_kineto_kernel_breakdown(directory)
    if summary is None:
        return -1.0

    total_kernel_dur, breakdown = summary
    memory_bound_dur = sum(
        breakdown.get(category, 0.0)
        for category in ("elementwise", "normalization", "softmax", "reduction", "memory_transfer")
    )
    if total_kernel_dur <= 0:
        return -1.0

    return round((memory_bound_dur / total_kernel_dur) * 100.0, 4)


def metric_cal(directory: str) -> float:
    yaml_data = load_yaml(directory)
    trace_types = get_trace_types(yaml_data)

    if "nsys" in trace_types:
        return _calc_nsys(directory)
    if "json" in trace_types:
        return _calc_json(directory)
    if "json_tpu" in trace_types:
        print(
            f"[memory_bound_fraction] json_tpu traces are not supported for {directory}",
            file=sys.stderr,
        )
        return -1.0

    print(f"[memory_bound_fraction] No supported trace type in {trace_types}", file=sys.stderr)
    return -1.0


def calculate_metric(path: str) -> float:
    """Backward-compatible wrapper used by older imports."""
    return metric_cal(path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python memory_bound_fraction_group_9.py <trace_directory_or_sqlite_file>")
        sys.exit(1)

    print(metric_cal(sys.argv[1]))
