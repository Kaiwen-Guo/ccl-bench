"""
Metric: memory_transfer_overhead
Description: Percentage of trace time spent in memory transfer operations.
             For NSYS traces, this is derived from CUPTI memcpy activity.
             For Kineto JSON traces, this is estimated from copy-like kernel names.
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
        print(f"[memory_transfer_overhead/nsys] No .sqlite file found in {path}", file=sys.stderr)
        return -1.0

    try:
        conn = sqlite3.connect(sqlite_path)
        cursor = conn.cursor()
        has_memcpy = cursor.execute(
            """
            SELECT 1 FROM sqlite_master
            WHERE type='table' AND name='CUPTI_ACTIVITY_KIND_MEMCPY'
            LIMIT 1
            """
        ).fetchone()
        if not has_memcpy:
            conn.close()
            return -1.0

        memcpy_rows = cursor.execute(
            """
            SELECT start, end, (end - start) as duration
            FROM CUPTI_ACTIVITY_KIND_MEMCPY
            """
        ).fetchall()
        kernel_rows = cursor.execute(
            """
            SELECT start, end
            FROM CUPTI_ACTIVITY_KIND_KERNEL
            """
        ).fetchall()
        conn.close()

        if not memcpy_rows or not kernel_rows:
            return -1.0

        total_memcpy_time = sum(row[2] for row in memcpy_rows if row[2] is not None)
        trace_start = min(min(row[0] for row in kernel_rows), min(row[0] for row in memcpy_rows))
        trace_end = max(max(row[1] for row in kernel_rows), max(row[1] for row in memcpy_rows))
        trace_duration = trace_end - trace_start

        if trace_duration <= 0:
            return -1.0

        return float((total_memcpy_time / trace_duration) * 100)
    except Exception as e:
        print(f"[memory_transfer_overhead/nsys] {e}", file=sys.stderr)
        return -1.0


def _calc_json(directory: str) -> float:
    summary = summarize_kineto_kernel_breakdown(directory)
    if summary is None:
        return -1.0

    total_kernel_dur, breakdown = summary
    mem_transfer_dur = breakdown.get("memory_transfer", 0.0)
    if total_kernel_dur <= 0:
        return -1.0

    return round((mem_transfer_dur / total_kernel_dur) * 100.0, 4)


def metric_cal(directory: str) -> float:
    yaml_data = load_yaml(directory)
    trace_types = get_trace_types(yaml_data)

    if "nsys" in trace_types:
        return _calc_nsys(directory)
    if "json" in trace_types:
        return _calc_json(directory)
    if "json_tpu" in trace_types:
        print(
            f"[memory_transfer_overhead] json_tpu traces are not supported for {directory}",
            file=sys.stderr,
        )
        return -1.0

    print(f"[memory_transfer_overhead] No supported trace type in {trace_types}", file=sys.stderr)
    return -1.0


def calculate_metric(path: str) -> float:
    """Backward-compatible wrapper used by older imports."""
    return metric_cal(path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python memory_transfer_overhead_group_9.py <trace_directory_or_sqlite_file>")
        sys.exit(1)

    print(metric_cal(sys.argv[1]))
