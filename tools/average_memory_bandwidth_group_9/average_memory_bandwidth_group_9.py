"""
Metric: average_memory_bandwidth
Description: Average memory bandwidth achieved during memory copy operations.
             Supported for NSYS traces with CUPTI memcpy data.
Unit: GB/s (Gigabytes per second)
Returns: Float (GB/s), or -1 if data unavailable
"""

import sqlite3
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trace_metric_utils import find_sqlite_file, get_trace_types, load_yaml


def _calc_nsys(path: str) -> float:
    sqlite_path = find_sqlite_file(path)
    if sqlite_path is None:
        print(f"[average_memory_bandwidth/nsys] No .sqlite file found in {path}", file=sys.stderr)
        return -1.0

    try:
        conn = sqlite3.connect(sqlite_path)
        cursor = conn.cursor()
        has_memcpy = cursor.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='CUPTI_ACTIVITY_KIND_MEMCPY'
            LIMIT 1
            """
        ).fetchone()
        if not has_memcpy:
            conn.close()
            return -1.0

        memcpy_rows = cursor.execute(
            """
            SELECT (end - start) as duration, bytes
            FROM CUPTI_ACTIVITY_KIND_MEMCPY
            """
        ).fetchall()
        conn.close()

        if not memcpy_rows:
            return -1.0

        bandwidths = []
        for duration, byte_count in memcpy_rows:
            if not duration or not byte_count:
                continue
            bandwidth_gbps = (byte_count / duration) * 1e9 / 1e9
            if bandwidth_gbps > 0:
                bandwidths.append(bandwidth_gbps)

        if not bandwidths:
            return -1.0

        return float(sum(bandwidths) / len(bandwidths))
    except Exception as e:
        print(f"[average_memory_bandwidth/nsys] {e}", file=sys.stderr)
        return -1.0


def metric_cal(directory: str) -> float:
    yaml_data = load_yaml(directory)
    trace_types = get_trace_types(yaml_data)

    if "nsys" in trace_types:
        return _calc_nsys(directory)
    if "json" in trace_types or "json_tpu" in trace_types:
        print(
            f"[average_memory_bandwidth] No real memory bandwidth counters available for trace types {trace_types}",
            file=sys.stderr,
        )
        return -1.0

    print(f"[average_memory_bandwidth] No supported trace type in {trace_types}", file=sys.stderr)
    return -1.0


def calculate_metric(path: str) -> float:
    """Backward-compatible wrapper used by older imports."""
    return metric_cal(path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python average_memory_bandwidth_group_9.py <trace_directory_or_sqlite_file>")
        sys.exit(1)

    print(metric_cal(sys.argv[1]))
