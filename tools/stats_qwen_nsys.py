#!/usr/bin/env python3
"""
Parse nsys stats nvtx_gpu_proj_trace CSV and summarize per NVTX range.
Produces nsys_summary.csv with mean/total/fraction per range.
"""
import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from collections import defaultdict


KEY_RANGES = [
    "request_total",
    "image_preprocess",
    "processor",
    "visual_encoder",
    "visual_merger",
    "patch_embed",
    "vit.patch_embed",
    "llm_prefill",
    "decode_loop",
    "generate_total",
    "smoke_total",
    "smoke_matmul",
]

BLOCK_PATTERN = re.compile(r"vit\.blocks\.(\d+)\.(.*)")


def parse_nsys_csv(csv_path: str):
    """
    Parse nsys stats nvtx_gpu_proj_trace CSV.
    Column headers vary by nsys version; handle flexibly.
    Returns list of dicts with normalized fields.
    """
    rows = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        # Skip comment lines starting with #
        lines = [l for l in f.readlines() if not l.startswith("#")]

    reader = csv.DictReader(lines)
    for row in reader:
        rows.append(dict(row))

    return rows


def normalize_col(row: dict, candidates: list, default=None):
    for c in candidates:
        if c in row and row[c] not in ("", None):
            try:
                return float(row[c].replace(",", "").replace(" ", ""))
            except (ValueError, AttributeError):
                return default
    return default


def summarize_nvtx(rows: list) -> dict:
    """
    Aggregate rows by NVTX range name.
    Returns: {range_name: {"total_ns": ..., "count": ..., "mean_ns": ...}}
    """
    agg = defaultdict(lambda: {"total_ns": 0.0, "count": 0})

    for row in rows:
        # Detect range name column
        name = (row.get("Range") or row.get("Name") or row.get("range") or
                row.get("RangeName") or "")
        if not name:
            continue
        # Strip leading colon (nsys adds domain prefix like ":range_name")
        name = name.lstrip(":")

        # Duration: try multiple column names (nsys uses ns)
        dur_ns = normalize_col(row, [
            "Projected Duration (ns)", "Duration (ns)", "Total Time (ns)",
            "GPU Duration (ns)", "GpuDuration", "duration_ns",
            "Avg (ns)", "Total (ns)",
        ])
        if dur_ns is None:
            dur_ns = normalize_col(row, [
                "Projected Duration", "Duration", "Total Time",
                "GPU Duration", "GpuDuration", "Avg", "Total",
            ])
            if dur_ns is not None:
                dur_ns *= 1e6  # assume ms → ns

        if dur_ns is None:
            continue

        agg[name]["total_ns"] += dur_ns
        agg[name]["count"] += 1

    result = {}
    for name, d in agg.items():
        result[name] = {
            "total_ns": d["total_ns"],
            "count": d["count"],
            "mean_ns": d["total_ns"] / d["count"] if d["count"] > 0 else 0,
            "total_ms": d["total_ns"] / 1e6,
            "mean_ms": d["total_ns"] / d["count"] / 1e6 if d["count"] > 0 else 0,
        }
    return result


def compute_fractions(summary: dict) -> dict:
    """Compute fraction of each range relative to request_total."""
    ref_key = next((k for k in ["request_total", "generate_total", "smoke_total"]
                    if k in summary), None)
    if ref_key is None:
        return summary

    ref_ms = summary[ref_key]["mean_ms"]
    if ref_ms <= 0:
        return summary

    result = dict(summary)
    for name in result:
        result[name]["fraction"] = result[name]["mean_ms"] / ref_ms

    return result


def main():
    parser = argparse.ArgumentParser(description="Summarize nsys nvtx_gpu_proj_trace CSV")
    parser.add_argument("--input", required=True, help="Path to nsys nvtx_gpu_proj_trace CSV")
    parser.add_argument("--output", required=True, help="Output summary CSV path")
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"ERROR: input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    rows = parse_nsys_csv(args.input)
    if not rows:
        print(f"WARNING: no rows in {args.input}")
        # Create empty output
        Path(args.output).write_text("range_name,total_ms,mean_ms,count,fraction\n")
        sys.exit(0)

    summary = summarize_nvtx(rows)
    summary = compute_fractions(summary)

    # Write output
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["range_name", "total_ms", "mean_ms", "count", "fraction", "total_ns", "mean_ns"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for name, vals in sorted(summary.items()):
            writer.writerow({
                "range_name": name,
                "total_ms": round(vals["total_ms"], 4),
                "mean_ms": round(vals["mean_ms"], 4),
                "count": vals["count"],
                "fraction": round(vals.get("fraction", 0.0), 6),
                "total_ns": round(vals["total_ns"], 0),
                "mean_ns": round(vals["mean_ns"], 0),
            })

    print(f"Written: {out_path} ({len(summary)} ranges)")

    # Check required ranges
    found = set(summary.keys())
    required = {"request_total", "visual_encoder", "decode_loop"}
    missing = required - found
    if missing:
        print(f"WARNING: required ranges not found: {missing}")
        print(f"  Available: {sorted(found)[:20]}")

    # Print key summary
    for k in ["request_total", "visual_encoder", "visual_merger", "llm_prefill", "decode_loop"]:
        if k in summary:
            v = summary[k]
            print(f"  {k}: mean={v['mean_ms']:.3f}ms count={v['count']} frac={v.get('fraction',0):.3f}")


if __name__ == "__main__":
    main()
