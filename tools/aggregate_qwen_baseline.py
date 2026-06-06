#!/usr/bin/env python3
"""
Aggregate Qwen2.5-VL baseline timing results from multiple stage runs
into a unified QWEN25VL_BASELINE_SUMMARY.csv.
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path
from collections import defaultdict


SUMMARY_FIELDS = [
    "run_id", "stage", "benchmark", "sample_id", "model_path", "precision",
    "device_name", "resolution", "max_new_tokens", "batch_size", "seed",
    "temperature", "do_sample",
    "num_input_text_tokens", "num_visual_tokens_before_merger", "num_visual_tokens_after_merger",
    "actual_generated_tokens",
    "request_total_ms", "image_preprocess_ms", "processor_ms", "visual_encoder_ms",
    "visual_merger_ms", "llm_prefill_ms", "ttft_ms", "decode_loop_ms", "tpot_ms", "e2e_ms",
    "request_output_tok_s", "decode_only_tok_s",
    "f_vision", "f_visual_prefill", "f_decode",
    "generated_text_path", "nsys_report_path", "ncu_report_path",
    "status", "error_message",
]


def collect_jsonl(path: Path, stage_label: str, extra_fields: dict = None) -> list:
    rows = []
    if not path.exists():
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                r["stage"] = stage_label
                if extra_fields:
                    r.update(extra_fields)
                rows.append(r)
            except json.JSONDecodeError:
                pass
    return rows


def normalize_row(row: dict, defaults: dict) -> dict:
    out = {f: "" for f in SUMMARY_FIELDS}
    out.update(defaults)
    for k, v in row.items():
        if k in SUMMARY_FIELDS:
            out[k] = v
    return out


def main():
    parser = argparse.ArgumentParser(description="Aggregate Qwen baseline results")
    parser.add_argument("--output-root", required=True,
                        help="Path to output/qwen25vl_baseline/")
    parser.add_argument("--model-path", default="models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--precision", default="bf16")
    parser.add_argument("--device-name", default="")
    args = parser.parse_args()

    root = Path(args.output_root)
    all_rows = []

    defaults = {
        "model_path": args.model_path,
        "precision": args.precision,
        "device_name": args.device_name,
        "temperature": 0.0,
        "do_sample": False,
        "batch_size": 1,
        "seed": 42,
        "status": "OK",
        "error_message": "",
        "generated_text_path": "",
        "nsys_report_path": "",
        "ncu_report_path": "",
    }

    # Stage 0.75: smoke results
    smoke_dir = root / "stage0_75_model_data_smoke"
    for fname in ["smoke_generated_outputs.jsonl", "smoke_timing.jsonl"]:
        rows = collect_jsonl(smoke_dir / fname, "stage0_75_smoke", defaults)
        if rows:
            all_rows.extend(rows)
            break  # only need one

    # Stage 1: generation sanity
    s1_dir = root / "stage1_generation_sanity"
    rows = collect_jsonl(s1_dir / "generated_outputs.jsonl", "stage1_sanity", defaults)
    all_rows.extend(rows)

    # Stage 2: app timing
    s2_dir = root / "stage2_app_timing_sanity"
    rows = collect_jsonl(s2_dir / "timing_raw.jsonl", "stage2_app_timing", defaults)
    all_rows.extend(rows)

    # Stage 5: resolution sweep
    s5_dir = root / "stage5_resolution_sweep"
    rows = collect_jsonl(s5_dir / "resolution_sweep_raw.jsonl", "stage5_resolution_sweep", defaults)
    all_rows.extend(rows)
    # Also collect per-config run dirs
    for run_dir in sorted(s5_dir.glob("*/timing_raw.jsonl")):
        extra = {"resolution": _extract_resolution_from_path(str(run_dir))}
        rows = collect_jsonl(run_dir, "stage5_resolution_sweep", {**defaults, **extra})
        all_rows.extend(rows)

    # Stage 6: output length sweep
    s6_dir = root / "stage6_output_length_sweep"
    rows = collect_jsonl(s6_dir / "output_length_raw.jsonl", "stage6_output_length_sweep", defaults)
    all_rows.extend(rows)
    for run_dir in sorted(s6_dir.glob("*/timing_raw.jsonl")):
        rows = collect_jsonl(run_dir, "stage6_output_length_sweep", defaults)
        all_rows.extend(rows)

    # Stage 7: workload table
    s7_dir = root / "stage7_workload_table"
    rows = collect_jsonl(s7_dir / "workload_raw.jsonl", "stage7_workload_table", defaults)
    all_rows.extend(rows)

    # Stage 8: visual encoder breakdown
    s8_dir = root / "stage8_visual_encoder_breakdown"
    rows = collect_jsonl(s8_dir / "visual_encoder_breakdown_raw.jsonl",
                          "stage8_ve_breakdown", defaults)
    all_rows.extend(rows)

    if not all_rows:
        print("WARNING: no rows collected from any stage")
        all_rows = []

    # Normalize
    normalized = [normalize_row(r, defaults) for r in all_rows]

    # Write summary
    out_path = root / "QWEN25VL_BASELINE_SUMMARY.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(normalized)

    print(f"Written: {out_path} ({len(normalized)} rows)")
    return len(normalized)


def _extract_resolution_from_path(path_str: str) -> int:
    import re
    m = re.search(r"res(\d+)", path_str)
    return int(m.group(1)) if m else 0


if __name__ == "__main__":
    n = main()
    sys.exit(0 if n >= 0 else 1)
