#!/usr/bin/env python3
"""
Amdahl's Law analysis for Qwen2.5-VL baseline.
Computes predicted E2E speedup for hypothetical visual encoder speedups.

S_e2e = 1 / (1 - f_vision + f_vision / S_vision)
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path
from collections import defaultdict


S_VISION_CANDIDATES = [1.2, 1.5, 2.0, 3.0]


def load_summary(csv_path: str) -> list:
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def parse_float(v, default=0.0):
    try:
        return float(v) if v not in ("", None) else default
    except (ValueError, TypeError):
        return default


def amdahl_speedup(f_vision: float, s_vision: float) -> float:
    if s_vision <= 0:
        return 0.0
    denom = (1.0 - f_vision) + f_vision / s_vision
    return 1.0 / denom if denom > 0 else float("inf")


def main():
    parser = argparse.ArgumentParser(description="Amdahl analysis for Qwen baseline")
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = load_summary(args.summary_csv)
    if not summary:
        print("ERROR: empty summary CSV", file=sys.stderr)
        # Create placeholder outputs
        _write_empty(out_dir)
        sys.exit(1)

    # Filter rows with valid f_vision and e2e_ms
    valid = [r for r in summary
             if parse_float(r.get("f_vision")) > 0
             and parse_float(r.get("e2e_ms")) > 0]

    if not valid:
        print(f"WARNING: no valid rows with f_vision > 0 (total rows: {len(summary)})")
        _write_empty(out_dir)
        return

    # Per-config aggregation: group by (stage, benchmark, resolution, max_new_tokens)
    groups = defaultdict(list)
    for r in valid:
        key = (
            r.get("stage", ""),
            r.get("benchmark", ""),
            str(r.get("resolution", "")),
            str(r.get("max_new_tokens", "")),
        )
        groups[key].append(r)

    amdahl_rows = []
    for (stage, benchmark, resolution, max_new_tokens), rows in sorted(groups.items()):
        f_v_vals = [parse_float(r.get("f_vision")) for r in rows]
        e2e_vals = [parse_float(r.get("e2e_ms")) for r in rows]
        ve_vals = [parse_float(r.get("visual_encoder_ms")) for r in rows]
        dec_vals = [parse_float(r.get("f_decode")) for r in rows]

        mean_f_v = sum(f_v_vals) / len(f_v_vals)
        mean_e2e = sum(e2e_vals) / len(e2e_vals)
        mean_ve = sum(ve_vals) / len(ve_vals)
        mean_f_d = sum(dec_vals) / len(dec_vals)

        base_row = {
            "stage": stage,
            "benchmark": benchmark,
            "resolution": resolution,
            "max_new_tokens": max_new_tokens,
            "n_samples": len(rows),
            "mean_f_vision": round(mean_f_v, 6),
            "mean_e2e_ms": round(mean_e2e, 3),
            "mean_visual_encoder_ms": round(mean_ve, 3),
            "mean_f_decode": round(mean_f_d, 6),
        }

        for sv in S_VISION_CANDIDATES:
            se2e = amdahl_speedup(mean_f_v, sv)
            base_row[f"S_e2e_at_Sv{sv:.1f}"] = round(se2e, 4)

        amdahl_rows.append(base_row)

    if not amdahl_rows:
        _write_empty(out_dir)
        return

    # Find best GUIDE candidates
    # High f_vision + short output = best target
    best_rows = sorted(amdahl_rows,
                        key=lambda r: r.get("S_e2e_at_Sv2.0", 0),
                        reverse=True)

    # Write amdahl_raw.csv
    fieldnames = list(amdahl_rows[0].keys())
    with open(out_dir / "amdahl_raw.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(amdahl_rows)

    # Write amdahl_summary.csv (top configs)
    with open(out_dir / "amdahl_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(best_rows[:20])

    print(f"Written amdahl_raw.csv ({len(amdahl_rows)} configs)")

    # Print key results
    print("\n=== AMDAHL ANALYSIS RESULTS ===")
    print(f"{'Stage':<25} {'Benchmark':<12} {'Res':<6} {'Tok':<6} "
          f"{'f_v':<7} {'S_e2e@2x':<9} {'S_e2e@3x':<9}")
    print("-" * 85)
    for r in best_rows[:15]:
        print(f"{r['stage']:<25} {r['benchmark']:<12} {r['resolution']:<6} "
              f"{r['max_new_tokens']:<6} {r['mean_f_vision']:<7.3f} "
              f"{r.get('S_e2e_at_Sv2.0', '?'):<9} {r.get('S_e2e_at_Sv3.0', '?'):<9}")

    # Best GUIDE target
    if best_rows:
        best = best_rows[0]
        print(f"\n>>> Best +GUIDE target: benchmark={best['benchmark']} "
              f"resolution={best['resolution']} max_new_tokens={best['max_new_tokens']}")
        print(f"    f_vision={best['mean_f_vision']:.3f}  "
              f"S_e2e@2x={best.get('S_e2e_at_Sv2.0','?')}  "
              f"S_e2e@3x={best.get('S_e2e_at_Sv3.0','?')}")

    # Write report
    with open(out_dir / "amdahl_report.md", "w") as f:
        f.write("# Amdahl Analysis: Qwen2.5-VL Baseline\n\n")
        f.write("## Formula\n\n")
        f.write("```\nf_vision = visual_encoder_ms / e2e_ms\n")
        f.write("S_e2e = 1 / (1 - f_vision + f_vision / S_vision)\n```\n\n")
        f.write("## Results by Config\n\n")
        headers = ["benchmark", "resolution", "max_new_tokens", "mean_f_vision",
                   "S_e2e_at_Sv2.0", "S_e2e_at_Sv3.0", "n_samples"]
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for r in best_rows[:20]:
            vals = [str(r.get(h, "")) for h in headers]
            f.write("| " + " | ".join(vals) + " |\n")

        if best_rows:
            best = best_rows[0]
            f.write(f"\n## Recommended +GUIDE Target\n\n")
            f.write(f"- **benchmark**: {best['benchmark']}\n")
            f.write(f"- **resolution**: {best['resolution']}\n")
            f.write(f"- **max_new_tokens**: {best['max_new_tokens']}\n")
            f.write(f"- **f_vision**: {best['mean_f_vision']}\n")
            f.write(f"- **S_e2e at S_vision=2x**: {best.get('S_e2e_at_Sv2.0', '?')}\n")
            f.write(f"- **S_e2e at S_vision=3x**: {best.get('S_e2e_at_Sv3.0', '?')}\n")

            # Interpretation
            fv = best["mean_f_vision"]
            se2 = best.get("S_e2e_at_Sv2.0", 0)
            se3 = best.get("S_e2e_at_Sv3.0", 0)
            if fv >= 0.35:
                interpretation = f"f_vision={fv:.3f} >= 0.35: GUIDE visual-prefill case study is strongly justified."
            elif fv >= 0.25:
                interpretation = f"f_vision={fv:.3f} ∈ [0.25, 0.35): GUIDE is applicable, moderate E2E impact."
            else:
                interpretation = f"f_vision={fv:.3f} < 0.25: visual encoder is not the dominant bottleneck at this config."
            f.write(f"\n**Interpretation**: {interpretation}\n")

    # Write stage status
    with open(out_dir / "stage_status.json", "w") as f:
        json.dump({
            "status": "PASS",
            "num_configs_analyzed": len(amdahl_rows),
            "best_f_vision": best_rows[0]["mean_f_vision"] if best_rows else 0,
            "best_S_e2e_2x": best_rows[0].get("S_e2e_at_Sv2.0", 0) if best_rows else 0,
            "best_config": {
                "benchmark": best_rows[0]["benchmark"],
                "resolution": best_rows[0]["resolution"],
                "max_new_tokens": best_rows[0]["max_new_tokens"],
            } if best_rows else {},
        }, f, indent=2)

    print(f"\nAmdahl analysis complete. Output: {out_dir}")


def _write_empty(out_dir: Path):
    for fname in ["amdahl_raw.csv", "amdahl_summary.csv"]:
        (out_dir / fname).write_text("stage,benchmark,resolution,max_new_tokens,"
                                      "mean_f_vision,S_e2e_at_Sv2.0\n")
    with open(out_dir / "stage_status.json", "w") as f:
        json.dump({"status": "FAIL", "message": "no valid data"}, f)
    with open(out_dir / "amdahl_report.md", "w") as f:
        f.write("# Amdahl Analysis\n\nNo valid data found.\n")


if __name__ == "__main__":
    main()
