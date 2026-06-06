#!/usr/bin/env python3
"""
Stage 12 parser: segment-compare NVTX CSV -> per-range CSV + derived metrics.

Derived (plan §17):
  attention_core_speedup           = baseline_attn_mean / guide_attn_mean
  layout_total_ms                  = hidden_reorder + rope_reorder + reverse_restore
  layout_amortized_per_window_layer= layout_total / num_window_layers
  baseline_equivalent_per_layer    = baseline_attn_mean + layout_amortized
  guide_equivalent_per_layer       = guide_attn_mean
  replacement_equivalent_speedup   = baseline_equivalent_per_layer / guide_attn_mean
"""
import argparse
import csv
import json
import re
from pathlib import Path


def find_duration_col(header):
    for cand in ["Projected Duration (ns)", "Duration (ns)", "Total Time (ns)"]:
        if cand in header:
            return cand
    for h in header:
        if "Duration" in h:
            return h
    raise SystemExit(f"no duration column in {header}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--info", default="")
    ap.add_argument("--derived", default="")
    args = ap.parse_args()

    inp = Path(args.input)
    base = inp.parent
    info = {}
    if args.info and Path(args.info).exists():
        info = json.loads(Path(args.info).read_text())
    else:
        for c in base.rglob("segment_info.json"):
            info = json.loads(c.read_text()); break
    PI = int(info.get("profile_iters", 1)) or 1
    NWL = int(info.get("num_window_layers", 28)) or 28

    rows = list(csv.DictReader(open(inp)))
    dur_col = find_duration_col(rows[0].keys()) if rows else None

    agg = {}
    for r in rows:
        name = (r.get("Name") or "").lstrip(":")
        if not (name.startswith("baseline.") or name.startswith("guide.")):
            continue
        agg.setdefault(name, 0.0)
        agg[name] += float(r[dur_col])

    out_rows = []
    for name, dur_ns in sorted(agg.items()):
        mean_ms = dur_ns / 1e6 / PI
        is_guide = name.startswith("guide.")
        is_layout = ".layout." in name
        is_attn = ".attn_core." in name
        m = re.search(r"layer(\d+)", name)
        out_rows.append({
            "range_name": name,
            "range_type": "layout" if is_layout else ("attn_core" if is_attn else "other"),
            "layer_idx": int(m.group(1)) if m else "",
            "profile_iters": PI,
            "total_projected_gpu_time_ms": round(dur_ns / 1e6, 6),
            "mean_ms_per_iter": round(mean_ms, 8),
            "is_baseline": not is_guide,
            "is_guide": is_guide,
            "includes_layout_reorder": is_layout,
            "includes_attention_core": is_attn,
            "head_dim": info.get("head_dim", ""), "num_heads": info.get("num_heads", ""),
            "H": info.get("H", ""), "W": info.get("W", ""), "S": info.get("S", ""),
            "num_windows": info.get("num_windows", ""),
            "tokens_per_window": info.get("tokens_per_window", ""),
            "block_size": info.get("block_size", ""), "num_warps": info.get("num_warps", ""),
            "num_stages": info.get("num_stages", ""),
        })
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader(); w.writerows(out_rows)

    def mean_of(pred):
        vals = [r["mean_ms_per_iter"] for r in out_rows if pred(r)]
        return sum(vals) / len(vals) if vals else 0.0

    base_attn = mean_of(lambda r: r["is_baseline"] and r["includes_attention_core"])
    guide_attn = mean_of(lambda r: r["is_guide"] and r["includes_attention_core"])
    lay_hidden = mean_of(lambda r: "hidden_reorder" in r["range_name"])
    lay_rope = mean_of(lambda r: "rope_reorder" in r["range_name"])
    lay_rev = mean_of(lambda r: "reverse_restore" in r["range_name"])
    layout_total = lay_hidden + lay_rope + lay_rev
    layout_amort = layout_total / NWL
    base_equiv = base_attn + layout_amort

    derived = {
        "baseline_attention_core_mean_ms": round(base_attn, 8),
        "guide_attention_core_mean_ms": round(guide_attn, 8),
        "attention_core_speedup": round(base_attn / guide_attn, 4) if guide_attn else None,
        "layout_hidden_reorder_ms": round(lay_hidden, 8),
        "layout_rope_reorder_ms": round(lay_rope, 8),
        "layout_reverse_restore_ms": round(lay_rev, 8),
        "layout_total_ms": round(layout_total, 8),
        "num_window_layers": NWL,
        "layout_amortized_per_window_layer_ms": round(layout_amort, 8),
        "baseline_equivalent_per_layer_ms": round(base_equiv, 8),
        "guide_equivalent_per_layer_ms": round(guide_attn, 8),
        "replacement_equivalent_speedup": round(base_equiv / guide_attn, 4) if guide_attn else None,
    }
    dpath = args.derived or str(base / "segment_compare_derived.json")
    Path(dpath).write_text(json.dumps(derived, indent=2))

    print("[parse-segment] derived metrics:")
    for k, v in derived.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
