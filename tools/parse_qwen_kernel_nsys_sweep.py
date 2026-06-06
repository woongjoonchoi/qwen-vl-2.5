#!/usr/bin/env python3
"""
Stage 11 parser: nsys nvtx_gpu_proj_trace CSV -> block-sweep summary + winner.

Winner rule (plan §16):
  primary  : lowest mean_ms_per_iter among correctness-PASS candidates
  tie-break: lower BLOCK_SIZE within 3%
Only correctness-passing candidates are eligible.
"""
import argparse
import csv
import json
import re
from pathlib import Path

RANGE_RE = re.compile(r"guide\.block_sweep\.block(\d+)\.warps(\d+)\.stages(\d+)")


def find_duration_col(header):
    for cand in ["Projected Duration (ns)", "Duration (ns)", "Total Time (ns)",
                 "Projected Duration", "Duration"]:
        if cand in header:
            return cand
    # fallback: first column containing 'Duration'
    for h in header:
        if "Duration" in h:
            return h
    raise SystemExit(f"no duration column in {header}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="nvtx_gpu_proj_trace CSV")
    ap.add_argument("--output", required=True, help="summary CSV")
    ap.add_argument("--info", default="")
    ap.add_argument("--correctness", default="")
    ap.add_argument("--best", default="")
    args = ap.parse_args()

    inp = Path(args.input)
    base_dir = inp.parent
    info = {}
    if args.info and Path(args.info).exists():
        info = json.loads(Path(args.info).read_text())
    else:
        # search nearby for bench_info.json
        for c in base_dir.rglob("bench_info.json"):
            info = json.loads(c.read_text()); break
    correctness = {}
    cpath = args.correctness
    if not cpath:
        for c in base_dir.rglob("candidates_correctness.json"):
            cpath = str(c); break
    if cpath and Path(cpath).exists():
        correctness = json.loads(Path(cpath).read_text())

    profile_iters = int(info.get("profile_iters", 1)) or 1

    rows = list(csv.DictReader(open(inp)))
    dur_col = find_duration_col(rows[0].keys()) if rows else None

    # aggregate duration per range tag (sum across any duplicate rows)
    agg = {}
    for r in rows:
        name = (r.get("Name") or "").lstrip(":")
        m = RANGE_RE.fullmatch(name)
        if not m:
            continue
        bs, nw, ns = (int(x) for x in m.groups())
        dur_ns = float(r[dur_col])
        tag = f"block{bs}.warps{nw}.stages{ns}"
        agg.setdefault(tag, {"bs": bs, "nw": nw, "ns": ns, "dur_ns": 0.0})
        agg[tag]["dur_ns"] += dur_ns

    out_rows = []
    for tag, a in agg.items():
        mean_ms = a["dur_ns"] / 1e6 / profile_iters
        corr = correctness.get(tag, {})
        out_rows.append({
            "block_size": a["bs"], "num_warps": a["nw"], "num_stages": a["ns"],
            "mode": info.get("mode", ""), "target_layer": info.get("target_layer", ""),
            "head_dim": info.get("head_dim", ""), "num_heads": info.get("num_heads", ""),
            "H": info.get("H", ""), "W": info.get("W", ""), "S": info.get("S", ""),
            "num_windows": info.get("num_windows", ""),
            "tokens_per_window": info.get("tokens_per_window", ""),
            "num_iterations": profile_iters,
            "nvtx_gpu_projected_time_total_ms": round(a["dur_ns"] / 1e6, 6),
            "mean_ms_per_iter": round(mean_ms, 8),
            "correctness_status": corr.get("status", "UNKNOWN"),
            "rel_l2_error": corr.get("rel_l2_error", ""),
        })

    out_rows.sort(key=lambda r: (r["block_size"], r["num_warps"], r["num_stages"]))
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader(); w.writerows(out_rows)

    # winner: lowest mean among PASS, tie-break lower block within 3%
    passing = [r for r in out_rows if r["correctness_status"] == "PASS"]
    best = None
    if passing:
        fastest = min(passing, key=lambda r: r["mean_ms_per_iter"])
        thr = fastest["mean_ms_per_iter"] * 1.03
        within = [r for r in passing if r["mean_ms_per_iter"] <= thr]
        best = min(within, key=lambda r: (r["block_size"], r["mean_ms_per_iter"]))

    best_path = args.best or str(base_dir / "best_block_config.json")
    Path(best_path).write_text(json.dumps(best or {}, indent=2))

    print(f"[parse-sweep] {len(out_rows)} candidates, {len(passing)} PASS")
    for r in out_rows:
        print(f"  bs={r['block_size']:<3} w={r['num_warps']} s={r['num_stages']} "
              f"mean={r['mean_ms_per_iter']*1e3:8.2f}us  {r['correctness_status']}")
    if best:
        print(f"[parse-sweep] WINNER: block={best['block_size']} "
              f"warps={best['num_warps']} stages={best['num_stages']} "
              f"mean={best['mean_ms_per_iter']*1e3:.2f}us")


if __name__ == "__main__":
    main()
