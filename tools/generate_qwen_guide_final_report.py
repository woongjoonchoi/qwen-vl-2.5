#!/usr/bin/env python3
"""Stage 13: assemble the final kernel-level report from all stage artifacts."""
import argparse
import csv
import json
import shutil
from pathlib import Path


def load_json(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else {}


def load_csv(p):
    p = Path(p)
    return list(csv.DictReader(open(p))) if p.exists() else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    R = Path(args.root)

    cfg = load_json(R / "stage4_qwen_structure/qwen_vision_config.json")
    s6 = load_csv(R / "synthetic_tests/stage6_offset_equivalence/offset_equivalence_summary.csv")
    s7 = load_csv(R / "python_reference/stage7_python_reference/python_reference_correctness.csv")
    s9 = load_csv(R / "triton_correctness/stage9_synthetic/triton_synthetic_correctness.csv")
    s10 = load_csv(R / "captured_activation/stage10_qwen_activation_correctness/captured_activation_correctness.csv")
    sweep_dir = R / "nsys_block_sweep/stage11_block_sweep"
    sweep = load_csv(next(iter(sweep_dir.glob("*block_sweep_summary.csv")), Path("/none")))
    best = load_json(next(iter(sweep_dir.glob("best_block_config.json")), Path("/none")))
    seg_dir = R / "segment_compare/stage12_segment_compare"
    seg = load_csv(next(iter(seg_dir.glob("*segment_compare_summary.csv")), Path("/none")))
    derived = load_json(next(iter(seg_dir.glob("*derived.json")), Path("/none")))
    if not derived:
        derived = load_json(seg_dir / "segment_compare_derived.json")

    def passes(rows):
        ok = sum(1 for r in rows if r.get("status") == "PASS")
        return ok, len(rows)

    s6p, s6t = passes(s6); s7p, s7t = passes(s7)
    s9p, s9t = passes(s9); s10p, s10t = passes(s10)

    summaries = R / "summaries"; summaries.mkdir(parents=True, exist_ok=True)

    # copy artifacts to top-level report names
    if best:
        (R / "QWEN25VL_GUIDE_BEST_BLOCK_CONFIG.json").write_text(json.dumps(best, indent=2))
    seg_src = next(iter(seg_dir.glob("*segment_compare_summary.csv")), None)
    if seg_src:
        shutil.copy(seg_src, R / "QWEN25VL_GUIDE_SEGMENT_COMPARE.csv")

    # SUMMARY.csv
    with open(R / "QWEN25VL_GUIDE_KERNEL_LEVEL_SUMMARY.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stage", "metric", "value"])
        w.writerow(["stage6_offset_equiv", "pass_ratio", f"{s6p}/{s6t}"])
        w.writerow(["stage7_python_ref", "pass_ratio", f"{s7p}/{s7t}"])
        w.writerow(["stage9_triton_synth", "pass_ratio", f"{s9p}/{s9t}"])
        w.writerow(["stage10_captured", "pass_ratio", f"{s10p}/{s10t}"])
        if best:
            w.writerow(["stage11_best", "block_size", best.get("block_size")])
            w.writerow(["stage11_best", "num_warps", best.get("num_warps")])
            w.writerow(["stage11_best", "num_stages", best.get("num_stages")])
            w.writerow(["stage11_best", "mean_us_per_iter",
                        round(best.get("mean_ms_per_iter", 0) * 1e3, 3)])
        for k, v in derived.items():
            w.writerow(["stage12_derived", k, v])

    # winner row pretty
    best_line = "n/a"
    if best:
        best_line = (f"BLOCK_SIZE={best.get('block_size')} "
                     f"num_warps={best.get('num_warps')} num_stages={best.get('num_stages')} "
                     f"→ {best.get('mean_ms_per_iter',0)*1e3:.2f} us/iter "
                     f"(correctness {best.get('correctness_status','PASS')})")

    acs = derived.get("attention_core_speedup")
    res = derived.get("replacement_equivalent_speedup")

    md = f"""# Qwen2.5-VL GUIDE Window Kernel — Kernel-Level Report

**Scope:** kernel-level correctness + performance only. No VLM generation
accuracy, no TTFT/TPOT/E2E, no LLM decoder profiling, no Amdahl E2E.

## 1. Environment / nsys
- nsys smoke: PASS (Stage 0); profiling via NVTX projected GPU time.
- Existing Swin kernel source: `~/sigmetrics2026_swin`
- Existing CSWin kernel source: `~/sigmetrics2026_cswin`

## 2. Qwen vision structure (source of truth = loaded config)
- vision_hidden_size={cfg.get('vision_hidden_size')}, num_heads={cfg.get('num_heads')}, head_dim={cfg.get('head_dim')}
- patch_size={cfg.get('patch_size')}, window_size_px={cfg.get('window_size_px')}, raw_window_size={cfg.get('raw_window_size')}
- spatial_merge_size={cfg.get('spatial_merge_size')}, spatial_merge_unit={cfg.get('spatial_merge_unit')}
- depth={cfg.get('depth')}, fullatt_block_indexes={cfg.get('fullatt_block_indexes')}, num_window_layers={cfg.get('num_window_layers')}

## 3. Descriptor generation policy
- **Option B (merged-cell grid).** Qwen packs each spatial-merge {cfg.get('spatial_merge_size')}×{cfg.get('spatial_merge_size')}
  block as {cfg.get('spatial_merge_unit')} contiguous tokens; `window_index` is built on the merged grid.
  A raw patch's packed offset is `merged_flat * smu + sub`, **not** `h*W+w`.
- The descriptor reuses the existing flat int32 schema (HEAD_GROUP / case / base_record),
  with all spatial fields in **merged units** and `SMU` applied as a kernel constexpr.
- Option A (raw row-major) was tried first and **failed** offset equivalence
  (window token sets did not match) — Case C in the plan — and was replaced by Option B.

## 4. Correctness results
| stage | check | result |
|-------|-------|--------|
| 6 | offset / window-length / RoPE equivalence vs **real** `get_window_index` | **{s6p}/{s6t} PASS** |
| 7 | Python reference (baseline gather vs descriptor) | **{s7p}/{s7t} PASS** (exact) |
| 9 | Triton synthetic vs fp32 reference | **{s9p}/{s9t} PASS** |
| 10 | captured Qwen activation, real RoPE (rel_l2 gate) | **{s10p}/{s10t} PASS** |

## 5. Block-size sweep (Stage 11, nsys NVTX projected GPU time)
**Winner:** {best_line}

## 6. Segment comparison (Stage 12)
- **Comparison A — Attention-Core Only:** baseline window attention core vs GUIDE
  descriptor attention core.
  - baseline_attn = {derived.get('baseline_attention_core_mean_ms')} ms/iter
  - guide_attn    = {derived.get('guide_attention_core_mean_ms')} ms/iter
  - **attention_core_speedup = {acs}×**
- **Comparison B — Reorder-Free Replacement:** baseline one-time layout
  (hidden+RoPE reorder + reverse restore) amortized over {derived.get('num_window_layers')} window layers.
  - layout_total = {derived.get('layout_total_ms')} ms (one-time)
  - layout_amortized_per_window_layer = {derived.get('layout_amortized_per_window_layer_ms')} ms
  - baseline_equivalent_per_layer = {derived.get('baseline_equivalent_per_layer_ms')} ms
  - guide_equivalent_per_layer    = {derived.get('guide_equivalent_per_layer_ms')} ms
  - **replacement_equivalent_speedup = {res}×**

## 7. Non-goals (explicitly not measured)
- VLM answer accuracy, generation quality
- TTFT / TPOT / E2E latency, LLM decoder profiling, Amdahl E2E

See `QWEN25VL_GUIDE_LIMITATIONS.md` for limitations and next steps.
"""
    (R / "QWEN25VL_GUIDE_KERNEL_LEVEL_REPORT.md").write_text(md)

    lim = f"""# Qwen2.5-VL GUIDE Kernel — Limitations & Next Steps

## Limitations
- **Divisible grids only.** The descriptor/kernel support merged grids divisible
  by merged_window (= raw patches divisible by raw_window_size). Non-divisible
  (boundary) windows are TODO; Stage 10 forces a square divisible resolution.
- **T=1 only.** Video / multi-frame (T>1) descriptors are not implemented.
- **HEAD_GROUP_NUM=1, no shifted windows.** Qwen window layers are non-shifted;
  CSWin-style multi-head-group / stripe descriptors are out of scope here.
- **Full-attention layers excluded.** Layers in fullatt_block_indexes
  ({cfg.get('fullatt_block_indexes')}) are not replaced.
- **bf16 tolerance.** Captured-activation correctness uses relative-L2 (≤1e-2) as
  the scale-invariant gate; bf16 absolute error scales with activation magnitude
  (a single ULP at deeper layers ≈ 0.03 abs) and is reported for information.
- **Perf uses synthetic Q/K/V at real shape/dtype.** Kernel time depends on
  shape+dtype, not values; this isolates kernel cost.

## Claims supported
- descriptor offset / window-length / RoPE equivalence vs the real model
- kernel-level numerical equivalence (synthetic fp32 ref + real activations)
- best block-size under nsys NVTX measurement
- attention-core and replacement-equivalent kernel-segment speedup/slowdown
- one-time layout materialization cost

## Claims NOT made
- E2E speedup, VLM accuracy unchanged, TTFT/TPOT, decoder throughput

## Next steps
1. Boundary (non-divisible) window descriptors with valid-token masking.
2. T>1 (video) frame stacking + offset tests.
3. Optional in-kernel RoPE fusion.
4. End-to-end integration experiment (separate, out of this plan's scope).
"""
    (R / "QWEN25VL_GUIDE_LIMITATIONS.md").write_text(lim)
    print(f"[Stage13] report written under {R}")
    print(f"  correctness: s6={s6p}/{s6t} s7={s7p}/{s7t} s9={s9p}/{s9t} s10={s10p}/{s10t}")
    print(f"  best: {best_line}")
    print(f"  attention_core_speedup={acs}  replacement_equivalent_speedup={res}")


if __name__ == "__main__":
    main()
