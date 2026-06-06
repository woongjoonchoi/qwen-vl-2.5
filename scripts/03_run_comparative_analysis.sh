#!/usr/bin/env bash
# Phase 3: FA2 vs GUIDE — Scientific Analysis & Report
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CSV="${PROJECT_ROOT}/results/fa2_vs_guide_500dp_tok64.csv"
AMDAHL_CSV="${PROJECT_ROOT}/output/qwen25vl_baseline/stage10_amdahl_analysis/amdahl_summary.csv"
OUT="${PROJECT_ROOT}/results"
mkdir -p "${OUT}"

python3 - "${CSV}" "${AMDAHL_CSV}" "${OUT}" <<'PY'
import sys, csv, json, math
from pathlib import Path
from collections import defaultdict

csv_path   = Path(sys.argv[1])
amdahl_csv = Path(sys.argv[2])
out        = Path(sys.argv[3])

rows = list(csv.DictReader(open(csv_path)))
print(f"Loaded {len(rows)} rows from {csv_path.name}")

def flt(v):
    try: return float(v)
    except: return None

def mean(vals):
    v = [x for x in vals if x is not None and x > 0]
    return sum(v)/len(v) if v else None

def pct(a, b):
    if a and b and b > 0: return (a - b) / b * 100
    return None

# ── Group by (attn_impl, benchmark, resolution) ──────────────────────────────
groups = defaultdict(list)
for r in rows:
    key = (r['attn_impl'], r['benchmark'], int(r['resolution']))
    groups[key].append(r)

metrics = ['visual_encoder_ms', 'ttft_ms', 'tpot_ms', 'e2e_ms',
           'llm_prefill_ms', 'decode_loop_ms', 'f_vision']

# ── Summary table ─────────────────────────────────────────────────────────────
summary = {}
for key, rlist in groups.items():
    attn, bm, res = key
    d = {m: mean([flt(r.get(m)) for r in rlist]) for m in metrics}
    d['n'] = len(rlist)
    summary[key] = d

# ── Build comparison table ────────────────────────────────────────────────────
benchmarks = ['docvqa', 'chartqa']
resolutions = [448, 896, 1344, 1792]

print("\n" + "="*90)
print(f"  FA2 vs GUIDE — 500-Sample Comparison (max_new_tokens=64, bf16, H200)")
print("="*90)
print(f"  {'benchmark':<10} {'res':>5}  {'VE_FA2':>8} {'VE_GUI':>8} {'VE_spd':>7}  "
      f"{'TTFT_FA2':>9} {'TTFT_GUI':>9} {'TTFT_spd':>8}  "
      f"{'f_vis_FA2':>10} {'f_vis_GUI':>10}")
print("  " + "-"*86)

comparison_rows = []
for bm in benchmarks:
    for res in resolutions:
        fa2  = summary.get(('fa2',   bm, res), {})
        guide= summary.get(('guide', bm, res), {})
        if not fa2 or not guide:
            continue
        ve_fa2  = fa2['visual_encoder_ms']
        ve_gui  = guide['visual_encoder_ms']
        ttft_fa2= fa2['ttft_ms']
        ttft_gui= guide['ttft_ms']
        e2e_fa2 = fa2['e2e_ms']
        e2e_gui = guide['e2e_ms']
        fv_fa2  = fa2['f_vision']
        fv_gui  = guide['f_vision']
        ve_spd  = ve_fa2/ve_gui if ve_fa2 and ve_gui else None
        ttft_spd= ttft_fa2/ttft_gui if ttft_fa2 and ttft_gui else None
        e2e_spd = e2e_fa2/e2e_gui if e2e_fa2 and e2e_gui else None
        print(f"  {bm:<10} {res:>5}  {ve_fa2:>8.1f} {ve_gui:>8.1f} {ve_spd:>6.3f}x  "
              f"{ttft_fa2:>9.1f} {ttft_gui:>9.1f} {ttft_spd:>7.3f}x  "
              f"{fv_fa2:>10.4f} {fv_gui:>10.4f}")
        comparison_rows.append({
            'benchmark': bm, 'resolution': res,
            'fa2_ve_ms': round(ve_fa2,3), 'guide_ve_ms': round(ve_gui,3),
            've_speedup': round(ve_spd,4),
            'fa2_ttft_ms': round(ttft_fa2,3), 'guide_ttft_ms': round(ttft_gui,3),
            'ttft_speedup': round(ttft_spd,4),
            'fa2_e2e_ms': round(e2e_fa2,3), 'guide_e2e_ms': round(e2e_gui,3),
            'e2e_speedup': round(e2e_spd,4),
            'fa2_f_vision': round(fv_fa2,6), 'guide_f_vision': round(fv_gui,6),
            'fa2_tpot_ms': round(fa2['tpot_ms'],3), 'guide_tpot_ms': round(guide['tpot_ms'],3),
            'fa2_n': fa2['n'], 'guide_n': guide['n'],
        })
    print()

# ── Workload Q1: DocVQA vs ChartQA difference ─────────────────────────────────
print("="*90)
print("  Q2. DocVQA vs ChartQA — GUIDE E2E speedup difference")
print("="*90)
for res in resolutions:
    doc_spd = next((r['e2e_speedup'] for r in comparison_rows if r['benchmark']=='docvqa' and r['resolution']==res), None)
    cht_spd = next((r['e2e_speedup'] for r in comparison_rows if r['benchmark']=='chartqa' and r['resolution']==res), None)
    if doc_spd and cht_spd:
        print(f"  res={res:4d}: docvqa={doc_spd:.4f}x  chartqa={cht_spd:.4f}x  "
              f"diff={abs(doc_spd-cht_spd):.4f}  "
              f"{'significant' if abs(doc_spd-cht_spd)>0.02 else 'not significant'}")

# ── Amdahl comparison (Q3) ────────────────────────────────────────────────────
print()
print("="*90)
print("  Q3. Amdahl Prediction vs Measured E2E Speedup")
print("="*90)

# Load old Amdahl predictions (baseline SDPA f_vision values)
amdahl = {}
if amdahl_csv.exists():
    for r in csv.DictReader(open(amdahl_csv)):
        bm = r['benchmark'].strip() if r.get('benchmark') else ''
        res_str = r.get('resolution','').strip()
        if not bm or not res_str: continue
        try:
            res = int(float(res_str)) if res_str else 0
        except: continue
        amdahl[(bm, res)] = r

print(f"  {'benchmark':<10} {'res':>5}  {'f_vis_fa2':>10}  {'pred_1.05x':>11}  {'pred_1.09x':>11}  {'actual_e2e':>11}  {'actual_ve':>10}")
print("  " + "-"*80)
for r in comparison_rows:
    bm, res = r['benchmark'], r['resolution']
    fv = r['fa2_f_vision']
    # Amdahl prediction: S_e2e = 1 / (1 - fv + fv/Sv)
    # For Sv=1.05x (GUIDE VE speedup ~5%), Sv=1.09x (from Stage12)
    def amdahl_pred(fv, Sv):
        if fv <= 0 or Sv <= 0: return None
        return 1.0 / (1 - fv + fv/Sv)
    pred_05 = amdahl_pred(fv, 1.05)
    pred_09 = amdahl_pred(fv, r['ve_speedup'])  # using actual VE speedup
    actual  = r['e2e_speedup']
    actual_ve = r['ve_speedup']
    print(f"  {bm:<10} {res:>5}  {fv:>10.4f}  {pred_05:>11.4f}  {pred_09:>11.4f}  {actual:>11.4f}  {actual_ve:>10.4f}")

# ── Save CSVs ──────────────────────────────────────────────────────────────────
comp_csv = out / 'fa2_vs_guide_comparison_summary.csv'
with open(comp_csv, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(comparison_rows[0].keys()))
    w.writeheader(); w.writerows(comparison_rows)

# Per-sample CSV already exists as fa2_vs_guide_500dp_tok64.csv
print(f"\nSaved: {comp_csv}")

# ── Final briefing ─────────────────────────────────────────────────────────────
print()
print("="*90)
print("  SCIENTIFIC CONCLUSIONS")
print("="*90)

# Q1: VE and TTFT speedup (1792 docvqa as representative)
rep = next((r for r in comparison_rows if r['benchmark']=='docvqa' and r['resolution']==1792), None)
if rep:
    print(f"\nQ1. FA2 vs GUIDE @ docvqa res=1792 (GUIDE 적용 대표 조건):")
    print(f"  Visual Encoder: FA2={rep['fa2_ve_ms']:.1f}ms → GUIDE={rep['guide_ve_ms']:.1f}ms  "
          f"speedup={rep['ve_speedup']:.3f}x")
    print(f"  TTFT:           FA2={rep['fa2_ttft_ms']:.1f}ms → GUIDE={rep['guide_ttft_ms']:.1f}ms  "
          f"speedup={rep['ttft_speedup']:.3f}x")
    print(f"  E2E:            FA2={rep['fa2_e2e_ms']:.1f}ms → GUIDE={rep['guide_e2e_ms']:.1f}ms  "
          f"speedup={rep['e2e_speedup']:.3f}x")
    print(f"  f_vision (FA2): {rep['fa2_f_vision']:.4f}")

print(f"\nQ2. DocVQA vs ChartQA GUIDE E2E speedup 차이:")
for res in [896, 1792]:
    doc_spd = next((r['e2e_speedup'] for r in comparison_rows if r['benchmark']=='docvqa' and r['resolution']==res), None)
    cht_spd = next((r['e2e_speedup'] for r in comparison_rows if r['benchmark']=='chartqa' and r['resolution']==res), None)
    if doc_spd and cht_spd:
        diff = abs(doc_spd-cht_spd)
        sig = "유의미 (>2%)" if diff > 0.02 else "비유의미 (<2%)"
        print(f"  res={res}: docvqa={doc_spd:.4f}x  chartqa={cht_spd:.4f}x  diff={diff:.4f}  → {sig}")

if rep:
    fv = rep['fa2_f_vision']
    actual_ve_spd = rep['ve_speedup']
    pred_amdahl = 1.0 / (1 - fv + fv / actual_ve_spd) if actual_ve_spd > 0 else None
    baseline_pred_09 = 1.0 / (1 - fv + fv / 1.09)
    print(f"\nQ3. Amdahl 예측 vs 실제 (docvqa res=1792):")
    print(f"  f_vision (FA2)              = {fv:.4f}")
    print(f"  Stage12 kernel VE speedup   = 1.09x  → 예측 S_e2e = {baseline_pred_09:.4f}x")
    print(f"  실측 VE speedup             = {actual_ve_spd:.4f}x  → Amdahl 예측 = {pred_amdahl:.4f}x")
    print(f"  실측 E2E speedup            = {rep['e2e_speedup']:.4f}x")
    gap = rep['e2e_speedup'] - (pred_amdahl or 0)
    print(f"  예측 vs 실측 gap            = {gap:+.4f}x")
    if abs(gap) < 0.02:
        print(f"  → 예측과 실측이 2% 이내로 일치 ✓")
    elif gap > 0:
        print(f"  → 실측이 예측보다 높음: GUIDE가 VE 외 다른 bottleneck도 개선")
    else:
        print(f"  → 실측이 예측보다 낮음: overhead가 존재하거나 decode가 병목")
PY

echo
echo "=================================================================="
echo " Phase 3 Analysis COMPLETE"
echo " Summary CSV: ${OUT}/fa2_vs_guide_comparison_summary.csv"
echo "=================================================================="
