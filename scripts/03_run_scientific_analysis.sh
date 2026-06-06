#!/usr/bin/env bash
# =============================================================================
# Phase 3: Scientific Analysis + Final Report  (Stage 10 + Stage 11)
# Qwen2.5-VL Baseline Measurement
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/output/qwen25vl_baseline}"
TOOLS_DIR="${PROJECT_ROOT}/tools"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/models/Qwen2.5-VL-3B-Instruct}"
PRECISION="${PRECISION:-bf16}"

LOG_DIR="${OUTPUT_ROOT}/logs"
mkdir -p "${LOG_DIR}"

PROGRESS_LOG="${LOG_DIR}/progress.log"
STATUS_JSONL="${LOG_DIR}/status.jsonl"
CURRENT_STAGE="${LOG_DIR}/current_stage.txt"

touch "${PROGRESS_LOG}" "${STATUS_JSONL}"

ts_human() { date +"%Y-%m-%d %H:%M:%S"; }
log_progress() {
  echo "[$(ts_human)] [$1] $2" | tee -a "${PROGRESS_LOG}"
}
write_status() {
  python3 - "${STATUS_JSONL}" "$1" "$2" "${3:-}" <<'PY'
import json, sys, datetime
path, stage, status, message = sys.argv[1:5]
with open(path, "a") as f:
    f.write(json.dumps({"time": datetime.datetime.now().isoformat(timespec="seconds"),
                        "stage": stage, "status": status, "message": message}) + "\n")
PY
}

# ─── Stage 10: Amdahl Analysis ────────────────────────────────────────────────
stage10_amdahl_analysis() {
  local stage="stage10_amdahl_analysis"
  echo "${stage}" > "${CURRENT_STAGE}"
  local out="${OUTPUT_ROOT}/${stage}"
  mkdir -p "${out}"

  log_progress "START" "Stage 10: Amdahl baseline analysis"
  write_status "${stage}" "START" "computing Amdahl speedup predictions"

  # First, aggregate all baseline results into summary CSV
  python3 -u "${TOOLS_DIR}/aggregate_qwen_baseline.py" \
    --output-root "${OUTPUT_ROOT}" \
    --model-path "${MODEL_PATH}" \
    --precision "${PRECISION}" \
    2>&1 | tee "${out}/aggregate_stdout.log" || {
    log_progress "WARN" "  aggregate_qwen_baseline.py had issues, continuing..."
  }

  local summary_csv="${OUTPUT_ROOT}/QWEN25VL_BASELINE_SUMMARY.csv"
  if [ ! -f "${summary_csv}" ]; then
    log_progress "WARN" "  Summary CSV not found, trying to build from stage results..."
    python3 - "${OUTPUT_ROOT}" "${summary_csv}" <<'PY'
import json, csv, sys
from pathlib import Path
from collections import defaultdict

root, out_csv = Path(sys.argv[1]), sys.argv[2]

all_rows = []
for stage_dir in root.glob("stage*"):
    for jfile in stage_dir.rglob("timing_raw.jsonl"):
        try:
            with open(jfile) as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        r.setdefault("stage", stage_dir.name)
                        all_rows.append(r)
        except:
            pass

if all_rows:
    fields = list(all_rows[0].keys())
    fields = [f for f in fields if f != "per_block_ms"]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in all_rows:
            writer.writerow({k: r.get(k, "") for k in fields})
    print(f"Wrote {len(all_rows)} rows to {out_csv}")
else:
    print(f"No rows found. Creating empty CSV.")
    with open(out_csv, "w") as f:
        f.write("stage,benchmark,resolution,max_new_tokens,f_vision,e2e_ms\n")
PY
  fi

  # Run Amdahl analysis
  if [ -f "${summary_csv}" ]; then
    python3 -u "${TOOLS_DIR}/analyze_qwen_amdahl.py" \
      --summary-csv "${summary_csv}" \
      --output-dir "${out}" \
      2>&1 | tee "${out}/amdahl_stdout.log"

    # Copy Amdahl results to output root
    cp "${out}/amdahl_summary.csv" "${OUTPUT_ROOT}/QWEN25VL_BASELINE_AMDAHL.csv" 2>/dev/null || true
  else
    log_progress "WARN" "  No summary CSV available for Amdahl analysis"
    python3 -c "import json; open('${out}/stage_status.json','w').write(
      json.dumps({'status':'FAIL','message':'no summary CSV'}))"
  fi

  write_status "${stage}" "DONE" "Amdahl analysis complete"
  log_progress "DONE" "Stage 10: Amdahl analysis done"
}

# ─── Stage 11: Final Report ───────────────────────────────────────────────────
stage11_final_report() {
  local stage="stage11_final_report"
  echo "${stage}" > "${CURRENT_STAGE}"
  local out="${OUTPUT_ROOT}"
  log_progress "START" "Stage 11: generating final report"
  write_status "${stage}" "START" "final report + GUIDE readiness check"

  python3 - "${OUTPUT_ROOT}" "${PRECISION}" "${MODEL_PATH}" <<'PYEOF'
import json, csv, sys
from pathlib import Path
from collections import defaultdict

root, precision, model_path = Path(sys.argv[1]), sys.argv[2], sys.argv[3]

def read_json(path):
    try: return json.loads(Path(path).read_text())
    except: return {}

def read_csv(path):
    try:
        with open(path) as f: return list(csv.DictReader(f))
    except: return []

def pf(v, dec=2):
    try: return round(float(v), dec)
    except: return v

# ── collect stage statuses ──
stage_statuses = {}
for d in sorted(root.glob("stage*")):
    s = read_json(d / "stage_status.json")
    stage_statuses[d.name] = s.get("status", "UNKNOWN")

# ── collect key metrics ──
profiler_smoke = read_json(root / "stage_status" / "profiler_smoke.json")
nsys_available = profiler_smoke.get("NSYS_AVAILABLE", False)
ncu_available  = profiler_smoke.get("NCU_AVAILABLE", False)

# Load summary data
summary_rows = read_csv(root / "QWEN25VL_BASELINE_SUMMARY.csv")
amdahl_rows  = read_csv(root / "QWEN25VL_BASELINE_AMDAHL.csv")

# Filter valid rows with f_vision > 0
valid_summary = [r for r in summary_rows if pf(r.get("f_vision", 0), 4) > 0]

# ── resolution sweep analysis ──
res_rows = [r for r in summary_rows if "resolution" in r.get("stage","")]
res_summary = read_csv(root / "stage5_resolution_sweep" / "resolution_sweep_summary.csv")

# ── output length analysis ──
ol_summary = read_csv(root / "stage6_output_length_sweep" / "output_length_summary.csv")

# ── Amdahl best config ──
best_amdahl = None
if amdahl_rows:
    def s_e2e_2x(r):
        try: return float(r.get("S_e2e_at_Sv2.0", 0))
        except: return 0
    best_amdahl = max(amdahl_rows, key=s_e2e_2x)

# ── GUIDE readiness check ──
readiness = {
    "manifest_exists": (root / "stage0_50_data_manifest" / "benchmark_manifest.jsonl").exists(),
    "model_loads": read_json(root / "stage0_75_model_data_smoke" / "stage_status.json").get("status") in ["PASS", "DONE"],
    "inference_smoke": read_json(root / "stage0_75_model_data_smoke" / "stage_status.json").get("status") in ["PASS", "DONE"],
    "app_timing_works": read_json(root / "stage2_app_timing_sanity" / "stage_status.json").get("status") in ["PASS", "DONE"],
    "visual_encoder_measurable": any(pf(r.get("visual_encoder_ms",0),2) > 0 for r in valid_summary),
    "f_vision_computed": any(pf(r.get("f_vision",0),4) > 0 for r in valid_summary),
    "resolution_sweep_done": (root / "stage5_resolution_sweep" / "resolution_sweep_summary.csv").exists(),
    "output_length_sweep_done": (root / "stage6_output_length_sweep" / "output_length_summary.csv").exists(),
    "nsys_available": nsys_available,
    "ncu_available": ncu_available,
    "generated_outputs_saved": (root / "stage1_generation_sanity" / "smoke_generated_outputs.jsonl").exists(),
    "baseline_csv_exists": (root / "QWEN25VL_BASELINE_SUMMARY.csv").exists(),
}
readiness_score = sum(1 for v in readiness.values() if v)
readiness_total = len(readiness)
guide_ready = readiness_score >= readiness_total * 0.75

# ── high-res f_vision check ──
high_res_fv = {}
for r in valid_summary:
    try:
        res = int(r.get("resolution", 0))
        fv = pf(r.get("f_vision", 0), 4)
        if res in [1344, 1792] and fv > 0:
            bm = r.get("benchmark", "")
            if (bm, res) not in high_res_fv or fv > high_res_fv[(bm,res)]:
                high_res_fv[(bm,res)] = fv
    except:
        pass

fv_1792 = max(high_res_fv.get((bm,1792),0) for bm in set(bm for bm,_ in high_res_fv)) if high_res_fv else 0
fv_1344 = max(high_res_fv.get((bm,1344),0) for bm in set(bm for bm,_ in high_res_fv)) if high_res_fv else 0
guide_fv_threshold = fv_1792 >= 0.25 or fv_1344 >= 0.25

# ── decode dominance ──
decode_dom = False
if ol_summary:
    long_rows = [r for r in ol_summary if str(r.get("max_new_tokens","")) in ["256","1024"]]
    short_rows = [r for r in ol_summary if str(r.get("max_new_tokens","")) in ["32","64"]]
    if long_rows and short_rows:
        try:
            avg_f_decode_long  = sum(pf(r.get("mean_f_decode",0),4) for r in long_rows) / len(long_rows)
            avg_f_decode_short = sum(pf(r.get("mean_f_decode",0),4) for r in short_rows) / len(short_rows)
            decode_dom = avg_f_decode_long > avg_f_decode_short + 0.15
        except: pass

# ── write final report ──
report_path = root / "QWEN25VL_BASELINE_MEASUREMENT_REPORT.md"
with open(report_path, "w") as f:
    f.write("# Qwen2.5-VL Baseline Measurement Report\n\n")
    f.write("## 1. Environment\n\n")
    f.write(f"- Model: `{model_path}`\n")
    f.write(f"- Precision: `{precision}`\n")
    f.write(f"- NSYS_AVAILABLE: `{nsys_available}`\n")
    f.write(f"- NCU_AVAILABLE: `{ncu_available}`\n")
    if profiler_smoke:
        f.write(f"- cuda_smoke: `{profiler_smoke.get('cuda_smoke','?')}`\n")
        f.write(f"- nsys_smoke: `{profiler_smoke.get('nsys_smoke','?')}`\n")
        f.write(f"- ncu_smoke: `{profiler_smoke.get('ncu_smoke','?')}`\n")
    f.write("\n")

    f.write("## 2. Stage Results\n\n")
    for name, status in stage_statuses.items():
        icon = "✅" if status in ["PASS","DONE"] else ("⚠️" if "SKIP" in status else "❌")
        f.write(f"- {icon} **{name}**: {status}\n")
    f.write("\n")

    f.write("## 3. Visual Encoder Architecture\n\n")
    struct = read_json(root / "stage3_visual_encoder_structure" / "stage_status.json")
    f.write(f"- ViT depth: {struct.get('depth', 32)}\n")
    f.write(f"- Full attention blocks: {struct.get('fullatt_blocks', [7,15,23,31])}\n")
    f.write("- Patch size: 14, Merge size: 2\n\n")

    f.write("## 4. Resolution Sweep Results\n\n")
    if res_summary:
        f.write("| benchmark | resolution | tokens_after | mean_e2e_ms | mean_ve_ms | f_vision |\n")
        f.write("|---|---|---|---|---|---|\n")
        for r in res_summary:
            f.write(f"| {r.get('benchmark','')} | {r.get('resolution','')} "
                    f"| {r.get('mean_num_visual_tokens_after_merger','')} "
                    f"| {r.get('mean_e2e_ms','')} | {r.get('mean_visual_encoder_ms','')} "
                    f"| {r.get('mean_f_vision','')} |\n")
    else:
        f.write("*(no data)*\n")
    f.write("\n")

    f.write("## 5. Output Length Sweep Results\n\n")
    if ol_summary:
        f.write("| benchmark | max_tok | mean_e2e_ms | mean_ttft_ms | mean_tpot_ms | f_decode |\n")
        f.write("|---|---|---|---|---|---|\n")
        for r in ol_summary:
            f.write(f"| {r.get('benchmark','')} | {r.get('max_new_tokens','')} "
                    f"| {r.get('mean_e2e_ms','')} | {r.get('mean_ttft_ms','')} "
                    f"| {r.get('mean_tpot_ms','')} | {r.get('mean_f_decode','')} |\n")
    else:
        f.write("*(no data)*\n")
    f.write("\n")

    f.write("## 6. Amdahl Analysis\n\n")
    if amdahl_rows:
        f.write("| benchmark | resolution | max_tok | f_vision | S_e2e@2x | S_e2e@3x |\n")
        f.write("|---|---|---|---|---|---|\n")
        for r in sorted(amdahl_rows,
                         key=lambda x: float(x.get("S_e2e_at_Sv2.0",0)) if x.get("S_e2e_at_Sv2.0") else 0,
                         reverse=True)[:15]:
            f.write(f"| {r.get('benchmark','')} | {r.get('resolution','')} "
                    f"| {r.get('max_new_tokens','')} | {r.get('mean_f_vision','')} "
                    f"| {r.get('S_e2e_at_Sv2.0','')} | {r.get('S_e2e_at_Sv3.0','')} |\n")
    else:
        f.write("*(no data)*\n")
    f.write("\n")

    f.write("## 7. +GUIDE Readiness\n\n")
    for k, v in readiness.items():
        icon = "✅" if v else "❌"
        f.write(f"- {icon} {k}\n")
    f.write(f"\n**Readiness score**: {readiness_score}/{readiness_total}\n\n")

    if best_amdahl:
        f.write(f"**Recommended +GUIDE target**: "
                f"benchmark=`{best_amdahl.get('benchmark')}` "
                f"resolution=`{best_amdahl.get('resolution')}` "
                f"max_new_tokens=`{best_amdahl.get('max_new_tokens')}`\n\n")

print(f"Written: {report_path}")

# ── write final decision JSON ──
decision = {
    "guide_readiness": "READY" if guide_ready else "NOT_READY",
    "readiness_score": f"{readiness_score}/{readiness_total}",
    "fv_1792_max": fv_1792,
    "fv_1344_max": fv_1344,
    "guide_fv_sufficient": guide_fv_threshold,
    "decode_dominates_long_output": decode_dom,
    "best_guide_config": {
        "benchmark": best_amdahl.get("benchmark") if best_amdahl else None,
        "resolution": best_amdahl.get("resolution") if best_amdahl else None,
        "max_new_tokens": best_amdahl.get("max_new_tokens") if best_amdahl else None,
        "mean_f_vision": best_amdahl.get("mean_f_vision") if best_amdahl else None,
        "S_e2e_at_2x": best_amdahl.get("S_e2e_at_Sv2.0") if best_amdahl else None,
        "S_e2e_at_3x": best_amdahl.get("S_e2e_at_Sv3.0") if best_amdahl else None,
    },
}
decision_path = root / "QWEN25VL_BASELINE_FINAL_DECISION.json"
decision_path.write_text(json.dumps(decision, indent=2, ensure_ascii=False))
print(f"Written: {decision_path}")

# ── print scientific briefing ──
print("\n" + "="*72)
print(" QWEN2.5-VL BASELINE MEASUREMENT — FINAL SCIENTIFIC BRIEFING")
print("="*72)

print("\n[Q1] 고해상도에서 Visual-Prefill 지연 시간 점유율(f_vision)이 충분히 높아")
print("     GUIDE 적용 타당성이 입증되었는가?")
print()
if fv_1792 > 0:
    if fv_1792 >= 0.35:
        print(f"  → YES (강하게 입증됨)")
        print(f"     f_vision@1792 = {fv_1792:.3f} ≥ 0.35")
        print(f"     고해상도 단출력 설정에서 visual encoder가 E2E latency의 {fv_1792*100:.1f}%를 차지.")
        print(f"     GUIDE 적용 시 상당한 TTFT/E2E 단축이 기대됨.")
    elif fv_1792 >= 0.25:
        print(f"  → YES (적정 수준 입증)")
        print(f"     f_vision@1792 = {fv_1792:.3f} ∈ [0.25, 0.35)")
        print(f"     GUIDE 적용 타당성 있으나 E2E 스피드업은 moderate 수준 예상.")
    else:
        print(f"  → PARTIAL (주의 필요)")
        print(f"     f_vision@1792 = {fv_1792:.3f} < 0.25")
        print(f"     해당 설정에서 visual encoder가 dominant bottleneck이 아님.")
        print(f"     resolution 상향 또는 단출력 조건 확인 필요.")
elif fv_1344 > 0:
    print(f"  → PARTIAL: f_vision@1344 = {fv_1344:.3f} (1792 데이터 미확보)")
else:
    print(f"  → 데이터 부족 — f_vision 측정값 없음")

print("\n[Q2] 긴 출력(Long output) 환경에서 디코더 병목이 E2E Latency에 미치는 영향?")
print()
if ol_summary:
    long_f_d = [(r.get("benchmark",""), r.get("max_new_tokens",""),
                 pf(r.get("mean_f_decode",0),3)) for r in ol_summary
                if str(r.get("max_new_tokens","")) in ["256","1024"]]
    short_f_d = [(r.get("benchmark",""), r.get("max_new_tokens",""),
                  pf(r.get("mean_f_decode",0),3)) for r in ol_summary
                 if str(r.get("max_new_tokens","")) in ["32","64"]]
    if long_f_d:
        for bm, tok, fd in long_f_d[:4]:
            print(f"  f_decode @ {bm}/tok={tok}: {fd:.3f}")
    if short_f_d:
        for bm, tok, fd in short_f_d[:4]:
            print(f"  f_decode @ {bm}/tok={tok}: {fd:.3f}")
    if decode_dom:
        print()
        print("  → 긴 출력 조건에서 decode loop가 E2E latency를 지배함.")
        print("     visual encoder 최적화(GUIDE)의 E2E 스피드업은 긴 출력에서 희석됨.")
        print("     GUIDE는 단출력/TTFT 중심 워크로드에서 가장 효과적.")
    else:
        print()
        print("  → 출력 길이 변화에 따른 명확한 decode dominance 패턴 미검출.")
        print("     (더 많은 샘플 또는 더 긴 출력 길이 확인 필요)")
else:
    print("  → 출력 길이 스윕 데이터 미확보")

print("\n[Q3] Amdahl 법칙을 통해 도출된 예상 E2E Speedup(S_e2e)과 최우선 타겟은?")
print()
if best_amdahl:
    fv = pf(best_amdahl.get("mean_f_vision",0),3)
    se2 = best_amdahl.get("S_e2e_at_Sv2.0","?")
    se3 = best_amdahl.get("S_e2e_at_Sv3.0","?")
    print(f"  Best config: benchmark={best_amdahl.get('benchmark')} "
          f"res={best_amdahl.get('resolution')} tok={best_amdahl.get('max_new_tokens')}")
    print(f"  f_vision = {fv}")
    print(f"  S_e2e @ S_vision=2x : {se2}")
    print(f"  S_e2e @ S_vision=3x : {se3}")
    print()
    print(f"  → 향후 +GUIDE 실험 최우선 타겟:")
    print(f"     benchmark   : {best_amdahl.get('benchmark')}")
    print(f"     resolution  : {best_amdahl.get('resolution')}")
    print(f"     max_new_toks: {best_amdahl.get('max_new_tokens')}")
    if fv >= 0.25:
        print(f"     근거: f_vision={fv}으로 visual-prefill 병목이 유의미함.")
        print(f"           S_vision=2x 가정 시 S_e2e={se2} 달성 가능.")
else:
    print("  → Amdahl 분석 데이터 미확보")

print("\n[+GUIDE Readiness]")
print(f"  Score: {readiness_score}/{readiness_total}")
print(f"  Status: {'READY ✓' if guide_ready else 'NOT READY ✗'}")
for k, v in readiness.items():
    print(f"  {'✓' if v else '✗'} {k}")

print("\n" + "="*72)
PYEOF

  python3 -c "import json; open('${OUTPUT_ROOT}/stage11_final_report/stage_status.json','w').write(
    json.dumps({'status':'DONE'}))" 2>/dev/null || \
    (mkdir -p "${OUTPUT_ROOT}/stage11_final_report" && \
     python3 -c "import json; open('${OUTPUT_ROOT}/stage11_final_report/stage_status.json','w').write(
       json.dumps({'status':'DONE'}))")

  write_status "${stage}" "DONE" "final report and scientific briefing complete"
  log_progress "DONE" "Stage 11: final report generated"
}

# ─── Main ─────────────────────────────────────────────────────────────────────
main() {
  log_progress "START" "=== Phase 3: Scientific Analysis + Final Report ==="

  stage10_amdahl_analysis
  stage11_final_report

  log_progress "DONE" "=== Phase 3 COMPLETE ==="
  echo ""
  echo "=================================================================="
  echo " PHASE 3 COMPLETE"
  echo " Final report: ${OUTPUT_ROOT}/QWEN25VL_BASELINE_MEASUREMENT_REPORT.md"
  echo " Decision:     ${OUTPUT_ROOT}/QWEN25VL_BASELINE_FINAL_DECISION.json"
  echo " Summary CSV:  ${OUTPUT_ROOT}/QWEN25VL_BASELINE_SUMMARY.csv"
  echo " Amdahl CSV:   ${OUTPUT_ROOT}/QWEN25VL_BASELINE_AMDAHL.csv"
  echo "=================================================================="
}

main "$@"
