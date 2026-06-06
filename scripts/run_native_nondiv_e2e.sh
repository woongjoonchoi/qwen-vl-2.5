#!/usr/bin/env bash
# Native-size non-divisible grid: FA2 vs GUIDE E2E comparison
#
# Uses original aspect ratio (no square resize).
# Records is_boundary flag per sample; analysis filters to boundary-only subset.
#
# GPU layout: 0/1 = FA2 (shards 0,1), 2/3 = GUIDE (shards 0,1)
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

IMAGE="${IMAGE:-qwen25vl-guide:latest}"
MODEL_PATH="${PROJECT_ROOT}/models/Qwen2.5-VL-3B-Instruct"
MANIFEST="${PROJECT_ROOT}/output/qwen25vl_baseline/manifests/benchmark_manifest_500_docker.jsonl"
FA2_OUT="${PROJECT_ROOT}/output/qwen25vl_native_nondiv/fa2"
GUIDE_OUT="${PROJECT_ROOT}/output/qwen25vl_native_nondiv/guide"
RESULTS_DIR="${PROJECT_ROOT}/results"

c_model="${MODEL_PATH/${PROJECT_ROOT}/\/workspace}"
c_manifest="${MANIFEST/${PROJECT_ROOT}/\/workspace}"
c_fa2="${FA2_OUT/${PROJECT_ROOT}/\/workspace}"
c_guide="${GUIDE_OUT/${PROJECT_ROOT}/\/workspace}"
c_results="${RESULTS_DIR/${PROJECT_ROOT}/\/workspace}"

mkdir -p "${FA2_OUT}" "${GUIDE_OUT}" "${RESULTS_DIR}"

ts()  { date +"%Y-%m-%dT%H:%M:%S"; }
log() { echo "[$(ts)] $1" | tee -a "${FA2_OUT}/../master.log"; }

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
N_SAMPLES="${N_SAMPLES:-50}"   # default 50 for quick run

log "================================================================"
log " Native-size Non-divisible Grid: FA2 vs GUIDE E2E"
log " manifest : ${MANIFEST}"
log " samples  : ${N_SAMPLES}"
log " max_new_tokens : ${MAX_NEW_TOKENS}"
log " GPUs: 0=FA2, 1=GUIDE"
log "================================================================"

run_worker() {
  local gpu="$1" mode="$2" c_out="$3"
  docker run --rm --gpus "device=${gpu}" --ipc=host --ulimit memlock=-1 \
    --user "$(id -u):$(id -g)" \
    -v "${PROJECT_ROOT}:/workspace" \
    -e CUDA_VISIBLE_DEVICES=0 \
    -e PYTHONUNBUFFERED=1 \
    -e HOME=/tmp \
    -e TRITON_CACHE_DIR=/tmp/.triton \
    -e XDG_CACHE_HOME=/tmp/.cache \
    -w /workspace \
    "${IMAGE}" -c "
python3 -u /workspace/tools/run_comparative_inference.py \
  --mode         ${mode} \
  --manifest     '${c_manifest}' \
  --output-dir   '${c_out}' \
  --gpu-id       0 \
  --start-idx    0 \
  --end-idx      ${N_SAMPLES} \
  --max-new-tokens ${MAX_NEW_TOKENS} \
  --model-path   '${c_model}'
" 2>&1 | grep -v "Warning\|deprecat\|fast processor\|Loading checkpoint\|UserWarning"
}

log "[gpu0] FA2   [0,${N_SAMPLES}) START"
( run_worker 0 fa2   "${c_fa2}" | tee "${FA2_OUT}/run.log";
  log "[gpu0] FA2 DONE" ) &  PID0=$!

log "[gpu1] GUIDE [0,${N_SAMPLES}) START"
( run_worker 1 guide "${c_guide}" | tee "${GUIDE_OUT}/run.log";
  log "[gpu1] GUIDE DONE" ) &  PID1=$!

log "Waiting for both workers..."
wait $PID0; E0=$?
wait $PID1; E1=$?
log "All workers done: fa2=$E0 guide=$E1"

# ── Analysis ────────────────────────────────────────────────────────────────────
log "Running analysis..."

python3 - \
  "${FA2_OUT}" "${GUIDE_OUT}" \
  "${RESULTS_DIR}/native_nondiv_fa2_vs_guide_raw.csv" \
  "${RESULTS_DIR}/native_nondiv_fa2_vs_guide_summary.csv" \
<<'PYEOF'
import sys, json, csv, statistics
from pathlib import Path
from collections import defaultdict

fa2_root   = Path(sys.argv[1])
guide_root = Path(sys.argv[2])
raw_csv    = Path(sys.argv[3])
sum_csv    = Path(sys.argv[4])

def load_results(root):
    rows = {}
    for p in sorted(root.rglob("results_gpu*.jsonl")):
        for line in open(p):
            r = json.loads(line)
            sid = r.get("sample_id")
            if sid and r.get("status") == "OK":
                rows[sid] = r
    return rows

fa2   = load_results(fa2_root)
guide = load_results(guide_root)

common = sorted(set(fa2) & set(guide))
print(f"\nLoaded: FA2={len(fa2)} GUIDE={len(guide)} common={len(common)}")

# ── per-sample joined table ──────────────────────────────────────────────────
MERGE_SIZE = 2
MERGED_WINDOW = 4

joined = []
for sid in common:
    f = fa2[sid]; g = guide[sid]
    bm = f.get("benchmark", "?")
    mh = f.get("merged_h", 0) or g.get("merged_h", 0)
    mw = f.get("merged_w", 0) or g.get("merged_w", 0)
    is_boundary = f.get("is_boundary", False) or g.get("is_boundary", False)
    n_vis = f.get("n_visual_tokens", mh * mw)
    tok_match = (f.get("n_out_tokens", -1) == g.get("n_out_tokens", -1))

    fa2_ve    = f.get("ve_ms", -1)
    guide_ve  = g.get("ve_ms", -1)
    fa2_ttft  = f.get("ttft_ms", -1)
    guide_ttft= g.get("ttft_ms", -1)
    fa2_e2e   = f.get("total_gen_ms", -1)
    guide_e2e = g.get("total_gen_ms", -1)
    fa2_tpot  = f.get("tpot_ms", -1)
    guide_tpot= g.get("tpot_ms", -1)

    ve_speedup   = (fa2_ve   / guide_ve)   if guide_ve   > 0 else -1
    ttft_speedup = (fa2_ttft / guide_ttft) if guide_ttft > 0 else -1
    e2e_speedup  = (fa2_e2e  / guide_e2e)  if guide_e2e  > 0 else -1

    joined.append({
        "sample_id": sid, "benchmark": bm,
        "merged_h": mh, "merged_w": mw, "n_visual_tokens": n_vis,
        "is_boundary": is_boundary,
        "fa2_n_out": f.get("n_out_tokens"), "guide_n_out": g.get("n_out_tokens"),
        "tok_match": tok_match,
        "fa2_ve_ms": fa2_ve, "guide_ve_ms": guide_ve, "ve_speedup": round(ve_speedup, 4),
        "fa2_ttft_ms": fa2_ttft, "guide_ttft_ms": guide_ttft, "ttft_speedup": round(ttft_speedup, 4),
        "fa2_e2e_ms": fa2_e2e, "guide_e2e_ms": guide_e2e, "e2e_speedup": round(e2e_speedup, 4),
        "fa2_tpot_ms": fa2_tpot, "guide_tpot_ms": guide_tpot,
        "fa2_anls":        f.get("anls", -1), "guide_anls":        g.get("anls", -1),
        "fa2_relaxed_acc": f.get("relaxed_acc", -1), "guide_relaxed_acc": g.get("relaxed_acc", -1),
    })

# Write raw CSV
with open(raw_csv, "w", newline="") as f:
    if joined:
        w = csv.DictWriter(f, fieldnames=list(joined[0].keys()))
        w.writeheader(); w.writerows(joined)
print(f"Saved raw CSV: {raw_csv} ({len(joined)} rows)")

# ── Summary ──────────────────────────────────────────────────────────────────
def med(vals):
    v = [x for x in vals if x > 0]
    return statistics.median(v) if v else float("nan")

def mean(vals):
    v = [x for x in vals if x > 0]
    return sum(v)/len(v) if v else float("nan")

print()
print("=" * 75)
print("  Native-size: FA2 vs GUIDE — Non-divisible Grid Analysis")
print("=" * 75)

for subset_name, rows in [
    ("ALL native samples", joined),
    ("Boundary (non-divisible) only", [r for r in joined if r["is_boundary"]]),
    ("Divisible (control)",           [r for r in joined if not r["is_boundary"]]),
]:
    if not rows:
        continue
    bnd = sum(1 for r in rows if r["is_boundary"])
    tok_matched = [r for r in rows if r["tok_match"]]
    print(f"\n  [{subset_name}]  n={len(rows)}  boundary={bnd}  tok_matched={len(tok_matched)}")

    for bm_filter, bm_name in [("all","(all bm)"), ("docvqa","docvqa"), ("chartqa","chartqa")]:
        if bm_filter == "all":
            sub = rows
        else:
            sub = [r for r in rows if r["benchmark"] == bm_filter]
        if not sub: continue

        sub_tok = [r for r in sub if r["tok_match"]]

        ve_sp    = med([r["ve_speedup"]   for r in sub])
        ttft_sp  = med([r["ttft_speedup"] for r in sub])
        e2e_sp   = med([r["e2e_speedup"]  for r in (sub_tok if sub_tok else sub)])
        tpot_fa2 = med([r["fa2_tpot_ms"]  for r in sub])
        tpot_gui = med([r["guide_tpot_ms"]for r in sub])
        n_vis_med= med([r["n_visual_tokens"]for r in sub])

        print(f"    {bm_name:<10}  n={len(sub):4d}  "
              f"VE_speedup={ve_sp:.3f}x  TTFT_speedup={ttft_sp:.3f}x  "
              f"E2E_speedup={e2e_sp:.3f}x (tok_match={len(sub_tok)})  "
              f"TPOT_FA2={tpot_fa2:.2f}ms  TPOT_GUIDE={tpot_gui:.2f}ms  "
              f"n_vis_tok={n_vis_med:.0f}")

    # Token mismatch rate
    n_mismatch = sum(1 for r in rows if not r["tok_match"])
    print(f"    token mismatch: {n_mismatch}/{len(rows)} ({100*n_mismatch/max(len(rows),1):.1f}%)")

print()
print("=" * 75)

# ── summary CSV ─────────────────────────────────────────────────────────────
summary_rows = []
for bm in ["docvqa", "chartqa"]:
    for is_bnd, cat in [(True, "boundary"), (False, "divisible")]:
        sub = [r for r in joined if r["benchmark"] == bm and r["is_boundary"] == is_bnd]
        sub_tok = [r for r in sub if r["tok_match"]]
        if not sub: continue
        summary_rows.append({
            "benchmark": bm, "grid_type": cat, "n": len(sub),
            "n_tok_matched": len(sub_tok),
            "med_n_visual_tokens": round(med([r["n_visual_tokens"] for r in sub])),
            "med_fa2_ve_ms":     round(med([r["fa2_ve_ms"]    for r in sub]), 1),
            "med_guide_ve_ms":   round(med([r["guide_ve_ms"]  for r in sub]), 1),
            "med_ve_speedup":    round(med([r["ve_speedup"]   for r in sub]), 4),
            "med_ttft_speedup":  round(med([r["ttft_speedup"] for r in sub]), 4),
            "med_e2e_speedup":   round(med([r["e2e_speedup"]  for r in (sub_tok or sub)]), 4),
            "med_tpot_fa2_ms":   round(med([r["fa2_tpot_ms"]  for r in sub]), 2),
            "med_tpot_guide_ms": round(med([r["guide_tpot_ms"]for r in sub]), 2),
        })

with open(sum_csv, "w", newline="") as f:
    if summary_rows:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader(); w.writerows(summary_rows)
print(f"Saved summary CSV: {sum_csv}")
PYEOF

log "================================================================"
log " DONE. Results:"
log "   ${RESULTS_DIR}/native_nondiv_fa2_vs_guide_raw.csv"
log "   ${RESULTS_DIR}/native_nondiv_fa2_vs_guide_summary.csv"
log "================================================================"
