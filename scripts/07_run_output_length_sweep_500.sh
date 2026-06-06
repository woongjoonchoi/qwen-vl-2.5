#!/usr/bin/env bash
# =============================================================================
# Output Length Sweep — 500 data points per (benchmark × max_new_tokens)
# res=1792 고정
# GPU0: docvqa tok=32,64  GPU1: docvqa tok=256,1024
# GPU2: chartqa tok=32,64  GPU3: chartqa tok=256,1024
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/models/Qwen2.5-VL-3B-Instruct}"
MANIFEST="${PROJECT_ROOT}/output/qwen25vl_baseline/manifests/benchmark_manifest_500_docker.jsonl"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/output/qwen25vl_sweep_outlength_500}"
IMAGE_NAME="${IMAGE_NAME:-qwen25vl-baseline:latest}"
PRECISION="${PRECISION:-bf16}"
SEED="${SEED:-42}"
WARMUP="${WARMUP:-2}"
MEASURED_ITERS="${MEASURED_ITERS:-500}"
RESOLUTION=1792

mkdir -p "${OUTPUT_ROOT}/logs"
LOG_DIR="${OUTPUT_ROOT}/logs"
ts() { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(ts)] $1" | tee -a "${LOG_DIR}/sweep.log"; }

log "=== Output Length Sweep (500 dp/config, res=${RESOLUTION}) ==="
log "Configs: docvqa/chartqa × tok(32,64,256,1024)"

run_config() {
  local gpu_id="$1" benchmark="$2" max_new_tokens="$3"
  local job="${benchmark}_tok${max_new_tokens}"
  local log_file="${LOG_DIR}/gpu${gpu_id}_${job}.log"
  local out_dir="${OUTPUT_ROOT}/${job}"
  local c_model="${MODEL_PATH/${PROJECT_ROOT}/\/workspace}"
  local c_manifest="${MANIFEST/${PROJECT_ROOT}/\/workspace}"
  local c_out="${out_dir/${PROJECT_ROOT}/\/workspace}"
  mkdir -p "${out_dir}"
  log "  [GPU${gpu_id}] ${job}"
  docker run --rm --gpus all \
    --user "$(id -u):$(id -g)" --ipc=host --ulimit memlock=-1 \
    -v "${PROJECT_ROOT}:/workspace" \
    -e "CUDA_VISIBLE_DEVICES=${gpu_id}" -e "PYTHONUNBUFFERED=1" \
    "${IMAGE_NAME}" \
    -c "python3 -u /workspace/tools/profile_qwen25vl_baseline.py \
      --mode app_timing --model-path ${c_model} --manifest ${c_manifest} \
      --benchmark ${benchmark} --num-samples 500 \
      --resolution ${RESOLUTION} --max-new-tokens ${max_new_tokens} \
      --batch-size 1 --precision ${PRECISION} \
      --warmup ${WARMUP} --measured-iters ${MEASURED_ITERS} \
      --seed ${SEED} --output-dir ${c_out}" \
    2>&1 | tee "${log_file}"
}

( run_config 0 docvqa  32  && run_config 0 docvqa  64  ) & PID0=$!
( run_config 1 docvqa  256 && run_config 1 docvqa  1024) & PID1=$!
( run_config 2 chartqa 32  && run_config 2 chartqa 64  ) & PID2=$!
( run_config 3 chartqa 256 && run_config 3 chartqa 1024) & PID3=$!

log "4 GPU groups launched. Waiting..."
FAILED=0
wait $PID0; RC=$?; log "  GPU0 done (rc=${RC})"; [ $RC -ne 0 ] && FAILED=1
wait $PID1; RC=$?; log "  GPU1 done (rc=${RC})"; [ $RC -ne 0 ] && FAILED=1
wait $PID2; RC=$?; log "  GPU2 done (rc=${RC})"; [ $RC -ne 0 ] && FAILED=1
wait $PID3; RC=$?; log "  GPU3 done (rc=${RC})"; [ $RC -ne 0 ] && FAILED=1

# ── Collect CSV ───────────────────────────────────────────────────────────────
log "Collecting → fig8b + fig9a + fig9b CSVs..."
python3 - "${OUTPUT_ROOT}" "${PROJECT_ROOT}/results" \
         "${PROJECT_ROOT}/output/qwen25vl_sweep_500" << 'PYEOF'
import json, csv, re, sys
from pathlib import Path
from collections import defaultdict

out_root  = Path(sys.argv[1])
csv_out   = Path(sys.argv[2])
res_sweep = Path(sys.argv[3])   # for fig9b
csv_out.mkdir(exist_ok=True)

FIELDS_8B = [
    'benchmark','resolution','max_new_tokens','iteration_idx','run_id','sample_id',
    'num_input_text_tokens',
    'num_visual_tokens_before_merger','num_visual_tokens_after_merger',
    'actual_generated_tokens',
    'visual_encoder_ms','visual_merger_ms','patch_embed_ms',
    'window_attn_total_ms','full_attn_total_ms',
    'llm_prefill_ms','ttft_ms','decode_loop_ms','tpot_ms','e2e_ms',
    'request_total_ms','request_output_tok_s','decode_only_tok_s',
    'f_vision','f_visual_prefill','f_decode',
]

# ── Fig 8b: output length sweep ───────────────────────────────────────────────
rows_8b = []
for d in sorted(out_root.iterdir()):
    if not d.is_dir() or d.name == 'logs': continue
    raw = d / 'timing_raw.jsonl'
    if not raw.exists(): continue
    m = re.match(r'(\w+)_tok(\d+)', d.name)
    if not m: continue
    bm, tok = m.group(1), int(m.group(2))
    records = [json.loads(l) for l in raw.read_text().strip().split('\n') if l.strip()]
    for i, r in enumerate(records):
        row = {'benchmark': bm, 'resolution': 1792, 'max_new_tokens': tok, 'iteration_idx': i}
        for f in FIELDS_8B[4:]: row[f] = r.get(f, '')
        rows_8b.append(row)

p8b = csv_out / 'fig8b_output_length_scaling_500dp.csv'
with open(p8b, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=FIELDS_8B); w.writeheader(); w.writerows(rows_8b)

counts = defaultdict(int)
for r in rows_8b: counts[(r['benchmark'], r['max_new_tokens'])] += 1
print(f"\n  Fig8b: {len(rows_8b)} rows")
for (bm, tok), n in sorted(counts.items()): print(f"    {bm} tok={tok}: {n} dp")

# ── Fig 9a: latency breakdown (docvqa res=1792 tok=64) ────────────────────────
# Use subset of Fig8b + resolution sweep 500dp
FIELDS_9A = [
    'source','benchmark','resolution','max_new_tokens','iteration_idx','run_id','sample_id',
    'num_visual_tokens_before_merger','num_visual_tokens_after_merger',
    'actual_generated_tokens',
    'image_preprocess_ms','processor_ms',
    'visual_encoder_ms','visual_merger_ms','patch_embed_ms',
    'window_attn_total_ms','full_attn_total_ms',
    'llm_prefill_ms','ttft_ms','decode_loop_ms','tpot_ms','e2e_ms',
    'request_total_ms','request_output_tok_s','decode_only_tok_s',
    'f_vision','f_visual_prefill','f_decode',
]
rows_9a = []

# From Fig8b: docvqa tok=64 (res=1792 fixed)
for r in rows_8b:
    if r['benchmark'] == 'docvqa' and int(r['max_new_tokens']) == 64:
        row = {'source': 'output_length_sweep', **r}
        rows_9a.append(row)

# From resolution sweep 500dp: docvqa res=1792 (tok=64 fixed)
res_raw = res_sweep / 'docvqa_res1792' / 'timing_raw.jsonl'
if res_raw.exists():
    for i, r in enumerate(json.loads(l) for l in res_raw.read_text().strip().split('\n') if l.strip()):
        row = {'source': 'resolution_sweep', 'benchmark': 'docvqa',
               'resolution': 1792, 'max_new_tokens': 64, 'iteration_idx': i}
        for f in FIELDS_9A[6:]: row[f] = r.get(f, '')
        rows_9a.append(row)

p9a = csv_out / 'fig9a_latency_breakdown_500dp.csv'
with open(p9a, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=FIELDS_9A, extrasaction='ignore')
    w.writeheader(); w.writerows(rows_9a)
print(f"\n  Fig9a: {len(rows_9a)} rows (docvqa res=1792 tok=64)")

# ── Fig 9b: Amdahl validation (from resolution sweep 500dp) ──────────────────
S_VISION = [1.2, 1.5, 2.0, 3.0]
FIELDS_9B = [
    'benchmark','resolution','max_new_tokens','iteration_idx','run_id','sample_id',
    'num_visual_tokens_before_merger','num_visual_tokens_after_merger',
    'actual_generated_tokens',
    'visual_encoder_ms','e2e_ms','ttft_ms','llm_prefill_ms',
    'window_attn_total_ms','full_attn_total_ms',
    'f_vision','f_visual_prefill','f_decode',
    'S_e2e_Sv1.2','S_e2e_Sv1.5','S_e2e_Sv2.0','S_e2e_Sv3.0',
]
rows_9b = []
for d in sorted(res_sweep.iterdir()):
    if not d.is_dir() or d.name in ('logs',): continue
    raw = d / 'timing_raw.jsonl'
    if not raw.exists(): continue
    m = re.match(r'(\w+)_res(\d+)', d.name)
    if not m: continue
    bm, res = m.group(1), int(m.group(2))
    for i, r in enumerate(json.loads(l) for l in raw.read_text().strip().split('\n') if l.strip()):
        fv = float(r.get('f_vision') or 0)
        row = {'benchmark': bm, 'resolution': res, 'max_new_tokens': 64, 'iteration_idx': i}
        for f in FIELDS_9B[4:18]: row[f] = r.get(f, '')
        for sv in S_VISION:
            denom = (1 - fv) + fv / sv
            row[f'S_e2e_Sv{sv}'] = round(1/denom, 6) if denom > 0 else ''
        rows_9b.append(row)

p9b = csv_out / 'fig9b_amdahl_validation_500dp.csv'
with open(p9b, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=FIELDS_9B); w.writeheader(); w.writerows(rows_9b)
counts9b = defaultdict(int)
for r in rows_9b: counts9b[(r['benchmark'], r['resolution'])] += 1
print(f"\n  Fig9b: {len(rows_9b)} rows")
for (bm, res), n in sorted(counts9b.items()): print(f"    {bm} res={res}: {n} dp")

print(f"\n  Files written to {csv_out}:")
for fn in sorted(csv_out.glob('fig*.csv')):
    n = sum(1 for _ in open(fn)) - 1
    print(f"    {fn.name:<50} {n:>5} rows  {fn.stat().st_size/1024:.0f} KB")
PYEOF

log "=== Output Length Sweep COMPLETE ==="
exit "${FAILED}"
