#!/usr/bin/env bash
# nsys block-size sweep for all 4 resolutions in parallel (one GPU each)
# resolutions: 448 896 1344 1792
# candidates: BS={16,32,64} x warps={4,8} x stages={3,4}
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

IMAGE="${IMAGE:-qwen25vl-guide:latest}"
MODEL_PATH="${PROJECT_ROOT}/models/Qwen2.5-VL-3B-Instruct"
NSYS_HOST_DIR="/opt/nvidia/nsight-systems/2025.3.2"
OUTPUT_ROOT="${PROJECT_ROOT}/output/qwen25vl_guide_kernel_level/nsys_block_sweep"
PRECISION="${PRECISION:-bf16}"
PROFILE_ITERS="${PROFILE_ITERS:-50}"
WARMUP="${WARMUP:-10}"

c_model="${MODEL_PATH/${PROJECT_ROOT}/\/workspace}"

run_docker_bg() {
  local gpu="$1" res="$2" patches="$3"
  local out="${OUTPUT_ROOT}/res${res}"
  mkdir -p "${out}"
  local c_out="${out/${PROJECT_ROOT}/\/workspace}"
  local base="qwen25vl-guide-kernel-block-sweep-res${res}-${PRECISION}"
  local mounts=("-v" "${PROJECT_ROOT}:/workspace")
  [ -d "${NSYS_HOST_DIR}" ] && mounts+=("-v" "${NSYS_HOST_DIR}:/opt/nsight-systems:ro")

  echo "[gpu${gpu}] starting res=${res} -> ${out}" | tee -a "${OUTPUT_ROOT}/sweep_master.log"

  docker run --rm --gpus "device=${gpu}" --ipc=host --ulimit memlock=-1 \
    --user "$(id -u):$(id -g)" "${mounts[@]}" \
    -e CUDA_VISIBLE_DEVICES=0 -e PYTHONUNBUFFERED=1 \
    -e HOME=/tmp -e TRITON_CACHE_DIR=/tmp/.triton -e XDG_CACHE_HOME=/tmp/.cache \
    -e PATH="/opt/nsight-systems/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin" \
    -w /workspace "${IMAGE}" -c "
set -e
echo '[res${res}] nsys profiling...'
nsys profile -t cuda,nvtx,osrt,cublas -s none --force-overwrite=true \
  -o '${c_out}/${base}' \
  python3 -u /workspace/tools/bench_qwen25vl_guide_kernel_nsys.py \
    --model-path '${c_model}' \
    --mode synthetic \
    --H ${patches} --W ${patches} \
    --block-sizes '16,32,64' \
    --num-warps '4,8' \
    --num-stages '3,4' \
    --profile-iters ${PROFILE_ITERS} \
    --warmup ${WARMUP} \
    --precision '${PRECISION}' \
    --output-dir '${c_out}/${base}_run'

echo '[res${res}] nsys stats...'
nsys stats -r nvtx_gpu_proj_trace --format csv \
  --output '${c_out}/${base}' \
  '${c_out}/${base}.nsys-rep'

echo '[res${res}] parsing...'
python3 /workspace/tools/parse_qwen_kernel_nsys_sweep.py \
  --input '${c_out}/${base}_nvtx_gpu_proj_trace.csv' \
  --output '${c_out}/${base}_block_sweep_summary.csv' \
  --info '${c_out}/${base}_run/bench_info.json' \
  --correctness '${c_out}/${base}_run/candidates_correctness.json' \
  --best '${c_out}/best_block_config.json'

echo '[res${res}] DONE'
" >> "${out}/docker_stdout.log" 2>&1
  echo "[gpu${gpu}] res=${res} DONE (exit=$?)" | tee -a "${OUTPUT_ROOT}/sweep_master.log"
}

mkdir -p "${OUTPUT_ROOT}"
echo "======================================" | tee "${OUTPUT_ROOT}/sweep_master.log"
echo "Block-size sweep: all 4 resolutions"  | tee -a "${OUTPUT_ROOT}/sweep_master.log"
echo "Started: $(date)"                     | tee -a "${OUTPUT_ROOT}/sweep_master.log"
echo "======================================" | tee -a "${OUTPUT_ROOT}/sweep_master.log"

# 4 GPUs in parallel  (res_px, H=W in patch units = res_px // 14)
run_docker_bg 0 448   32 &  PID0=$!
run_docker_bg 1 896   64 &  PID1=$!
run_docker_bg 2 1344  96 &  PID2=$!
run_docker_bg 3 1792 128 &  PID3=$!

echo "Waiting for all 4 jobs (PIDs: $PID0 $PID1 $PID2 $PID3)..."
wait $PID0; E0=$?
wait $PID1; E1=$?
wait $PID2; E2=$?
wait $PID3; E3=$?

echo "All done: exit codes $E0 $E1 $E2 $E3" | tee -a "${OUTPUT_ROOT}/sweep_master.log"

# ── Summary ──────────────────────────────────────────────────────────────────
python3 - "${OUTPUT_ROOT}" "${PRECISION}" <<'PY'
import sys, json, csv
from pathlib import Path
root = Path(sys.argv[1]); prec = sys.argv[2]

print("\n" + "="*72)
print(f"  Block-Size Sweep Results  ({prec})")
print("="*72)
print(f"  {'res':>6}  {'BS':>4}  {'warps':>5}  {'stages':>6}  {'mean_us':>8}  {'status':>6}  winner")
print("  " + "-"*68)

best_per_res = {}
for res in [448, 896, 1344, 1792]:
    csv_path = root / f"res{res}" / f"qwen25vl-guide-kernel-block-sweep-res{res}-{prec}_block_sweep_summary.csv"
    best_path = root / f"res{res}" / "best_block_config.json"
    if not csv_path.exists():
        print(f"  {res:>6}  MISSING"); continue
    rows = list(csv.DictReader(open(csv_path)))
    best = json.loads(best_path.read_text()) if best_path.exists() else {}
    best_per_res[res] = best
    for r in rows:
        bs = r['block_size']; nw = r['num_warps']; ns = r['num_stages']
        mean_us = float(r.get('mean_ms_per_iter', 0)) * 1000
        st = r.get('correctness_status','?')
        is_best = (str(bs)==str(best.get('block_size','')) and
                   str(nw)==str(best.get('num_warps','')) and
                   str(ns)==str(best.get('num_stages','')))
        mark = " ← BEST" if is_best else ""
        print(f"  {res:>6}  {bs:>4}  {nw:>5}  {ns:>6}  {mean_us:>8.1f}  {st:>6}{mark}")
    print()

print("="*72)
print("  Best config per resolution:")
for res, best in best_per_res.items():
    if best:
        print(f"  res={res:4d}: BS={best.get('block_size')} warps={best.get('num_warps')} "
              f"stages={best.get('num_stages')}  "
              f"{float(best.get('mean_ms_per_iter',0))*1e3:.1f} us/iter")
PY
