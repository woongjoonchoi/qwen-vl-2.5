#!/bin/bash
# E2E latency vs output_length at 1792px.
# Runs FA2 then GUIDE in separate Docker containers → no cache contamination.
# Output: results/output_len_e2e_fa2_raw.csv  +  results/output_len_e2e_guide_raw.csv

set -e
WORKSPACE=/home/wjchoi/Qwen3-VL
IMAGE=qwen25vl-guide:latest   # has flash_attn; used for both modes

DOCKER_COMMON="docker run --rm --gpus device=0
  -v ${WORKSPACE}:/workspace
  -e HOME=/tmp
  -e TRITON_CACHE_DIR=/tmp/.triton
  -e XDG_CACHE_HOME=/tmp/.cache"

echo "============================================================"
echo " [1/2] FA2"
echo "============================================================"
$DOCKER_COMMON $IMAGE bash -c "
  cd /workspace &&
  python3 tools/bench_output_len_e2e.py \
    --mode  fa2 \
    --output /workspace/results/output_len_e2e_fa2_raw.csv \
  2>&1 | tee /workspace/results/output_len_e2e_fa2.log
"

echo ""
echo "============================================================"
echo " [2/2] GUIDE"
echo "============================================================"
$DOCKER_COMMON $IMAGE bash -c "
  cd /workspace &&
  python3 tools/bench_output_len_e2e.py \
    --mode  guide \
    --output /workspace/results/output_len_e2e_guide_raw.csv \
  2>&1 | tee /workspace/results/output_len_e2e_guide.log
"

echo ""
echo "============================================================"
echo " Summary"
echo "============================================================"
python3 - <<'PYEOF'
import csv, statistics
from collections import defaultdict

def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))

fa2   = load('/home/wjchoi/Qwen3-VL/results/output_len_e2e_fa2_raw.csv')
guide = load('/home/wjchoi/Qwen3-VL/results/output_len_e2e_guide_raw.csv')

fa2_by_n   = defaultdict(list)
guide_by_n = defaultdict(list)
for r in fa2:   fa2_by_n[int(r['output_length'])].append(float(r['e2e_ms']))
for r in guide: guide_by_n[int(r['output_length'])].append(float(r['e2e_ms']))

print(f"{'N':>6} | {'FA2 E2E':>10} | {'GUIDE E2E':>10} | {'speedup':>8}")
print('-' * 45)
rows = []
for n in sorted(fa2_by_n.keys()):
    fa2_med  = statistics.median(fa2_by_n[n])
    gui_med  = statistics.median(guide_by_n.get(n, [fa2_med]))
    sp = fa2_med / gui_med
    print(f"{n:>6} | {fa2_med:>8.1f}ms | {gui_med:>8.1f}ms | {sp:>7.3f}x")
    rows.append({'output_length':n,'fa2_e2e_ms':round(fa2_med,1),'guide_e2e_ms':round(gui_med,1),'speedup_guide_vs_fa2':round(sp,4)})

# save summary
import csv as CSV
with open('/home/wjchoi/Qwen3-VL/results/output_len_e2e_summary.csv','w',newline='') as f:
    w = CSV.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader(); w.writerows(rows)
print('\nSaved summary → results/output_len_e2e_summary.csv')
PYEOF
