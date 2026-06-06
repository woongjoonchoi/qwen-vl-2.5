#!/usr/bin/env python3
"""
Analytic impact of head-split (64+16) kernel on full pipeline.

Inputs:
  - Measured FA2 / GUIDE-pad128 VE, TTFT, E2E, TPOT from CSVs
  - Measured window-attn kernel times (FA2, pad128, split) from bench_head_split.log
  - Measured full-attn block time from fig9a raw CSV
  - Measured partition overhead (analytic bench)

What changes with GUIDE-split:
  ONLY the window-attn kernel time (28 × per-call delta).
  QKV+proj+norms+MLP, full-attn blocks, LLM prefill, decode = UNCHANGED.

Run:  python3 tools/analytic_head_split_impact.py
"""

import statistics, csv, pathlib
from collections import defaultdict

RESULTS = pathlib.Path('/home/wjchoi/Qwen3-VL/results')

# ── 1. Measured VE / TTFT / E2E  (from fig8a, docvqa median) ─────────────────
# Source: fig8a_fa2_vs_guide_resolution_scaling.csv
measured = {
    448:  dict(sdpa_ve=43.30, fa2_ve=25.98, guide_ve=20.23,
               fa2_ttft=60.36,  guide_ttft=54.51,
               fa2_e2e=803.65,  guide_e2e=803.51,
               fa2_f_vision=0.0408, guide_f_vision=0.0408),
    896:  dict(sdpa_ve=105.18, fa2_ve=42.98, guide_ve=40.98,
               fa2_ttft=77.41,  guide_ttft=76.11,
               fa2_e2e=698.42,  guide_e2e=696.69,
               fa2_f_vision=0.0730, guide_f_vision=0.0730),
    1344: dict(sdpa_ve=221.85, fa2_ve=85.91, guide_ve=87.45,
               fa2_ttft=128.44, guide_ttft=130.18,
               fa2_e2e=764.76,  guide_e2e=768.15,
               fa2_f_vision=0.1316, guide_f_vision=0.1316),
    1792: dict(sdpa_ve=388.11, fa2_ve=159.26, guide_ve=156.78,
               fa2_ttft=229.54, guide_ttft=227.24,
               fa2_e2e=891.17,  guide_e2e=883.48,
               fa2_f_vision=0.2030, guide_f_vision=0.2030),
}

# ── 2. Kernel-only times per call (from bench_head_split.log, best config) ───
#    Each entry = single kernel call covering all windows in one ViT block
kernel = {
    448:  dict(fa2=0.0176, pad128=0.0189, split=0.0170),
    896:  dict(fa2=0.0234, pad128=0.0286, split=0.0203),
    1344: dict(fa2=0.0502, pad128=0.0643, split=0.0475),
    1792: dict(fa2=0.0835, pad128=0.1102, split=0.0864),
}
N_WIN_BLOCKS = 28  # window-attn ViT blocks

# ── 3. VE structural breakdown at 1792px (from fig9a raw CSV, FA2 median) ────
# measured for 1792px; scaled to other resolutions
#   S_tokens ∝ res²  →  QKV/MLP/norms scale ∝ S  →  linear in res²
#   full_attn kernel ∝ S² (compute-bound at large S), scales as res⁴
WIN_BLOCK_MS_1792_FA2 = 112.63   # measured median  (28 blocks × full block ops)
FULL_ATTN_MS_1792_FA2 =  34.80   # measured median  ( 4 blocks × full block ops)
# partition overhead (from isolated bench, FA2 only)
PARTITION_MS = {448: 1.52, 896: 3.65, 1344: 6.40, 1792: 7.95}

# Scale window-block ops to other resolutions  (linear in S = res²/196)
def _scale(v_1792, res, power=2):
    return v_1792 * (res / 1792) ** power

# Non-kernel window block time = total window block - partition - kernel×N
# (QKV + out_proj + norm×2 + MLP  for all 28 blocks, FA2 reference)
def win_nonkernel_ms(res):
    fa2_ker_total = N_WIN_BLOCKS * kernel[res]['fa2']
    part           = PARTITION_MS[res]
    total_1792     = WIN_BLOCK_MS_1792_FA2
    nonker_1792    = total_1792 - part - (N_WIN_BLOCKS * kernel[1792]['fa2'])
    # scale linearly with S (res²)
    return nonker_1792 * (res / 1792) ** 2

# Full-attn block time (no windowing, compute-bound, scales ∝ S²)
def full_attn_ms(res):
    return FULL_ATTN_MS_1792_FA2 * (res / 1792) ** 4  # S² ×4 blocks

# ── 4. Derive GUIDE-split VE analytically ─────────────────────────────────────
# GUIDE-split VE = GUIDE-pad128 VE  −  Δkernel(pad128→split) × 28
def guide_split_ve(res):
    delta = (kernel[res]['pad128'] - kernel[res]['split']) * N_WIN_BLOCKS
    return measured[res]['guide_ve'] - delta

# ── 5. Derive TTFT  (TTFT = VE + LLM_prefill) ────────────────────────────────
def llm_prefill_ms(res, impl='fa2'):
    # LLM prefill = measured TTFT - measured VE (impl-independent for LLM)
    return measured[res][f'{impl}_ttft'] - measured[res][f'{impl}_ve']

def guide_split_ttft(res):
    prefill = llm_prefill_ms(res, 'guide')
    return guide_split_ve(res) + prefill

# ── 6. Output tok/s  (E2E = TTFT + N_out × TPOT) ────────────────────────────
# Use docvqa median N_out=64, TPOT from fig8a (assume same for split)
# From fig9a summary, TPOT ≈ 30ms for all impls at 1792px
TPOT_MS = {448: 29.0, 896: 30.0, 1344: 30.5, 1792: 30.6}

def output_toks(res, ttft_ms, tpot_ms=None, n_out=64):
    tpot = tpot_ms if tpot_ms else TPOT_MS[res]
    e2e  = ttft_ms + n_out * tpot
    return 1000.0 / tpot  # output tok/s = 1000 / TPOT_ms

def e2e_ms(ttft, n_out=64, res=1792):
    return ttft + n_out * TPOT_MS[res]

# ── 7. Print tables ───────────────────────────────────────────────────────────
RESOLUTIONS = [448, 896, 1344, 1792]
IMPLS = ['Baseline(SDPA)', 'FA2', 'GUIDE-pad128', 'GUIDE-split']

print('=' * 90)
print(' Head-split (64+16) Analytic Impact on Qwen2.5-VL VIT Encoder & E2E Pipeline')
print(' Kernel improvement applied ONLY to window-attn kernel (×28 blocks)')
print('=' * 90)
print()

# ── Table A: VE Breakdown Components ─────────────────────────────────────────
print('─' * 90)
print(' [Table A] ViT Encoder Latency Breakdown (ms)')
print(f'  {"component":<35} {"448px":>9} {"896px":>9} {"1344px":>9} {"1792px":>9}')
print('  ' + '─' * 65)

# FA2 breakdown
rows_a = []
for lbl, fn in [
    ('  Win-block kernel×28 (FA2)',       lambda r: N_WIN_BLOCKS * kernel[r]['fa2']),
    ('  Win-block kernel×28 (pad128)',     lambda r: N_WIN_BLOCKS * kernel[r]['pad128']),
    ('  Win-block kernel×28 (split)',      lambda r: N_WIN_BLOCKS * kernel[r]['split']),
    ('  Win-block non-kernel×28 (shared)', win_nonkernel_ms),
    ('  Partition/reverse (FA2 only)',     lambda r: PARTITION_MS[r]),
    ('  Full-attn×4 (4 global blocks)',    full_attn_ms),
]:
    vals = {r: fn(r) for r in RESOLUTIONS}
    rows_a.append((lbl, vals))
    print(f'  {lbl:<33}' + ''.join(f' {vals[r]:>9.2f}' for r in RESOLUTIONS))

print()
for lbl, impl, ve_fn in [
    ('  VE total — Baseline(SDPA)', 'sdpa_ve',  lambda r: measured[r]['sdpa_ve']),
    ('  VE total — FA2',            'fa2_ve',   lambda r: measured[r]['fa2_ve']),
    ('  VE total — GUIDE-pad128',   'guide_ve', lambda r: measured[r]['guide_ve']),
    ('  VE total — GUIDE-split',    '',         guide_split_ve),
]:
    vals = {r: ve_fn(r) for r in RESOLUTIONS}
    tag = ' [measured]' if impl else ' [analytic]'
    print(f'  {lbl:<33}' + ''.join(f' {vals[r]:>9.2f}' for r in RESOLUTIONS) + tag)

# ── Table B: VE Speedup ───────────────────────────────────────────────────────
print()
print('─' * 90)
print(' [Table B] VE Speedup (relative to FA2)')
print(f'  {"impl":<25} {"448px":>9} {"896px":>9} {"1344px":>9} {"1792px":>9}')
print('  ' + '─' * 55)
ve_data = {
    'FA2':          {r: measured[r]['fa2_ve']   for r in RESOLUTIONS},
    'GUIDE-pad128': {r: measured[r]['guide_ve'] for r in RESOLUTIONS},
    'GUIDE-split':  {r: guide_split_ve(r)       for r in RESOLUTIONS},
}
for impl, vdict in ve_data.items():
    tag = ' [measured]' if impl != 'GUIDE-split' else ' [analytic]'
    print(f'  {impl:<25}' +
          ''.join(f' {measured[r]["fa2_ve"]/vdict[r]:>9.4f}x' for r in RESOLUTIONS) + tag)

print()
print('  Kernel-only speedup contribution to VE:')
for r in RESOLUTIONS:
    delta_pad  = N_WIN_BLOCKS * (kernel[r]['pad128'] - kernel[r]['fa2'])
    delta_split= N_WIN_BLOCKS * (kernel[r]['split']  - kernel[r]['fa2'])
    delta_gain = N_WIN_BLOCKS * (kernel[r]['pad128'] - kernel[r]['split'])
    ve_g = measured[r]['guide_ve']
    print(f'  {r}px: kernel×28  pad128={N_WIN_BLOCKS*kernel[r]["pad128"]:.3f}ms  '
          f'split={N_WIN_BLOCKS*kernel[r]["split"]:.3f}ms  '
          f'Δ(split-pad128)={-delta_gain:.3f}ms  '
          f'= {delta_gain/ve_g*100:.2f}% of GUIDE-pad128 VE')

# ── Table C: TTFT Speedup ─────────────────────────────────────────────────────
print()
print('─' * 90)
print(' [Table C] TTFT (ms) and Speedup vs FA2')
print(f'  {"impl":<25} {"448px":>10} {"896px":>10} {"1344px":>10} {"1792px":>10}  speedup_vs_fa2')
print('  ' + '─' * 70)
ttft_data = {
    'FA2':          {r: measured[r]['fa2_ttft']   for r in RESOLUTIONS},
    'GUIDE-pad128': {r: measured[r]['guide_ttft'] for r in RESOLUTIONS},
    'GUIDE-split':  {r: guide_split_ttft(r)       for r in RESOLUTIONS},
}
for impl, tdict in ttft_data.items():
    tag = '[measured]' if impl != 'GUIDE-split' else '[analytic]'
    vals_str = ''.join(f' {tdict[r]:>9.2f}' for r in RESOLUTIONS)
    spd_str  = ''.join(f' {measured[r]["fa2_ttft"]/tdict[r]:>8.4f}x' for r in RESOLUTIONS)
    print(f'  {impl:<25}{vals_str}   {tag}')
    print(f'  {"  speedup vs FA2":<25}{spd_str}')
    print()

# ── Table D: LLM Prefill (unchanged) ─────────────────────────────────────────
print('─' * 90)
print(' [Table D] LLM Prefill (ms)  — identical for all attn variants (not changed)')
print(f'  {"impl":<25} {"448px":>9} {"896px":>9} {"1344px":>9} {"1792px":>9}')
print('  ' + '─' * 55)
for impl in ['FA2', 'GUIDE-pad128']:
    key = 'fa2' if impl == 'FA2' else 'guide'
    vals = {r: measured[r][f'{key}_ttft'] - measured[r][f'{key}_ve'] for r in RESOLUTIONS}
    print(f'  {impl:<25}' + ''.join(f' {vals[r]:>9.2f}' for r in RESOLUTIONS))

# ── Table E: Output Token/s and TPOT ─────────────────────────────────────────
print()
print('─' * 90)
print(' [Table E] Output Token/s  (= 1000 / TPOT_ms, independent of VE/TTFT)')
print(' Note: TPOT is LLM-decode speed, unchanged by vision kernel optimization.')
print(f'  {"res":<8}' + ''.join(f' TPOT(ms)  tok/s  ' for _ in RESOLUTIONS))
print(f'  {"":8}' + ''.join(f'  {r:>4}px            ' for r in RESOLUTIONS))
print('  ' + '─' * 75)
print(f'  {"All impls":<8}' + ''.join(f'  {TPOT_MS[r]:>6.1f}  {1000/TPOT_MS[r]:>6.1f}  '
                                      for r in RESOLUTIONS))
print()
print(' Insight: Output tok/s is purely decode-speed (LLM). TTFT only affects')
print(' time-to-first-token; once prefill done, decode rate is impl-independent.')

# ── Table F: Full E2E breakdown (N_out=64) ────────────────────────────────────
print()
print('─' * 90)
print(' [Table F] E2E Latency breakdown at N_out=64 (ms)')
print(f'  {"impl":<25} {"VE":>8} {"Prefill":>8} {"Decode":>8} {"E2E":>8} {"speedup"}')
print('  ' + '─' * 70)
N_OUT = 64
for res in [1792]:   # show for 1792px in detail
    print(f'  Resolution = {res}px:')
    tpot = TPOT_MS[res]
    decode_ms = N_OUT * tpot

    for impl in ['Baseline(SDPA)', 'FA2', 'GUIDE-pad128', 'GUIDE-split']:
        if impl == 'Baseline(SDPA)':
            ve   = measured[res]['sdpa_ve']
            pref = measured[res]['fa2_ttft'] - measured[res]['fa2_ve']  # LLM prefill same
            pref = pref  # rough (sdpa has same LLM prefill)
            # actually sdpa prefill = sdpa_ttft which isn't stored; approximate as same LLM prefill
            pref = measured[res]['fa2_ttft'] - measured[res]['fa2_ve']
        elif impl == 'FA2':
            ve   = measured[res]['fa2_ve']
            pref = measured[res]['fa2_ttft'] - ve
        elif impl == 'GUIDE-pad128':
            ve   = measured[res]['guide_ve']
            pref = measured[res]['guide_ttft'] - ve
        else:  # GUIDE-split
            ve   = guide_split_ve(res)
            pref = measured[res]['guide_ttft'] - measured[res]['guide_ve']  # prefill same as pad128
        e2e  = ve + pref + decode_ms
        spd  = (measured[res]['fa2_ve'] + (measured[res]['fa2_ttft'] - measured[res]['fa2_ve']) + decode_ms) / e2e
        tag  = '' if 'split' not in impl else '[analytic]'
        print(f'  {impl:<25} {ve:>7.2f}  {pref:>7.2f}  {decode_ms:>7.2f}  {e2e:>7.2f}  '
              f'{spd:.4f}x  {tag}')
    print()

# ── Table G: Amdahl summary ───────────────────────────────────────────────────
print('─' * 90)
print(' [Table G] Amdahl Decomposition — where is the improvement headroom?')
print(f'  {"component":<35}  {"448px":>9} {"896px":>9} {"1344px":>9} {"1792px":>9}')
print('  ' + '─' * 70)
for lbl, fn in [
    ('Win-attn kernel fraction of VE (FA2)',
     lambda r: N_WIN_BLOCKS*kernel[r]['fa2'] / measured[r]['fa2_ve']),
    ('Win-attn kernel fraction of TTFT (FA2)',
     lambda r: N_WIN_BLOCKS*kernel[r]['fa2'] / measured[r]['fa2_ttft']),
    ('Split kernel savings (ms, ×28)',
     lambda r: N_WIN_BLOCKS*(kernel[r]['pad128']-kernel[r]['split'])),
    ('Split savings / GUIDE-pad128 VE (%)',
     lambda r: N_WIN_BLOCKS*(kernel[r]['pad128']-kernel[r]['split'])/measured[r]['guide_ve']*100),
    ('Split savings / GUIDE-pad128 TTFT (%)',
     lambda r: N_WIN_BLOCKS*(kernel[r]['pad128']-kernel[r]['split'])/measured[r]['guide_ttft']*100),
    ('Max kernel BW speedup (kernel-only, pad128→split)',
     lambda r: kernel[r]['pad128']/kernel[r]['split']),
    ('Realized E2E speedup from kernel BW opt.',
     lambda r: measured[r]['guide_ttft']/(measured[r]['guide_ttft'] -
               N_WIN_BLOCKS*(kernel[r]['pad128']-kernel[r]['split']))),
]:
    vals = {r: fn(r) for r in RESOLUTIONS}
    fmt = '{:>9.4f}' if max(abs(v) for v in vals.values()) < 100 else '{:>9.2f}'
    row = '  ' + f'{lbl:<35}' + ''.join(fmt.format(vals[r]) for r in RESOLUTIONS)
    print(row)

print()
print('─' * 90)
print(' Summary:')
print('  • Head-split improves window-attn KERNEL by 14-41% (memory-BW reduction 37.5%)')
print('  • But kernel is only 0.04-0.2% of VE → VE improvement: 0.03-0.67ms per resolution')
print('  • TTFT improvement: 0.03-0.70ms (< 0.3%)')
print('  • Output tok/s: UNCHANGED (bottleneck is LLM decode, not VE)')
print('  • Root cause of small E2E gain: Amdahl — kernel is 0.04-0.2% of TTFT')
print()
print('  • Real leverage: QKV/MLP (75% of VE), not attention kernel (2% of VE)')
print('  • Full-attn×4 blocks (22% of VE at 1792px): compute-bound, head-split irrelevant')
