#!/usr/bin/env python3
"""
Window attention kernel benchmark: FA2 varlen vs GUIDE (BSHD layout).

Layout comparison:
  BHSD [1, H, S, D] = seq_layout=False  (old, non-contiguous token access)
  BSHD [1, S, H, D] = BSHD  (new — heads contiguous per token, same stride as SHD)
  SHD  [S, H, D]    = seq_layout=True   (current optimized, equivalent to BSHD B=1)

Access pattern for token t, head h:
  BHSD: ptr + h*(S*D) + t*D   ← stride S*D between heads of same token
  BSHD: ptr + t*(H*D) + h*D   ← stride D between heads (contiguous within token)
  SHD:  ptr + t*(H*D) + h*D   ← identical to BSHD

Run inside qwen25vl-guide Docker container.
"""
import sys, time
sys.path.insert(0, '/workspace')
sys.path.insert(0, '/workspace/tools')

import torch
import triton

from tools.qwen_descriptor import QwenDescriptor, get_per_batch_blocks
from kernel_triton.qwen_window_attention.triton_ops_qwen import (
    qwen_window_attention, qwen_window_attention_kernel
)
from flash_attn import flash_attn_varlen_func

# ── model config (Qwen2.5-VL-3B) ─────────────────────────────────────────────
NUM_HEADS = 16
HEAD_DIM  = 80
SMS       = 2       # spatial_merge_size
SMU       = SMS**2  # spatial_merge_unit = 4
RWS       = 8       # raw window size = window_size_px // patch_size = 112 // 14
WMS       = RWS // SMS  # window size in merged cells = 4
WIN_TOK   = WMS * WMS * SMU   # tokens per window = 4*4*4 = 64
DTYPE     = torch.bfloat16
DEVICE    = 'cuda'

RESOLUTIONS = [448, 896, 1344, 1792]
WARMUP      = 20
REPEATS     = 100

gen = QwenDescriptor(NUM_HEADS, RWS, SMS)


def _kernel_cfg(S_raw: int):
    if S_raw <= 4096:
        return 64, 4, 2
    else:
        return 32, 1, 2


def make_qkv_bshd(S, NH, D):
    """Returns Q,K,V in BSHD [1, S, H, D] layout — contiguous."""
    q = torch.randn(1, S, NH, D, dtype=DTYPE, device=DEVICE)
    k = torch.randn(1, S, NH, D, dtype=DTYPE, device=DEVICE)
    v = torch.randn(1, S, NH, D, dtype=DTYPE, device=DEVICE)
    return q, k, v


def make_qkv_bhsd(S, NH, D):
    """Returns Q,K,V in BHSD [1, H, S, D] — non-contiguous token access."""
    q = torch.randn(1, NH, S, D, dtype=DTYPE, device=DEVICE)
    k = torch.randn(1, NH, S, D, dtype=DTYPE, device=DEVICE)
    v = torch.randn(1, NH, S, D, dtype=DTYPE, device=DEVICE)
    return q, k, v


def make_qkv_fa2(S, NH, D):
    """Returns Q,K,V in [S, NH, D] — FA2 varlen expected format."""
    q = torch.randn(S, NH, D, dtype=DTYPE, device=DEVICE)
    k = torch.randn(S, NH, D, dtype=DTYPE, device=DEVICE)
    v = torch.randn(S, NH, D, dtype=DTYPE, device=DEVICE)
    return q, k, v


def bshd_to_shd(x):
    """[1, S, H, D] → [S, H, D] view (zero copy, same memory)."""
    return x.squeeze(0)   # contiguous view


def fa2_window_attn(q_shd, k_shd, v_shd, H, W):
    """FA2 varlen window attention — uses get_window_index for cu_seqlens."""
    from transformers import Qwen2_5_VLForConditionalGeneration
    # build cu_seqlens manually from H, W grid
    MH, MW = H // SMS, W // SMS
    wh = RWS // SMS   # = 4
    ww = RWS // SMS

    import math
    n_win_h = math.ceil(MH / wh)
    n_win_w = math.ceil(MW / ww)

    # Simplified: assume divisible (interior windows only) for timing test
    win_size = wh * ww * SMU  # tokens per window
    n_wins   = n_win_h * n_win_w
    cu_seqlens = torch.arange(0, (n_wins + 1) * win_size, win_size,
                               dtype=torch.int32, device=DEVICE)
    # scatter q to window-major order (approximate — use identity for timing)
    out = flash_attn_varlen_func(
        q_shd.reshape(-1, NUM_HEADS, HEAD_DIM),
        k_shd.reshape(-1, NUM_HEADS, HEAD_DIM),
        v_shd.reshape(-1, NUM_HEADS, HEAD_DIM),
        cu_seqlens, cu_seqlens,
        seqlen_q=win_size, seqlen_k=win_size,
        dropout_p=0.0, causal=False,
    )
    return out


def timed(fn, warmup, repeats):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(repeats):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / repeats


def bench_one_res(res):
    px   = res
    H    = px // 14       # raw patch grid H = W
    W    = px // 14
    S    = H * W          # total raw tokens
    MH   = H // SMS
    MW   = W // SMS
    BLOCK_SIZE, NUM_WARPS, NUM_STAGES = _kernel_cfg(S)

    # build descriptor
    meta = gen.build(H, W, BLOCK_SIZE, device=DEVICE)
    pbk  = gen.per_batch_blocks(H, W, BLOCK_SIZE)

    # ── prepare inputs ──────────────────────────────────────────────────────
    # BSHD [1, S, H, D]
    qB, kB, vB = make_qkv_bshd(S, NUM_HEADS, HEAD_DIM)
    # BHSD [1, H, S, D]
    qH, kH, vH = make_qkv_bhsd(S, NUM_HEADS, HEAD_DIM)
    # SHD  [S, H, D] for FA2
    qF, kF, vF = make_qkv_fa2(S, NUM_HEADS, HEAD_DIM)

    # cu_seqlens for FA2 (divisible interior windows only)
    n_wins    = (MH // WMS) * (MW // WMS)
    cu = torch.arange(0, (n_wins + 1) * WIN_TOK, WIN_TOK,
                       dtype=torch.int32, device=DEVICE)

    # ── bench FA2 ────────────────────────────────────────────────────────────
    def run_fa2():
        flash_attn_varlen_func(
            qF, kF, vF, cu, cu,
            max_seqlen_q=WIN_TOK, max_seqlen_k=WIN_TOK,
            dropout_p=0.0, causal=False,
        )

    # ── bench GUIDE BSHD (use seq_layout=True after squeeze) ─────────────────
    qBs = bshd_to_shd(qB)   # [S, H, D] view — zero copy
    kBs = bshd_to_shd(kB)
    vBs = bshd_to_shd(vB)

    def run_bshd():
        qwen_window_attention(
            qBs, kBs, vBs, meta, MW, SMU, pbk,
            BLOCK_SIZE=BLOCK_SIZE, seq_layout=True,
            num_warps=NUM_WARPS, num_stages=NUM_STAGES,
        )

    # ── bench GUIDE BHSD (seq_layout=False, old path) ────────────────────────
    def run_bhsd():
        qwen_window_attention(
            qH, kH, vH, meta, MW, SMU, pbk,
            BLOCK_SIZE=BLOCK_SIZE, seq_layout=False,
            num_warps=NUM_WARPS, num_stages=NUM_STAGES,
        )

    # warmup all three
    for _ in range(WARMUP):
        run_fa2(); run_bshd(); run_bhsd()
    torch.cuda.synchronize()

    t_fa2  = timed(run_fa2,  WARMUP, REPEATS)
    t_bshd = timed(run_bshd, WARMUP, REPEATS)
    t_bhsd = timed(run_bhsd, WARMUP, REPEATS)

    return {
        'res': res, 'S': S, 'n_wins': n_wins,
        'fa2_ms':  t_fa2,
        'bshd_ms': t_bshd,
        'bhsd_ms': t_bhsd,
        'bshd_vs_fa2': t_bshd / t_fa2,
        'bhsd_vs_fa2': t_bhsd / t_fa2,
        'bshd_vs_bhsd': t_bhsd / t_bshd,
        'BS': BLOCK_SIZE, 'warps': NUM_WARPS,
    }


def main():
    print(f'Device: {torch.cuda.get_device_name(0)}')
    print(f'Warmup={WARMUP}  Repeats={REPEATS}')
    print()
    print('Layout definitions:')
    print('  FA2   : flash_attn_varlen  [S, H, D] (scatter reorder + varlen)')
    print('  BSHD  : GUIDE Triton [B,S,H,D] → squeeze → [S,H,D] seq_layout=True')
    print('           stride: S*(H*D) | H*D | D | 1  ← heads contiguous per token')
    print('  BHSD  : GUIDE Triton [B,H,S,D] seq_layout=False')
    print('           stride: H*(S*D) | S*D | D | 1  ← non-contiguous token access')
    print()

    results = []
    for res in RESOLUTIONS:
        print(f'Benchmarking {res}px (S={(res//14)**2})...', flush=True)
        r = bench_one_res(res)
        results.append(r)
        print(f'  FA2={r["fa2_ms"]:.4f}ms  BSHD={r["bshd_ms"]:.4f}ms  BHSD={r["bhsd_ms"]:.4f}ms  '
              f'[BS={r["BS"]},w={r["warps"]}]')

    print()
    print('='*72)
    print('  Window Attention Kernel Benchmark')
    print(f'  {"res":>6} | {"S":>6} | {"n_wins":>7} | {"FA2":>9} | {"BSHD":>9} | {"BHSD":>9} | {"BSHD/FA2":>9} | {"BHSD/FA2":>9}')
    print('  '+'-'*72)
    for r in results:
        print(f'  {r["res"]:>4}px | {r["S"]:>6} | {r["n_wins"]:>7} | '
              f'{r["fa2_ms"]:>7.4f}ms | {r["bshd_ms"]:>7.4f}ms | {r["bhsd_ms"]:>7.4f}ms | '
              f'{r["bshd_vs_fa2"]:>7.3f}x  | {r["bhsd_vs_fa2"]:>7.3f}x')

    print()
    print('  BSHD vs BHSD (memory layout effect):')
    for r in results:
        print(f'  {r["res"]:>4}px: BHSD/BSHD = {r["bshd_vs_bhsd"]:.3f}x  '
              f'(BSHD {(1-1/r["bshd_vs_bhsd"])*100:.1f}% faster than BHSD)')


if __name__ == '__main__':
    main()
