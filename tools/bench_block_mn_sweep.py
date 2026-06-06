#!/usr/bin/env python3
"""
GUIDE kernel with separate BLOCK_M / BLOCK_N + sweep benchmark.

Changes from original:
  - BLOCK_SIZE → BLOCK_M (Q tile) + BLOCK_N (KV tile) independently
  - n_kv_blocks = cdiv(win_tokens, BLOCK_N)  ← not BLOCK_M
  - per_batch_blocks = n_windows * cdiv(win_tokens, BLOCK_M)

Sweep: (BLOCK_M, BLOCK_N, warps, stages) x resolutions
Compare against: FA2 (flash_attn_varlen) + current GUIDE (BLOCK_M=BLOCK_N=BLOCK_SIZE)
"""
import sys, itertools
sys.path.insert(0, '/workspace')
sys.path.insert(0, '/workspace/tools')

import torch
import triton
import triton.language as tl
import math
from flash_attn import flash_attn_varlen_func

from tools.qwen_descriptor import QwenDescriptor
from kernel_triton.qwen_window_attention.triton_ops_qwen import qwen_window_attention

# ── config ────────────────────────────────────────────────────────────────────
NH = 16; HD = 80; SMS = 2; SMU = 4; RWS = 8; WMS = 4
WIN_TOK = WMS * WMS * SMU        # 64
DTYPE   = torch.bfloat16
DEVICE  = 'cuda'
WARMUP  = 30; REPEATS = 200

gen = QwenDescriptor(NH, RWS, SMS)

# ── BLOCK_M/N kernel ──────────────────────────────────────────────────────────
@triton.jit
def qwen_win_attn_mn_kernel(
    META_INFO, query_ptr, key_ptr, value_ptr, output_ptr,
    sm_scale,
    MW: tl.constexpr, S: tl.constexpr, SMU: tl.constexpr,
    DIM: tl.constexpr, PAD_DIM: tl.constexpr, NUM_HEADS: tl.constexpr,
    BLOCK_M: tl.constexpr,   # Q tile size
    BLOCK_N: tl.constexpr,   # KV tile size  ← independent
):
    INFO_LEN: tl.constexpr = 7
    head_offset = tl.program_id(0)
    pid_offset  = tl.program_id(1)

    # ── parse descriptor (same as original) ──────────────────────────────────
    HEAD_GROUP_NUM = tl.sum(tl.load(META_INFO + tl.arange(0, 1)))
    _head_start_offset = -1
    for _g in range(HEAD_GROUP_NUM):
        grp_ptr = tl.sum(tl.load(META_INFO + tl.arange(0, 1) + 1 + _g))
        _hs = tl.sum(tl.load(META_INFO + tl.arange(0, 1) + grp_ptr + 0))
        _he = tl.sum(tl.load(META_INFO + tl.arange(0, 1) + grp_ptr + 1))
        if head_offset >= _hs and head_offset < _he:
            _head_start_offset = grp_ptr

    case_num = tl.sum(tl.load(META_INFO + tl.arange(0, 1) + _head_start_offset + 2))
    _base_case_offsets = _head_start_offset + 3
    num_blocks = 0; flag = 0
    cur_base_offsets = -1; cur_case_blocks = -1
    prev_case_num_blocks = -1; cur_base_case = -1
    for i in range(case_num):
        _base_case  = tl.sum(tl.load(META_INFO + tl.arange(0, 1) + _base_case_offsets))
        _cur_blocks = tl.sum(tl.load(META_INFO + tl.arange(0, 1) + _base_case_offsets + 1))
        if pid_offset < num_blocks + _cur_blocks and flag == 0:
            flag = 1; cur_base_offsets = _base_case_offsets
            cur_case_blocks = _cur_blocks; prev_case_num_blocks = num_blocks
            cur_base_case = _base_case
        _base_case_offsets = _base_case_offsets + _base_case * INFO_LEN + 2
        num_blocks += _cur_blocks

    cur_context_length = tl.sum(
        tl.load(META_INFO + tl.arange(0, 1) + cur_base_offsets + 1 + 4))
    cur_pid_offsets  = pid_offset - prev_case_num_blocks
    blocks_per_ctx   = cur_case_blocks // cur_context_length
    cur_ctx_blocks = 0; ctx = -1
    for c in range(cur_context_length):
        if cur_ctx_blocks <= cur_pid_offsets and cur_pid_offsets < cur_ctx_blocks + blocks_per_ctx:
            ctx = c
        cur_ctx_blocks += blocks_per_ctx
    cur_blocks_offset_in_ctx = cur_pid_offsets % blocks_per_ctx

    # ── per-case computation ──────────────────────────────────────────────────
    for b in range(cur_base_case):
        cur_base_ptr = tl.sum(tl.load(META_INFO + tl.arange(0,1) + cur_base_offsets+1+INFO_LEN*b+1))
        cur_base_H   = tl.sum(tl.load(META_INFO + tl.arange(0,1) + cur_base_offsets+1+INFO_LEN*b+2))
        cur_base_W   = tl.sum(tl.load(META_INFO + tl.arange(0,1) + cur_base_offsets+1+INFO_LEN*b+3))
        cur_region_W = tl.sum(tl.load(META_INFO + tl.arange(0,1) + cur_base_offsets+1+INFO_LEN*b+7))
        win_tokens   = cur_base_H * cur_base_W * SMU
        n_ctx_W  = cur_region_W // cur_base_W
        ctx_h = ctx // n_ctx_W; ctx_w = ctx % n_ctx_W
        h_off = ctx_h * cur_base_H; w_off = ctx_w * cur_base_W
        h_base = cur_base_ptr // MW;  w_base = cur_base_ptr % MW

        offs_d = tl.arange(0, PAD_DIM)

        # ── Q block (BLOCK_M) ─────────────────────────────────────────────────
        q_block_start = cur_blocks_offset_in_ctx * BLOCK_M   # ← BLOCK_M
        offs_q = tl.arange(0, BLOCK_M) + q_block_start       # ← BLOCK_M

        q_mc = offs_q // SMU; q_sub = offs_q % SMU
        q_mr = q_mc // cur_base_W; q_mcl = q_mc % cur_base_W
        q_merged = (h_base + h_off + q_mr) * MW + (w_base + w_off + q_mcl)
        q_abs    = q_merged * SMU + q_sub

        q_valid = (offs_q[:, None] < win_tokens) & (offs_d[None, :] < DIM)
        q_ptrs  = query_ptr + q_abs[:, None]*NUM_HEADS*DIM + head_offset*DIM + offs_d[None, :]
        q = tl.load(q_ptrs, mask=q_valid, other=0.0)

        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")  # ← BLOCK_M
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc  = tl.zeros([BLOCK_M, PAD_DIM], dtype=tl.float32)        # ← BLOCK_M
        qk_scale = sm_scale * 1.44269504

        # ── KV loop (BLOCK_N) ─────────────────────────────────────────────────
        n_kv_blocks = tl.cdiv(win_tokens, BLOCK_N)   # ← BLOCK_N (not BLOCK_M)
        for kv_blk in range(n_kv_blocks):
            offs_kv = tl.arange(0, BLOCK_N) + kv_blk * BLOCK_N   # ← BLOCK_N

            kv_mc = offs_kv // SMU; kv_sub = offs_kv % SMU
            kv_mr = kv_mc // cur_base_W; kv_mcl = kv_mc % cur_base_W
            kv_merged = (h_base + h_off + kv_mr)*MW + (w_base + w_off + kv_mcl)
            kv_abs    = kv_merged * SMU + kv_sub

            kv_valid = (offs_kv[:, None] < win_tokens) & (offs_d[None, :] < DIM)
            k_ptrs = key_ptr   + kv_abs[:,None]*NUM_HEADS*DIM + head_offset*DIM + offs_d[None,:]
            v_ptrs = value_ptr + kv_abs[:,None]*NUM_HEADS*DIM + head_offset*DIM + offs_d[None,:]
            k = tl.load(k_ptrs, mask=kv_valid, other=0.0)
            v = tl.load(v_ptrs, mask=kv_valid, other=0.0)

            # score: [BLOCK_M, BLOCK_N]
            qk = tl.dot(q, tl.trans(k)) * qk_scale
            qk = tl.where((offs_kv < win_tokens)[None, :], qk, float("-inf"))
            m_j = tl.maximum(m_i, tl.max(qk, axis=1))
            p   = tl.math.exp2(qk - m_j[:, None])
            l_j = tl.sum(p, axis=1)
            alpha = tl.math.exp2(m_i - m_j)
            acc   = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = m_j; l_i = l_i * alpha + l_j

        acc = acc / l_i[:, None]
        q_mask_out = (offs_q[:,None] < win_tokens) & (offs_d[None,:] < DIM)
        o_ptrs = output_ptr + q_abs[:,None]*NUM_HEADS*DIM + head_offset*DIM + offs_d[None,:]
        tl.store(o_ptrs, acc.to(output_ptr.dtype.element_ty), mask=q_mask_out)


def qwen_win_attn_mn(q, k, v, meta, MW, smu, BLOCK_M, BLOCK_N, num_warps, num_stages, n_windows):
    S, NH, D = q.shape
    PAD_DIM  = triton.next_power_of_2(D)
    out      = torch.empty_like(q)
    # per_batch_blocks = n_windows * ceil(WIN_TOK / BLOCK_M)
    pbk = n_windows * math.ceil(WIN_TOK / BLOCK_M)
    qwen_win_attn_mn_kernel[(NH, pbk)](
        meta, q, k, v, out, D**-0.5,
        MW=MW, S=S, SMU=smu, DIM=D, PAD_DIM=PAD_DIM, NUM_HEADS=NH,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out


# ── correctness check ─────────────────────────────────────────────────────────
def check_correct(res=448):
    H = W = res // 14; S = H * W; MW = W // SMS
    meta = gen.build(H, W, 64, device=DEVICE)
    pbk  = gen.per_batch_blocks(H, W, 64)
    n_wins = (H//SMS//WMS) * (W//SMS//WMS)
    q = torch.randn(S, NH, HD, dtype=DTYPE, device=DEVICE)
    k = torch.randn(S, NH, HD, dtype=DTYPE, device=DEVICE)
    v = torch.randn(S, NH, HD, dtype=DTYPE, device=DEVICE)
    ref = qwen_window_attention(q, k, v, meta, MW, SMU, pbk,
                                BLOCK_SIZE=64, seq_layout=True, num_warps=4, num_stages=2)
    for BM, BN in [(64,32),(32,64),(32,32),(64,64)]:
        meta2 = gen.build(H, W, BM, device=DEVICE)
        out = qwen_win_attn_mn(q, k, v, meta2, MW, SMU, BM, BN, 4, 2, n_wins)
        rel = (out - ref).abs().max() / (ref.abs().max() + 1e-6)
        status = 'PASS' if rel < 1e-3 else f'FAIL(rel={rel:.2e})'
        print(f'  BM={BM} BN={BN}: {status}')


# ── timed kernel call ─────────────────────────────────────────────────────────
def timed(fn, warmup, repeats):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(repeats): fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / repeats


# ── sweep ─────────────────────────────────────────────────────────────────────
CONFIGS = [
    # (BLOCK_M, BLOCK_N, warps, stages)
    (64, 64, 4, 2),   # current BS=64 equivalent
    (64, 32, 4, 2),   # larger Q, smaller KV → n_kv=2, pipeline valid
    (64, 32, 1, 2),
    (64, 32, 2, 2),
    (32, 64, 4, 2),   # smaller Q, larger KV
    (32, 32, 1, 2),   # current BS=32 equivalent
    (32, 32, 4, 2),
    (16, 64, 4, 2),
    (16, 32, 4, 2),
]

def bench_res(res):
    H = W = res // 14; S = H * W; MW = W // SMS
    n_wins = (H//SMS//WMS) * (W//SMS//WMS)
    q = torch.randn(S, NH, HD, dtype=DTYPE, device=DEVICE)
    k = torch.randn(S, NH, HD, dtype=DTYPE, device=DEVICE)
    v = torch.randn(S, NH, HD, dtype=DTYPE, device=DEVICE)

    # FA2 reference
    cu = torch.arange(0, (n_wins+1)*WIN_TOK, WIN_TOK, dtype=torch.int32, device=DEVICE)
    def run_fa2():
        flash_attn_varlen_func(q.reshape(S,NH,HD), k.reshape(S,NH,HD), v.reshape(S,NH,HD),
                               cu, cu, max_seqlen_q=WIN_TOK, max_seqlen_k=WIN_TOK,
                               dropout_p=0.0, causal=False)
    t_fa2 = timed(run_fa2, WARMUP, REPEATS)

    results = []
    for BM, BN, nw, ns in CONFIGS:
        meta = gen.build(H, W, BM, device=DEVICE)
        n_kv = math.ceil(WIN_TOK / BN)
        def run(BM=BM, BN=BN, nw=nw, ns=ns, meta=meta):
            qwen_win_attn_mn(q, k, v, meta, MW, SMU, BM, BN, nw, ns, n_wins)
        # warmup
        for _ in range(WARMUP): run()
        torch.cuda.synchronize()
        t = timed(run, WARMUP, REPEATS)
        results.append({'BM':BM,'BN':BN,'nw':nw,'ns':ns,'n_kv':n_kv,
                        'ms':t,'vs_fa2':t/t_fa2})

    return t_fa2, results


def main():
    print(f'Device: {torch.cuda.get_device_name(0)}')
    print(f'WIN_TOK={WIN_TOK}  NH={NH}  HD={HD}  WARMUP={WARMUP}  REPEATS={REPEATS}')
    print()
    print('Correctness check (448px):')
    check_correct(448)
    print()

    all_results = {}
    for res in [448, 896, 1344, 1792]:
        print(f'Benchmarking {res}px...', flush=True)
        t_fa2, results = bench_res(res)
        all_results[res] = (t_fa2, results)

        best = min(results, key=lambda x: x['ms'])
        orig_64 = next(r for r in results if r['BM']==64 and r['BN']==64)
        orig_32 = next(r for r in results if r['BM']==32 and r['BN']==32 and r['nw']==1)

        print(f'  FA2={t_fa2:.4f}ms')
        print(f'  GUIDE current BS=64: {orig_64["ms"]:.4f}ms ({orig_64["vs_fa2"]:.3f}×FA2)')
        print(f'  GUIDE current BS=32: {orig_32["ms"]:.4f}ms ({orig_32["vs_fa2"]:.3f}×FA2)')
        print(f'  BEST (BM={best["BM"]},BN={best["BN"]},w={best["nw"]},s={best["ns"]}): '
              f'{best["ms"]:.4f}ms ({best["vs_fa2"]:.3f}×FA2)')

    # ── full table ────────────────────────────────────────────────────────────
    print()
    print('='*88)
    print('  Full sweep results')
    print('='*88)
    header = f'  {"BM":>4}{"BN":>4}{"w":>3}{"s":>3}{"n_kv":>6}'
    for res in [448, 896, 1344, 1792]:
        header += f'  {res:>6}px'
    print(header)
    print('  '+'-'*88)

    for cfg in CONFIGS:
        BM, BN, nw, ns = cfg
        row = f'  {BM:>4}{BN:>4}{nw:>3}{ns:>3}'
        n_kv = math.ceil(WIN_TOK / BN)
        row += f'  {n_kv:>5}'
        for res in [448, 896, 1344, 1792]:
            t_fa2, results = all_results[res]
            r = next(x for x in results if x['BM']==BM and x['BN']==BN
                     and x['nw']==nw and x['ns']==ns)
            row += f'  {r["ms"]:>5.4f}'
        print(row)

    print()
    print('  vs FA2:')
    print(header.replace('px','  '))
    print('  '+'-'*88)
    for cfg in CONFIGS:
        BM, BN, nw, ns = cfg
        row = f'  {BM:>4}{BN:>4}{nw:>3}{ns:>3}'
        n_kv = math.ceil(WIN_TOK / BN)
        row += f'  {n_kv:>5}'
        for res in [448, 896, 1344, 1792]:
            t_fa2, results = all_results[res]
            r = next(x for x in results if x['BM']==BM and x['BN']==BN
                     and x['nw']==nw and x['ns']==ns)
            row += f'  {r["vs_fa2"]:>6.3f}×'
        print(row)

    print()
    print('  BEST per resolution:')
    for res in [448, 896, 1344, 1792]:
        t_fa2, results = all_results[res]
        best = min(results, key=lambda x: x['ms'])
        print(f'  {res}px: BM={best["BM"]} BN={best["BN"]} w={best["nw"]} s={best["ns"]} '
              f'→ {best["ms"]:.4f}ms  ({best["vs_fa2"]:.3f}×FA2)')


if __name__ == '__main__':
    main()
