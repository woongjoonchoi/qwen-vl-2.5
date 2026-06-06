#!/usr/bin/env python3
"""
Head-dim split benchmark: PAD_DIM=128 vs split 64+16 for HD=80.

Problem:  head_dim=80 → PAD_DIM=128 wastes 48 lanes (37.5% BW waste).
Solution: split head into main[0:64] + tail[64:80=16], both exact powers of 2.

  QK score:  dot(q_main, k_main.T) + dot(q_tail, k_tail.T)   [BM×BN]
  V accum:   acc_main += p @ v_main   [BM×64]
             acc_tail += p @ v_tail   [BM×16]

Expected BW reduction: 80/128 = 62.5% → up to ~37.5% speedup (if memory-bound).

Variants tested (all seq_layout=True):
  FA2          : flash_attn_varlen baseline
  guide_pad128 : current GUIDE (PAD_DIM=128), best BM/BN from sweep
  guide_split  : head-split kernel (DIM_MAIN=64, DIM_TAIL=16), sweep BM/BN

Run:
  docker run --rm --gpus device=0 -v /home/wjchoi/Qwen3-VL:/workspace
    qwen25vl-guide:latest python3 /workspace/tools/bench_head_split.py
    2>&1 | tee /workspace/results/bench_head_split.log
"""
import sys, math
sys.path.insert(0, '/workspace')
sys.path.insert(0, '/workspace/tools')

import torch
import triton
import triton.language as tl
from flash_attn import flash_attn_varlen_func

from tools.qwen_descriptor import QwenDescriptor
from kernel_triton.qwen_window_attention.triton_ops_qwen import qwen_window_attention

# ── model config ──────────────────────────────────────────────────────────────
NH      = 16;  HD = 80;  SMS = 2;  SMU = 4;  RWS = 8;  WMS = 4
WIN_TOK = WMS * WMS * SMU   # 64
DIM_MAIN: int = 64           # first chunk  (exact power of 2)
DIM_TAIL: int = 16           # second chunk (HD - DIM_MAIN = 16, exact power of 2)
assert DIM_MAIN + DIM_TAIL == HD

DTYPE   = torch.bfloat16
DEVICE  = 'cuda'
WARMUP  = 30;  REPEATS = 200

gen = QwenDescriptor(NH, RWS, SMS)


# ─────────────────────────────────────────────────────────────────────────────
#  Head-split kernel
#  Splits head_dim into [0:DIM_MAIN] + [DIM_MAIN:HD]
#  Both chunks are exact powers of 2 → no padding lanes wasted.
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def _head_split_kernel(
    META_INFO,
    query_ptr, key_ptr, value_ptr, output_ptr,
    sm_scale,
    MW: tl.constexpr, S: tl.constexpr, SMU: tl.constexpr,
    DIM: tl.constexpr,          # actual head_dim = 80
    DIM_MAIN: tl.constexpr,     # 64
    DIM_TAIL: tl.constexpr,     # 16
    NUM_HEADS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    INFO_LEN: tl.constexpr = 7
    head_offset = tl.program_id(0)
    pid_offset  = tl.program_id(1)

    # ── descriptor routing (identical to original) ────────────────────────────
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
        cur_base_ptr = tl.sum(tl.load(META_INFO+tl.arange(0,1)+cur_base_offsets+1+INFO_LEN*b+1))
        cur_base_H   = tl.sum(tl.load(META_INFO+tl.arange(0,1)+cur_base_offsets+1+INFO_LEN*b+2))
        cur_base_W   = tl.sum(tl.load(META_INFO+tl.arange(0,1)+cur_base_offsets+1+INFO_LEN*b+3))
        cur_region_W = tl.sum(tl.load(META_INFO+tl.arange(0,1)+cur_base_offsets+1+INFO_LEN*b+7))
        win_tokens = cur_base_H * cur_base_W * SMU
        n_ctx_W = cur_region_W // cur_base_W
        ctx_h = ctx // n_ctx_W; ctx_w = ctx % n_ctx_W
        h_off = ctx_h * cur_base_H; w_off = ctx_w * cur_base_W
        h_base = cur_base_ptr // MW; w_base = cur_base_ptr % MW

        # ── dimension offset vectors (split) ─────────────────────────────────
        offs_d_main = tl.arange(0, DIM_MAIN)           # [0..63]
        offs_d_tail = tl.arange(0, DIM_TAIL)           # [0..15]

        # ── Q block (BLOCK_M rows) ────────────────────────────────────────────
        q_block_start = cur_blocks_offset_in_ctx * BLOCK_M
        offs_q = tl.arange(0, BLOCK_M) + q_block_start
        q_mc = offs_q // SMU; q_sub = offs_q % SMU
        q_mr = q_mc // cur_base_W; q_mcl = q_mc % cur_base_W
        q_merged = (h_base + h_off + q_mr) * MW + (w_base + w_off + q_mcl)
        q_abs = q_merged * SMU + q_sub

        q_tok_valid = offs_q < win_tokens   # [BLOCK_M]

        # base pointer for each Q token: query_ptr + abs * NH * DIM + head * DIM
        q_base = query_ptr + q_abs * NUM_HEADS * DIM + head_offset * DIM  # [BLOCK_M]

        # load Q split: [BLOCK_M, DIM_MAIN] and [BLOCK_M, DIM_TAIL]
        q_main = tl.load(
            q_base[:, None] + offs_d_main[None, :],
            mask=q_tok_valid[:, None], other=0.0)   # [BM, 64]
        q_tail = tl.load(
            q_base[:, None] + DIM_MAIN + offs_d_tail[None, :],
            mask=q_tok_valid[:, None], other=0.0)   # [BM, 16]

        # ── online softmax state ──────────────────────────────────────────────
        m_i     = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i     = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc_main = tl.zeros([BLOCK_M, DIM_MAIN], dtype=tl.float32)  # [BM, 64]
        acc_tail = tl.zeros([BLOCK_M, DIM_TAIL], dtype=tl.float32)  # [BM, 16]
        qk_scale = sm_scale * 1.44269504

        # ── KV loop ───────────────────────────────────────────────────────────
        n_kv_blocks = tl.cdiv(win_tokens, BLOCK_N)
        for kv_blk in range(n_kv_blocks):
            offs_kv = tl.arange(0, BLOCK_N) + kv_blk * BLOCK_N
            kv_mc = offs_kv // SMU; kv_sub = offs_kv % SMU
            kv_mr = kv_mc // cur_base_W; kv_mcl = kv_mc % cur_base_W
            kv_merged = (h_base+h_off+kv_mr)*MW + (w_base+w_off+kv_mcl)
            kv_abs = kv_merged * SMU + kv_sub

            kv_tok_valid = offs_kv < win_tokens   # [BLOCK_N]
            kv_base = key_ptr + kv_abs * NUM_HEADS * DIM + head_offset * DIM   # [BN]

            # load K split
            k_main = tl.load(
                kv_base[:, None] + offs_d_main[None, :],
                mask=kv_tok_valid[:, None], other=0.0)   # [BN, 64]
            k_tail = tl.load(
                kv_base[:, None] + DIM_MAIN + offs_d_tail[None, :],
                mask=kv_tok_valid[:, None], other=0.0)   # [BN, 16]

            # QK score: dot over full dim = main + tail parts
            qk = tl.dot(q_main, tl.trans(k_main))            # [BM, BN], fp32
            qk = tl.dot(q_tail, tl.trans(k_tail), acc=qk)    # accumulate tail
            qk = qk * qk_scale
            qk = tl.where(kv_tok_valid[None, :], qk, float("-inf"))

            # online softmax update
            m_j = tl.maximum(m_i, tl.max(qk, axis=1))
            p   = tl.math.exp2(qk - m_j[:, None])             # [BM, BN]
            l_j = tl.sum(p, axis=1)
            alpha = tl.math.exp2(m_i - m_j)
            m_i = m_j
            l_i = l_i * alpha + l_j

            # load V split
            v_base = value_ptr + kv_abs * NUM_HEADS * DIM + head_offset * DIM
            v_main = tl.load(
                v_base[:, None] + offs_d_main[None, :],
                mask=kv_tok_valid[:, None], other=0.0)   # [BN, 64]
            v_tail = tl.load(
                v_base[:, None] + DIM_MAIN + offs_d_tail[None, :],
                mask=kv_tok_valid[:, None], other=0.0)   # [BN, 16]

            p_bf16 = p.to(v_main.dtype)
            # accumulate output split
            acc_main = acc_main * alpha[:, None] + tl.dot(p_bf16, v_main)  # [BM,64]
            acc_tail = acc_tail * alpha[:, None] + tl.dot(p_bf16, v_tail)  # [BM,16]

        # normalize
        acc_main = acc_main / l_i[:, None]
        acc_tail = acc_tail / l_i[:, None]

        # ── write output ──────────────────────────────────────────────────────
        o_base = output_ptr + q_abs * NUM_HEADS * DIM + head_offset * DIM
        tl.store(
            o_base[:, None] + offs_d_main[None, :],
            acc_main.to(output_ptr.dtype.element_ty),
            mask=q_tok_valid[:, None])
        tl.store(
            o_base[:, None] + DIM_MAIN + offs_d_tail[None, :],
            acc_tail.to(output_ptr.dtype.element_ty),
            mask=q_tok_valid[:, None])


def run_head_split(q, k, v, meta, MW, smu, BM, BN, num_warps, num_stages, n_wins):
    S, NH, D = q.shape
    out = torch.empty_like(q)
    pbk = n_wins * math.ceil(WIN_TOK / BM)
    _head_split_kernel[(NH, pbk)](
        meta, q, k, v, out, D**-0.5,
        MW=MW, S=S, SMU=smu,
        DIM=D, DIM_MAIN=DIM_MAIN, DIM_TAIL=DIM_TAIL,
        NUM_HEADS=NH, BLOCK_M=BM, BLOCK_N=BN,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out


# ── reference: current GUIDE (PAD_DIM=128) ───────────────────────────────────
def run_guide_pad128(q, k, v, meta, MW, smu, BM, BN, nw, ns, n_wins):
    from kernel_triton.qwen_window_attention.triton_ops_qwen import (
        qwen_window_attention_kernel
    )
    S, NH, D = q.shape
    PAD_DIM = triton.next_power_of_2(D)
    out = torch.empty_like(q)
    pbk = n_wins * math.ceil(WIN_TOK / BM)
    # use original kernel with BLOCK_SIZE=BM (BN not independently controllable in orig)
    qwen_window_attention_kernel[(NH, pbk)](
        meta, q, k, v, out, D**-0.5,
        MW=MW, S=S, SMU=smu, DIM=D, PAD_DIM=PAD_DIM, NUM_HEADS=NH,
        BLOCK_SIZE=BM, IS_SEQ_LAYOUT=True,
        num_warps=nw, num_stages=ns,
    )
    return out


# ── correctness check ─────────────────────────────────────────────────────────
def check_correct(res=448):
    H = W = res // 14; S = H * W; MW = W // SMS
    n_wins = (H // SMS // WMS) * (W // SMS // WMS)
    q = torch.randn(S, NH, HD, dtype=DTYPE, device=DEVICE)
    k = torch.randn(S, NH, HD, dtype=DTYPE, device=DEVICE)
    v = torch.randn(S, NH, HD, dtype=DTYPE, device=DEVICE)

    meta64 = gen.build(H, W, 64, device=DEVICE)
    ref = qwen_window_attention(q, k, v, meta64, MW, SMU, gen.per_batch_blocks(H, W, 64),
                                BLOCK_SIZE=64, seq_layout=True, num_warps=4, num_stages=2)

    print(f'  {"BM":>4} {"BN":>4}  split_result  vs_ref')
    for BM, BN in [(64, 64), (64, 32), (32, 64), (32, 32)]:
        if WIN_TOK % BN != 0: continue
        meta = gen.build(H, W, BM, device=DEVICE)
        out  = run_head_split(q, k, v, meta, MW, SMU, BM, BN, 4, 2, n_wins)
        rel  = (out - ref).abs().max() / (ref.abs().max() + 1e-6)
        status = 'PASS' if rel < 1e-2 else f'FAIL(rel={rel:.2e})'
        print(f'  {BM:>4} {BN:>4}  {status:<14}  rel={rel:.2e}')


# ── timing util ───────────────────────────────────────────────────────────────
def timed(fn, warmup, repeats):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(repeats): fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / repeats


CONFIGS = [
    # (BM, BN, warps, stages)
    (64, 64, 4, 2),
    (64, 32, 4, 2),
    (64, 32, 2, 2),
    (32, 64, 4, 2),
    (32, 32, 1, 2),
    (32, 32, 4, 2),
]
RESOLUTIONS = [448, 896, 1344, 1792]


def bench_res(res):
    H = W = res // 14; S = H * W; MW = W // SMS
    n_wins = (H // SMS // WMS) * (W // SMS // WMS)
    q = torch.randn(S, NH, HD, dtype=DTYPE, device=DEVICE)
    k = torch.randn(S, NH, HD, dtype=DTYPE, device=DEVICE)
    v = torch.randn(S, NH, HD, dtype=DTYPE, device=DEVICE)

    # FA2 baseline
    cu = torch.arange(0, (n_wins+1)*WIN_TOK, WIN_TOK, dtype=torch.int32, device=DEVICE)
    def run_fa2():
        flash_attn_varlen_func(q, k, v, cu, cu,
                               max_seqlen_q=WIN_TOK, max_seqlen_k=WIN_TOK,
                               dropout_p=0.0, causal=False)
    t_fa2 = timed(run_fa2, WARMUP, REPEATS)

    results = []
    for BM, BN, nw, ns in CONFIGS:
        if WIN_TOK % BN != 0: continue
        meta = gen.build(H, W, BM, device=DEVICE)

        def mk_split(BM=BM, BN=BN, nw=nw, ns=ns, meta=meta):
            return lambda: run_head_split(q, k, v, meta, MW, SMU, BM, BN, nw, ns, n_wins)
        def mk_pad(BM=BM, nw=nw, ns=ns, meta=meta):
            return lambda: run_guide_pad128(q, k, v, meta, MW, SMU, BM, BM, nw, ns, n_wins)

        fn_s = mk_split(); fn_p = mk_pad()
        for _ in range(WARMUP): fn_s(); fn_p()
        torch.cuda.synchronize()
        t_split = timed(fn_s, WARMUP, REPEATS)
        t_pad   = timed(fn_p, WARMUP, REPEATS)
        results.append({
            'BM': BM, 'BN': BN, 'nw': nw, 'ns': ns,
            't_split': t_split, 't_pad': t_pad,
            'split_vs_fa2':  t_split / t_fa2,
            'pad_vs_fa2':    t_pad   / t_fa2,
            'split_vs_pad':  t_pad   / t_split,  # >1 means split is faster
        })
    return t_fa2, results


def main():
    print(f'Device    : {torch.cuda.get_device_name(0)}')
    print(f'Triton    : {triton.__version__}')
    print(f'HD={HD}  DIM_MAIN={DIM_MAIN}  DIM_TAIL={DIM_TAIL}  '
          f'PAD_old=128  PAD_new={DIM_MAIN}+{DIM_TAIL}')
    print(f'BW reduction analytic: {HD}/128 = {HD/128*100:.1f}% of original '
          f'({(1-HD/128)*100:.1f}% saved)')
    print(f'WIN_TOK={WIN_TOK}  WARMUP={WARMUP}  REPEATS={REPEATS}')
    print()

    print('Correctness check (448px):')
    check_correct(448)
    print()

    all_res = {}
    for res in RESOLUTIONS:
        print(f'Benchmarking {res}px...', flush=True)
        t_fa2, results = bench_res(res)
        all_res[res] = (t_fa2, results)
        best_s = min(results, key=lambda x: x['t_split'])
        best_p = min(results, key=lambda x: x['t_pad'])
        print(f'  FA2        = {t_fa2:.4f}ms')
        print(f'  pad128 best (BM={best_p["BM"]},BN={best_p["BM"]},w={best_p["nw"]}) '
              f'= {best_p["t_pad"]:.4f}ms ({best_p["pad_vs_fa2"]:.3f}×FA2)')
        print(f'  split  best (BM={best_s["BM"]},BN={best_s["BN"]},w={best_s["nw"]}) '
              f'= {best_s["t_split"]:.4f}ms ({best_s["split_vs_fa2"]:.3f}×FA2)  '
              f'[{best_s["split_vs_pad"]:.3f}×pad128]')

    # ── per-config table ───────────────────────────────────────────────────────
    print()
    print('=' * 100)
    print('  Latency (ms): pad128 / split  — and split speedup over pad128')
    print('=' * 100)
    hdr = f'  {"BM":>4}{"BN":>4}{"w":>3}  '
    for res in RESOLUTIONS:
        hdr += f'  {res:>4}px[pad/split(su)]'
    print(hdr)
    print('  ' + '-' * 100)
    for BM, BN, nw, ns in CONFIGS:
        if WIN_TOK % BN != 0: continue
        row = f'  {BM:>4}{BN:>4}{nw:>3}  '
        for res in RESOLUTIONS:
            _, results = all_res[res]
            r = next(x for x in results if x['BM']==BM and x['BN']==BN and x['nw']==nw)
            row += f'  {r["t_pad"]:.4f}/{r["t_split"]:.4f}({r["split_vs_pad"]:.3f}×)'
        print(row)

    # ── summary: best of each vs FA2 ──────────────────────────────────────────
    print()
    print('=' * 80)
    print('  BEST per resolution vs FA2')
    print('  (split speedup = pad128_best / split_best)')
    print('=' * 80)
    print(f'  {"res":>6}  {"FA2":>8}  {"pad128_best":>12}  {"split_best":>12}  '
          f'{"split/FA2":>10}  {"split/pad128":>12}')
    print('  ' + '-' * 75)
    for res in RESOLUTIONS:
        t_fa2, results = all_res[res]
        best_s = min(results, key=lambda x: x['t_split'])
        best_p = min(results, key=lambda x: x['t_pad'])
        print(f'  {res:>4}px  {t_fa2:>6.4f}ms  '
              f'{best_p["t_pad"]:>8.4f}ms({best_p["pad_vs_fa2"]:.3f}×)  '
              f'{best_s["t_split"]:>8.4f}ms({best_s["split_vs_fa2"]:.3f}×)  '
              f'{best_s["split_vs_fa2"]:>8.3f}×FA2  '
              f'{best_p["t_pad"]/best_s["t_split"]:>8.3f}×pad128')

    print()
    print('Analytic BW model check:')
    print(f'  If purely memory-bound: expected speedup = 128/80 = {128/80:.3f}×')
    print(f'  (actual results above show how close we get to that bound)')


if __name__ == '__main__':
    main()
