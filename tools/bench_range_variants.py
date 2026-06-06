#!/usr/bin/env python3
"""
KV-loop range variant benchmark: range vs tl.static_range vs tl.range.

Three kernel variants — identical except line 119 (the KV-tile loop):
  RANGE_MODE=0  range(n_kv_blocks)                  [current]
  RANGE_MODE=1  tl.static_range(N_KV_BLOCKS)        [full unroll at compile time]
  RANGE_MODE=2  tl.range(n_kv_blocks)               [Triton 3.x explicit hints]

N_KV_BLOCKS is passed as tl.constexpr for modes 1&2 (= WIN_TOK // BLOCK_N).
The descriptor routing is identical to bench_block_mn_sweep; only the KV loop changes.

Run:
  docker run --rm --gpus device=0 -v /home/wjchoi/Qwen3-VL:/workspace
    qwen25vl-guide:latest python3 /workspace/tools/bench_range_variants.py
    2>&1 | tee /workspace/results/bench_range_variants.log
"""
import sys, math, itertools
sys.path.insert(0, '/workspace')
sys.path.insert(0, '/workspace/tools')

import torch
import triton
import triton.language as tl
from flash_attn import flash_attn_varlen_func

from tools.qwen_descriptor import QwenDescriptor

# ── check tl.range availability (Triton ≥ 3.x) ───────────────────────────────
_HAS_TL_RANGE = hasattr(tl, 'range')

# ── model config ──────────────────────────────────────────────────────────────
NH = 16; HD = 80; SMS = 2; SMU = 4; RWS = 8; WMS = 4
WIN_TOK = WMS * WMS * SMU        # 64
DTYPE   = torch.bfloat16
DEVICE  = 'cuda'
WARMUP  = 30; REPEATS = 200

gen = QwenDescriptor(NH, RWS, SMS)

# ─────────────────────────────────────────────────────────────────────────────
#  Kernel: RANGE_MODE=0  →  range(n_kv_blocks)          [current baseline]
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def _kv_range_kernel(
    META_INFO, query_ptr, key_ptr, value_ptr, output_ptr,
    sm_scale,
    MW: tl.constexpr, S: tl.constexpr, SMU: tl.constexpr,
    DIM: tl.constexpr, PAD_DIM: tl.constexpr, NUM_HEADS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    N_KV_BLOCKS: tl.constexpr,   # WIN_TOK // BLOCK_N (unused in this mode)
):
    INFO_LEN: tl.constexpr = 7
    head_offset = tl.program_id(0)
    pid_offset  = tl.program_id(1)

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

    for b in range(cur_base_case):
        cur_base_ptr = tl.sum(tl.load(META_INFO + tl.arange(0,1)+cur_base_offsets+1+INFO_LEN*b+1))
        cur_base_H   = tl.sum(tl.load(META_INFO + tl.arange(0,1)+cur_base_offsets+1+INFO_LEN*b+2))
        cur_base_W   = tl.sum(tl.load(META_INFO + tl.arange(0,1)+cur_base_offsets+1+INFO_LEN*b+3))
        cur_region_W = tl.sum(tl.load(META_INFO + tl.arange(0,1)+cur_base_offsets+1+INFO_LEN*b+7))
        win_tokens = cur_base_H * cur_base_W * SMU
        n_ctx_W = cur_region_W // cur_base_W
        ctx_h = ctx // n_ctx_W; ctx_w = ctx % n_ctx_W
        h_off = ctx_h * cur_base_H; w_off = ctx_w * cur_base_W
        h_base = cur_base_ptr // MW; w_base = cur_base_ptr % MW

        offs_d = tl.arange(0, PAD_DIM)
        q_block_start = cur_blocks_offset_in_ctx * BLOCK_M
        offs_q = tl.arange(0, BLOCK_M) + q_block_start
        q_mc = offs_q // SMU; q_sub = offs_q % SMU
        q_mr = q_mc // cur_base_W; q_mcl = q_mc % cur_base_W
        q_merged = (h_base + h_off + q_mr) * MW + (w_base + w_off + q_mcl)
        q_abs = q_merged * SMU + q_sub
        q_valid = (offs_q[:, None] < win_tokens) & (offs_d[None, :] < DIM)
        q_ptrs  = query_ptr + q_abs[:,None]*NUM_HEADS*DIM + head_offset*DIM + offs_d[None,:]
        q = tl.load(q_ptrs, mask=q_valid, other=0.0)

        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc  = tl.zeros([BLOCK_M, PAD_DIM], dtype=tl.float32)
        qk_scale = sm_scale * 1.44269504

        # ── KV LOOP: range ────────────────────────────────────────────────────
        n_kv_blocks = tl.cdiv(win_tokens, BLOCK_N)
        for kv_blk in range(n_kv_blocks):
            offs_kv = tl.arange(0, BLOCK_N) + kv_blk * BLOCK_N
            kv_mc = offs_kv // SMU; kv_sub = offs_kv % SMU
            kv_mr = kv_mc // cur_base_W; kv_mcl = kv_mc % cur_base_W
            kv_merged = (h_base+h_off+kv_mr)*MW + (w_base+w_off+kv_mcl)
            kv_abs = kv_merged * SMU + kv_sub
            kv_valid = (offs_kv[:, None] < win_tokens) & (offs_d[None, :] < DIM)
            k = tl.load(key_ptr   + kv_abs[:,None]*NUM_HEADS*DIM + head_offset*DIM + offs_d[None,:], mask=kv_valid, other=0.0)
            v = tl.load(value_ptr + kv_abs[:,None]*NUM_HEADS*DIM + head_offset*DIM + offs_d[None,:], mask=kv_valid, other=0.0)
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
        tl.store(output_ptr + q_abs[:,None]*NUM_HEADS*DIM + head_offset*DIM + offs_d[None,:],
                 acc.to(output_ptr.dtype.element_ty), mask=q_mask_out)


# ─────────────────────────────────────────────────────────────────────────────
#  Kernel: RANGE_MODE=1  →  tl.static_range(N_KV_BLOCKS)  [fully unrolled]
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def _kv_static_range_kernel(
    META_INFO, query_ptr, key_ptr, value_ptr, output_ptr,
    sm_scale,
    MW: tl.constexpr, S: tl.constexpr, SMU: tl.constexpr,
    DIM: tl.constexpr, PAD_DIM: tl.constexpr, NUM_HEADS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    N_KV_BLOCKS: tl.constexpr,   # compile-time constant = WIN_TOK // BLOCK_N
):
    INFO_LEN: tl.constexpr = 7
    head_offset = tl.program_id(0)
    pid_offset  = tl.program_id(1)

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

    for b in range(cur_base_case):
        cur_base_ptr = tl.sum(tl.load(META_INFO + tl.arange(0,1)+cur_base_offsets+1+INFO_LEN*b+1))
        cur_base_H   = tl.sum(tl.load(META_INFO + tl.arange(0,1)+cur_base_offsets+1+INFO_LEN*b+2))
        cur_base_W   = tl.sum(tl.load(META_INFO + tl.arange(0,1)+cur_base_offsets+1+INFO_LEN*b+3))
        cur_region_W = tl.sum(tl.load(META_INFO + tl.arange(0,1)+cur_base_offsets+1+INFO_LEN*b+7))
        win_tokens = cur_base_H * cur_base_W * SMU
        n_ctx_W = cur_region_W // cur_base_W
        ctx_h = ctx // n_ctx_W; ctx_w = ctx % n_ctx_W
        h_off = ctx_h * cur_base_H; w_off = ctx_w * cur_base_W
        h_base = cur_base_ptr // MW; w_base = cur_base_ptr % MW

        offs_d = tl.arange(0, PAD_DIM)
        q_block_start = cur_blocks_offset_in_ctx * BLOCK_M
        offs_q = tl.arange(0, BLOCK_M) + q_block_start
        q_mc = offs_q // SMU; q_sub = offs_q % SMU
        q_mr = q_mc // cur_base_W; q_mcl = q_mc % cur_base_W
        q_merged = (h_base + h_off + q_mr) * MW + (w_base + w_off + q_mcl)
        q_abs = q_merged * SMU + q_sub
        q_valid = (offs_q[:, None] < win_tokens) & (offs_d[None, :] < DIM)
        q_ptrs  = query_ptr + q_abs[:,None]*NUM_HEADS*DIM + head_offset*DIM + offs_d[None,:]
        q = tl.load(q_ptrs, mask=q_valid, other=0.0)

        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc  = tl.zeros([BLOCK_M, PAD_DIM], dtype=tl.float32)
        qk_scale = sm_scale * 1.44269504

        # ── KV LOOP: tl.static_range ──────────────────────────────────────────
        # N_KV_BLOCKS is constexpr → fully unrolled at PTX level
        for kv_blk in tl.static_range(N_KV_BLOCKS):
            offs_kv = tl.arange(0, BLOCK_N) + kv_blk * BLOCK_N
            kv_mc = offs_kv // SMU; kv_sub = offs_kv % SMU
            kv_mr = kv_mc // cur_base_W; kv_mcl = kv_mc % cur_base_W
            kv_merged = (h_base+h_off+kv_mr)*MW + (w_base+w_off+kv_mcl)
            kv_abs = kv_merged * SMU + kv_sub
            kv_valid = (offs_kv[:, None] < win_tokens) & (offs_d[None, :] < DIM)
            k = tl.load(key_ptr   + kv_abs[:,None]*NUM_HEADS*DIM + head_offset*DIM + offs_d[None,:], mask=kv_valid, other=0.0)
            v = tl.load(value_ptr + kv_abs[:,None]*NUM_HEADS*DIM + head_offset*DIM + offs_d[None,:], mask=kv_valid, other=0.0)
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
        tl.store(output_ptr + q_abs[:,None]*NUM_HEADS*DIM + head_offset*DIM + offs_d[None,:],
                 acc.to(output_ptr.dtype.element_ty), mask=q_mask_out)


# ─────────────────────────────────────────────────────────────────────────────
#  Kernel: RANGE_MODE=2  →  tl.range(n_kv_blocks)  [Triton ≥3.x explicit hints]
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def _kv_tlrange_kernel(
    META_INFO, query_ptr, key_ptr, value_ptr, output_ptr,
    sm_scale,
    MW: tl.constexpr, S: tl.constexpr, SMU: tl.constexpr,
    DIM: tl.constexpr, PAD_DIM: tl.constexpr, NUM_HEADS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    N_KV_BLOCKS: tl.constexpr,
):
    INFO_LEN: tl.constexpr = 7
    head_offset = tl.program_id(0)
    pid_offset  = tl.program_id(1)

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

    for b in range(cur_base_case):
        cur_base_ptr = tl.sum(tl.load(META_INFO + tl.arange(0,1)+cur_base_offsets+1+INFO_LEN*b+1))
        cur_base_H   = tl.sum(tl.load(META_INFO + tl.arange(0,1)+cur_base_offsets+1+INFO_LEN*b+2))
        cur_base_W   = tl.sum(tl.load(META_INFO + tl.arange(0,1)+cur_base_offsets+1+INFO_LEN*b+3))
        cur_region_W = tl.sum(tl.load(META_INFO + tl.arange(0,1)+cur_base_offsets+1+INFO_LEN*b+7))
        win_tokens = cur_base_H * cur_base_W * SMU
        n_ctx_W = cur_region_W // cur_base_W
        ctx_h = ctx // n_ctx_W; ctx_w = ctx % n_ctx_W
        h_off = ctx_h * cur_base_H; w_off = ctx_w * cur_base_W
        h_base = cur_base_ptr // MW; w_base = cur_base_ptr % MW

        offs_d = tl.arange(0, PAD_DIM)
        q_block_start = cur_blocks_offset_in_ctx * BLOCK_M
        offs_q = tl.arange(0, BLOCK_M) + q_block_start
        q_mc = offs_q // SMU; q_sub = offs_q % SMU
        q_mr = q_mc // cur_base_W; q_mcl = q_mc % cur_base_W
        q_merged = (h_base + h_off + q_mr) * MW + (w_base + w_off + q_mcl)
        q_abs = q_merged * SMU + q_sub
        q_valid = (offs_q[:, None] < win_tokens) & (offs_d[None, :] < DIM)
        q_ptrs  = query_ptr + q_abs[:,None]*NUM_HEADS*DIM + head_offset*DIM + offs_d[None,:]
        q = tl.load(q_ptrs, mask=q_valid, other=0.0)

        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc  = tl.zeros([BLOCK_M, PAD_DIM], dtype=tl.float32)
        qk_scale = sm_scale * 1.44269504

        # ── KV LOOP: tl.range  ────────────────────────────────────────────────
        # tl.range available in Triton ≥3.x; falls back to range in wrapper if absent
        n_kv_blocks = tl.cdiv(win_tokens, BLOCK_N)
        for kv_blk in tl.range(n_kv_blocks, num_stages=2):
            offs_kv = tl.arange(0, BLOCK_N) + kv_blk * BLOCK_N
            kv_mc = offs_kv // SMU; kv_sub = offs_kv % SMU
            kv_mr = kv_mc // cur_base_W; kv_mcl = kv_mc % cur_base_W
            kv_merged = (h_base+h_off+kv_mr)*MW + (w_base+w_off+kv_mcl)
            kv_abs = kv_merged * SMU + kv_sub
            kv_valid = (offs_kv[:, None] < win_tokens) & (offs_d[None, :] < DIM)
            k = tl.load(key_ptr   + kv_abs[:,None]*NUM_HEADS*DIM + head_offset*DIM + offs_d[None,:], mask=kv_valid, other=0.0)
            v = tl.load(value_ptr + kv_abs[:,None]*NUM_HEADS*DIM + head_offset*DIM + offs_d[None,:], mask=kv_valid, other=0.0)
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
        tl.store(output_ptr + q_abs[:,None]*NUM_HEADS*DIM + head_offset*DIM + offs_d[None,:],
                 acc.to(output_ptr.dtype.element_ty), mask=q_mask_out)


# ── wrappers ──────────────────────────────────────────────────────────────────
def _run_kernel(kernel_fn, q, k, v, meta, MW, smu, BM, BN, num_warps, num_stages, n_wins):
    S, NH, D = q.shape
    PAD_DIM  = triton.next_power_of_2(D)
    out      = torch.empty_like(q)
    pbk      = n_wins * math.ceil(WIN_TOK / BM)
    N_KV     = WIN_TOK // BN   # constexpr, valid when WIN_TOK divisible by BN
    kernel_fn[(NH, pbk)](
        meta, q, k, v, out, D**-0.5,
        MW=MW, S=S, SMU=smu, DIM=D, PAD_DIM=PAD_DIM, NUM_HEADS=NH,
        BLOCK_M=BM, BLOCK_N=BN, N_KV_BLOCKS=N_KV,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out


def run_range(q, k, v, meta, MW, smu, BM, BN, nw, ns, n_wins):
    return _run_kernel(_kv_range_kernel,  q, k, v, meta, MW, smu, BM, BN, nw, ns, n_wins)

def run_static(q, k, v, meta, MW, smu, BM, BN, nw, ns, n_wins):
    return _run_kernel(_kv_static_range_kernel, q, k, v, meta, MW, smu, BM, BN, nw, ns, n_wins)

def run_tlrange(q, k, v, meta, MW, smu, BM, BN, nw, ns, n_wins):
    return _run_kernel(_kv_tlrange_kernel, q, k, v, meta, MW, smu, BM, BN, nw, ns, n_wins)


# ── correctness check ─────────────────────────────────────────────────────────
def check_correct(res=448):
    H = W = res // 14; S = H * W; MW = W // SMS
    n_wins = (H // SMS // WMS) * (W // SMS // WMS)
    q = torch.randn(S, NH, HD, dtype=DTYPE, device=DEVICE)
    k = torch.randn(S, NH, HD, dtype=DTYPE, device=DEVICE)
    v = torch.randn(S, NH, HD, dtype=DTYPE, device=DEVICE)

    # reference: range kernel with BM=BN=64
    meta64 = gen.build(H, W, 64, device=DEVICE)
    ref = run_range(q, k, v, meta64, MW, SMU, 64, 64, 4, 2, n_wins)

    print(f'  {"BM":>4} {"BN":>4}  range   static  tl.range')
    for BM, BN in [(64, 64), (64, 32), (32, 64), (32, 32)]:
        if WIN_TOK % BN != 0:
            continue
        meta = gen.build(H, W, BM, device=DEVICE)
        def _chk(out):
            rel = (out - ref).abs().max() / (ref.abs().max() + 1e-6)
            return 'PASS' if rel < 1e-2 else f'FAIL({rel:.1e})'
        r_range  = _chk(run_range  (q, k, v, meta, MW, SMU, BM, BN, 4, 2, n_wins))
        r_static = _chk(run_static (q, k, v, meta, MW, SMU, BM, BN, 4, 2, n_wins))
        r_tl     = _chk(run_tlrange(q, k, v, meta, MW, SMU, BM, BN, 4, 2, n_wins))
        print(f'  {BM:>4} {BN:>4}  {r_range:<7} {r_static:<7} {r_tl}')


# ── timing util ───────────────────────────────────────────────────────────────
def timed(fn, warmup, repeats):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(repeats): fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / repeats


# ── sweep configs (optimal from bench_block_mn_sweep) ─────────────────────────
# Only configs where WIN_TOK % BN == 0  (static_range requires exact divisibility)
CONFIGS = [
    # (BM, BN, warps, stages)
    (64, 64, 4, 2),   # baseline: current GUIDE BS=64
    (64, 32, 4, 2),   # best at 1344px
    (64, 32, 2, 2),
    (32, 64, 4, 2),
    (32, 32, 1, 2),   # baseline: current GUIDE BS=32
    (32, 32, 4, 2),
]

RESOLUTIONS = [448, 896, 1344, 1792]


def bench_res(res):
    H = W = res // 14; S = H * W; MW = W // SMS
    n_wins = (H // SMS // WMS) * (W // SMS // WMS)
    q = torch.randn(S, NH, HD, dtype=DTYPE, device=DEVICE)
    k = torch.randn(S, NH, HD, dtype=DTYPE, device=DEVICE)
    v = torch.randn(S, NH, HD, dtype=DTYPE, device=DEVICE)

    # FA2 reference
    cu = torch.arange(0, (n_wins+1)*WIN_TOK, WIN_TOK, dtype=torch.int32, device=DEVICE)
    def run_fa2():
        flash_attn_varlen_func(q, k, v, cu, cu,
                               max_seqlen_q=WIN_TOK, max_seqlen_k=WIN_TOK,
                               dropout_p=0.0, causal=False)
    t_fa2 = timed(run_fa2, WARMUP, REPEATS)

    results = []
    for BM, BN, nw, ns in CONFIGS:
        if WIN_TOK % BN != 0:
            continue
        meta = gen.build(H, W, BM, device=DEVICE)
        N_KV = WIN_TOK // BN

        def mk(fn, BM=BM, BN=BN, nw=nw, ns=ns, meta=meta):
            return lambda: fn(q, k, v, meta, MW, SMU, BM, BN, nw, ns, n_wins)

        fn_r = mk(run_range)
        fn_s = mk(run_static)
        fn_t = mk(run_tlrange)

        # warmup all three together
        for _ in range(WARMUP): fn_r(); fn_s(); fn_t()
        torch.cuda.synchronize()

        t_r = timed(fn_r, WARMUP, REPEATS)
        t_s = timed(fn_s, WARMUP, REPEATS)
        t_t = timed(fn_t, WARMUP, REPEATS)
        results.append({
            'BM': BM, 'BN': BN, 'nw': nw, 'ns': ns, 'N_KV': N_KV,
            't_range': t_r, 't_static': t_s, 't_tlrange': t_t,
            'r_vs_fa2': t_r / t_fa2,
            's_vs_fa2': t_s / t_fa2,
            'tl_vs_fa2': t_t / t_fa2,
            's_vs_r':  t_r / t_s,        # speedup: static over range
            'tl_vs_r': t_r / t_t,        # speedup: tl.range over range
        })
    return t_fa2, results


def main():
    import triton
    print(f'Device    : {torch.cuda.get_device_name(0)}')
    print(f'Triton    : {triton.__version__}')
    print(f'tl.range  : {"available" if _HAS_TL_RANGE else "NOT available (Triton <3.x)"}')
    print(f'WIN_TOK={WIN_TOK}  NH={NH}  HD={HD}  WARMUP={WARMUP}  REPEATS={REPEATS}')
    print()

    if not _HAS_TL_RANGE:
        print('WARNING: tl.range not available — tl_range column will error or be skipped.')
        print()

    print('Correctness check (448px):')
    check_correct(448)
    print()

    all_res = {}
    for res in RESOLUTIONS:
        print(f'Benchmarking {res}px (S={(res//14)**2})...', flush=True)
        t_fa2, results = bench_res(res)
        all_res[res] = (t_fa2, results)

        best_r = min(results, key=lambda x: x['t_range'])
        best_s = min(results, key=lambda x: x['t_static'])
        best_t = min(results, key=lambda x: x['t_tlrange'])
        print(f'  FA2={t_fa2:.4f}ms')
        print(f'  BEST range:        BM={best_r["BM"]},BN={best_r["BN"]},w={best_r["nw"]} '
              f'→ {best_r["t_range"]:.4f}ms ({best_r["r_vs_fa2"]:.3f}×FA2)')
        print(f'  BEST static_range: BM={best_s["BM"]},BN={best_s["BN"]},w={best_s["nw"]} '
              f'→ {best_s["t_static"]:.4f}ms ({best_s["s_vs_fa2"]:.3f}×FA2)')
        print(f'  BEST tl.range:     BM={best_t["BM"]},BN={best_t["BN"]},w={best_t["nw"]} '
              f'→ {best_t["t_tlrange"]:.4f}ms ({best_t["tl_vs_fa2"]:.3f}×FA2)')

    # ── full table ─────────────────────────────────────────────────────────────
    print()
    print('=' * 100)
    print('  Per-config latency (ms) — all variants')
    print('=' * 100)
    hdr = f'  {"BM":>4}{"BN":>4}{"w":>3}{"N_KV":>5}  '
    for res in RESOLUTIONS:
        hdr += f'{res:>4}px[range/static/tl]  '
    print(hdr)
    print('  ' + '-' * 100)
    for BM, BN, nw, ns in CONFIGS:
        if WIN_TOK % BN != 0: continue
        row = f'  {BM:>4}{BN:>4}{nw:>3}{WIN_TOK//BN:>5}  '
        for res in RESOLUTIONS:
            t_fa2, results = all_res[res]
            r = next(x for x in results if x['BM']==BM and x['BN']==BN and x['nw']==nw)
            row += f'{r["t_range"]:.4f}/{r["t_static"]:.4f}/{r["t_tlrange"]:.4f}  '
        print(row)

    print()
    print('=' * 100)
    print('  Speedup over range baseline  (>1.0 = static/tl.range is faster)')
    print('=' * 100)
    hdr2 = f'  {"BM":>4}{"BN":>4}{"w":>3}{"N_KV":>5}  '
    for res in RESOLUTIONS:
        hdr2 += f'{res:>4}px[s/r  tl/r]  '
    print(hdr2)
    print('  ' + '-' * 100)
    for BM, BN, nw, ns in CONFIGS:
        if WIN_TOK % BN != 0: continue
        row = f'  {BM:>4}{BN:>4}{nw:>3}{WIN_TOK//BN:>5}  '
        for res in RESOLUTIONS:
            t_fa2, results = all_res[res]
            r = next(x for x in results if x['BM']==BM and x['BN']==BN and x['nw']==nw)
            row += f'{r["s_vs_r"]:>5.3f}x {r["tl_vs_r"]:>5.3f}x   '
        print(row)

    print()
    print('=' * 100)
    print('  BEST of each variant vs FA2')
    print('=' * 100)
    print('  {:>6}  {:>8}  {:>8}({:>6})  {:>8}({:>6})  {:>8}({:>6})'.format(
          'res', 'FA2', 'range', 'vs_FA2', 'static', 'vs_FA2', 'tl.range', 'vs_FA2'))
    print('  ' + '-' * 80)
    for res in RESOLUTIONS:
        t_fa2, results = all_res[res]
        best_r  = min(results, key=lambda x: x['t_range'])
        best_s  = min(results, key=lambda x: x['t_static'])
        best_tl = min(results, key=lambda x: x['t_tlrange'])
        print(f'  {res:>4}px  {t_fa2:>6.4f}ms  '
              f'{best_r["t_range"]:>6.4f}ms({best_r["r_vs_fa2"]:>5.3f}x)  '
              f'{best_s["t_static"]:>6.4f}ms({best_s["s_vs_fa2"]:>5.3f}x)  '
              f'{best_tl["t_tlrange"]:>6.4f}ms({best_tl["tl_vs_fa2"]:>5.3f}x)')


if __name__ == '__main__':
    main()
