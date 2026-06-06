"""
Qwen2.5-VL GUIDE Triton Window Attention Kernel (Option B: merged-cell grid)

Adapted from sigmetrics2026_swin/kernel_triton fused window attention.
Key changes from Swin:
  1. No relative position bias (RPB) — Qwen uses RoPE before attention.
  2. Q/K/V layout: [1, NUM_HEADS, S, head_dim] (packed) or [S, NUM_HEADS, head_dim].
  3. HEAD_GROUP_NUM = 1 (all heads share the window pattern).
  4. No shifted-window mask for the initial implementation.

Packed-sequence offset model (the crux of the Qwen port):
  Qwen stores each (sms x sms) spatial-merge block as SMU=sms**2 contiguous
  tokens, and window_index is on the merged grid.  A raw patch's packed offset
  is therefore  merged_flat * SMU + sub,  NOT  h*W + w.

  Within a window, raw token t maps to:
    mc      = t // SMU                  # merged cell index within window
    sub     = t %  SMU                  # sub-token inside the merge block
    m_row   = mc // base_W              # merged-local row
    m_col   = mc %  base_W              # merged-local col
    g_row   = h_base + h_off + m_row    # global merged row
    g_col   = w_base + w_off + m_col    # global merged col
    raw_off = (g_row * MW + g_col) * SMU + sub
  All descriptor spatial fields (base_ptr/base_H/base_W/region_H/region_W) are
  in MERGED units; SMU is a config constexpr.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def qwen_window_attention_kernel(
    META_INFO,          # flat int32 descriptor (merged-grid units)
    query_ptr,          # [1, NUM_HEADS, S, DIM] or [S, NUM_HEADS, DIM]
    key_ptr,
    value_ptr,
    output_ptr,
    sm_scale,           # 1/sqrt(head_dim)
    MW: tl.constexpr,           # merged-grid width (region width, merged units)
    S: tl.constexpr,            # total raw tokens (= MH*MW*SMU)
    SMU: tl.constexpr,          # spatial_merge_unit (sms**2)
    DIM: tl.constexpr,          # head_dim
    PAD_DIM: tl.constexpr,      # next power-of-2 >= DIM
    NUM_HEADS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,   # raw query tokens per program
    IS_SEQ_LAYOUT: tl.constexpr,  # True=[S,NH,D], False=[1,NH,S,D]
):
    INFO_LEN: tl.constexpr = 7

    head_offset = tl.program_id(0)
    pid_offset = tl.program_id(1)

    # ── Parse head group ──────────────────────────────────────────────────────
    HEAD_GROUP_NUM = tl.sum(tl.load(META_INFO + tl.arange(0, 1)))
    _head_start_offset = -1
    for _g in range(HEAD_GROUP_NUM):
        grp_ptr = tl.sum(tl.load(META_INFO + tl.arange(0, 1) + 1 + _g))
        _hs = tl.sum(tl.load(META_INFO + tl.arange(0, 1) + grp_ptr + 0))
        _he = tl.sum(tl.load(META_INFO + tl.arange(0, 1) + grp_ptr + 1))
        if head_offset >= _hs and head_offset < _he:
            _head_start_offset = grp_ptr

    case_num = tl.sum(tl.load(META_INFO + tl.arange(0, 1) + _head_start_offset + 2))

    # ── Locate case / context for this pid_offset ─────────────────────────────
    _base_case_offsets = _head_start_offset + 3
    num_blocks = 0
    flag = 0
    cur_base_offsets = -1
    cur_case_blocks = -1
    prev_case_num_blocks = -1
    cur_base_case = -1
    for i in range(case_num):
        _base_case = tl.sum(tl.load(META_INFO + tl.arange(0, 1) + _base_case_offsets))
        _cur_blocks = tl.sum(tl.load(META_INFO + tl.arange(0, 1) + _base_case_offsets + 1))
        if pid_offset < num_blocks + _cur_blocks and flag == 0:
            flag = 1
            cur_base_offsets = _base_case_offsets
            cur_case_blocks = _cur_blocks
            prev_case_num_blocks = num_blocks
            cur_base_case = _base_case
        _base_case_offsets = _base_case_offsets + _base_case * INFO_LEN + 2
        num_blocks += _cur_blocks

    cur_context_length = tl.sum(
        tl.load(META_INFO + tl.arange(0, 1) + cur_base_offsets + 1 + 4))
    cur_pid_offsets = pid_offset - prev_case_num_blocks
    blocks_per_ctx = cur_case_blocks // cur_context_length

    cur_ctx_blocks = 0
    ctx = -1
    for c in range(cur_context_length):
        if cur_ctx_blocks <= cur_pid_offsets and cur_pid_offsets < cur_ctx_blocks + blocks_per_ctx:
            ctx = c
        cur_ctx_blocks += blocks_per_ctx
    cur_blocks_offset_in_ctx = cur_pid_offsets % blocks_per_ctx

    # ── Process each base record ──────────────────────────────────────────────
    for b in range(cur_base_case):
        cur_base_ptr = tl.sum(tl.load(META_INFO + tl.arange(0, 1) + cur_base_offsets + 1 + INFO_LEN * b + 1))
        cur_base_H   = tl.sum(tl.load(META_INFO + tl.arange(0, 1) + cur_base_offsets + 1 + INFO_LEN * b + 2))
        cur_base_W   = tl.sum(tl.load(META_INFO + tl.arange(0, 1) + cur_base_offsets + 1 + INFO_LEN * b + 3))
        cur_region_W = tl.sum(tl.load(META_INFO + tl.arange(0, 1) + cur_base_offsets + 1 + INFO_LEN * b + 7))

        # window token count (raw) and merged base position
        win_tokens = cur_base_H * cur_base_W * SMU
        n_ctx_W = cur_region_W // cur_base_W
        ctx_h = ctx // n_ctx_W
        ctx_w = ctx % n_ctx_W
        h_off = ctx_h * cur_base_H
        w_off = ctx_w * cur_base_W
        # base_ptr is absolute in the full merged grid (width=MW), matching
        # the original Swin kernel convention.
        h_base = cur_base_ptr // MW
        w_base = cur_base_ptr % MW

        offs_d = tl.arange(0, PAD_DIM)

        # ── queries for this block ────────────────────────────────────────────
        q_block_start = cur_blocks_offset_in_ctx * BLOCK_SIZE
        offs_q = tl.arange(0, BLOCK_SIZE) + q_block_start

        q_mc  = offs_q // SMU
        q_sub = offs_q % SMU
        q_mr  = q_mc // cur_base_W
        q_mcl = q_mc % cur_base_W
        q_merged = (h_base + h_off + q_mr) * MW + (w_base + w_off + q_mcl)
        q_abs = q_merged * SMU + q_sub                  # raw packed offset

        q_valid = (offs_q[:, None] < win_tokens) & (offs_d[None, :] < DIM)
        if IS_SEQ_LAYOUT:
            q_ptrs = query_ptr + q_abs[:, None] * NUM_HEADS * DIM + head_offset * DIM + offs_d[None, :]
        else:
            q_ptrs = query_ptr + head_offset * S * DIM + q_abs[:, None] * DIM + offs_d[None, :]
        q = tl.load(q_ptrs, mask=q_valid, other=0.0)

        # ── online softmax over all KV tokens in the window ──────────────────
        m_i = tl.zeros([BLOCK_SIZE], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        acc = tl.zeros([BLOCK_SIZE, PAD_DIM], dtype=tl.float32)
        qk_scale = sm_scale * 1.44269504  # 1/ln(2)

        n_kv_blocks = tl.cdiv(win_tokens, BLOCK_SIZE)
        for kv_blk in range(n_kv_blocks):
            offs_kv = tl.arange(0, BLOCK_SIZE) + kv_blk * BLOCK_SIZE
            kv_mc  = offs_kv // SMU
            kv_sub = offs_kv % SMU
            kv_mr  = kv_mc // cur_base_W
            kv_mcl = kv_mc % cur_base_W
            kv_merged = (h_base + h_off + kv_mr) * MW + (w_base + w_off + kv_mcl)
            kv_abs = kv_merged * SMU + kv_sub

            kv_valid = (offs_kv[:, None] < win_tokens) & (offs_d[None, :] < DIM)
            if IS_SEQ_LAYOUT:
                k_ptrs = key_ptr   + kv_abs[:, None] * NUM_HEADS * DIM + head_offset * DIM + offs_d[None, :]
                v_ptrs = value_ptr + kv_abs[:, None] * NUM_HEADS * DIM + head_offset * DIM + offs_d[None, :]
            else:
                k_ptrs = key_ptr   + head_offset * S * DIM + kv_abs[:, None] * DIM + offs_d[None, :]
                v_ptrs = value_ptr + head_offset * S * DIM + kv_abs[:, None] * DIM + offs_d[None, :]
            k = tl.load(k_ptrs, mask=kv_valid, other=0.0)
            v = tl.load(v_ptrs, mask=kv_valid, other=0.0)

            qk = tl.zeros([BLOCK_SIZE, BLOCK_SIZE], dtype=tl.float32)
            qk = tl.dot(q, tl.trans(k), acc=qk) * qk_scale
            kv_mask = offs_kv < win_tokens
            qk = tl.where(kv_mask[None, :], qk, float("-inf"))

            m_j = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.math.exp2(qk - m_j[:, None])
            l_j = tl.sum(p, axis=1)
            alpha = tl.math.exp2(m_i - m_j)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = m_j
            l_i = l_i * alpha + l_j

        acc = acc / l_i[:, None]

        q_mask_out = (offs_q[:, None] < win_tokens) & (offs_d[None, :] < DIM)
        if IS_SEQ_LAYOUT:
            o_ptrs = output_ptr + q_abs[:, None] * NUM_HEADS * DIM + head_offset * DIM + offs_d[None, :]
        else:
            o_ptrs = output_ptr + head_offset * S * DIM + q_abs[:, None] * DIM + offs_d[None, :]
        tl.store(o_ptrs, acc.to(output_ptr.dtype.element_ty), mask=q_mask_out)


# ─── Python wrapper ───────────────────────────────────────────────────────────

def qwen_window_attention(
    query: torch.Tensor,   # [1, NH, S, D] or [S, NH, D]
    key: torch.Tensor,
    value: torch.Tensor,
    meta: torch.Tensor,    # int32 descriptor (merged-grid units)
    MW: int,               # merged-grid width
    spatial_merge_unit: int,
    per_batch_blocks: int,
    BLOCK_SIZE: int = 32,
    seq_layout: bool = False,
    num_warps: int = 4,
    num_stages: int = 3,
) -> torch.Tensor:
    """Qwen2.5-VL GUIDE window attention forward pass (merged-grid descriptor)."""
    if seq_layout:
        S, NH, D = query.shape
    else:
        _, NH, S, D = query.shape

    PAD_DIM = triton.next_power_of_2(D)
    sm_scale = D ** -0.5
    output = torch.empty_like(query)
    grid = (NH, per_batch_blocks)

    qwen_window_attention_kernel[grid](
        meta, query, key, value, output,
        sm_scale,
        MW=MW, S=S, SMU=spatial_merge_unit,
        DIM=D, PAD_DIM=PAD_DIM,
        NUM_HEADS=NH,
        BLOCK_SIZE=BLOCK_SIZE,
        IS_SEQ_LAYOUT=seq_layout,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output
