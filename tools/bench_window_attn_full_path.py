#!/usr/bin/env python3
"""
Full window-attention path benchmark (not kernel-only).

Measures QKV-proj + layout-prep + kernel + out-proj for 28 window-attn blocks,
across three implementations:

  FA2-path:       qkv → reshape+permute → rotary → v.flatten(copy) → FA2-varlen → proj
  GUIDE-BSHD:     qkv → reshape        → rotary → v.contiguous(1copy) → Triton(seq_layout=True) → proj
  GUIDE-ZEROCOPY: qkv → reshape        → rotary → v non-contiguous     → Triton(V_STRIDE_FACTOR=3) → proj

V_STRIDE_FACTOR trick: for qkv3[:,2] the S-stride = 3*H*D instead of H*D.
  Pass this to the kernel so it can read v directly without .contiguous().
"""
import sys
sys.path.insert(0, '/workspace')
sys.path.insert(0, '/workspace/tools')

import torch
import triton
import triton.language as tl
from flash_attn import flash_attn_varlen_func
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_rotary_pos_emb_vision

from tools.qwen_descriptor import QwenDescriptor
from kernel_triton.qwen_window_attention.triton_ops_qwen import (
    qwen_window_attention_kernel, qwen_window_attention
)

# ── config ────────────────────────────────────────────────────────────────────
NH    = 16;  HD = 80;  C = NH * HD   # 1280
SMS   = 2;   SMU = SMS**2
RWS   = 8;   WMS = RWS // SMS        # 4 merged cells per window edge
WIN_TOK = WMS * WMS * SMU            # 64 raw tokens per window
N_WIN_BLOCKS = 28
DTYPE = torch.bfloat16;  DEVICE = 'cuda'
WARMUP = 30;  REPEATS = 200

gen = QwenDescriptor(NH, RWS, SMS)

# ── zero-copy kernel: adds V_STRIDE_FACTOR constexpr ─────────────────────────
@triton.jit
def _qwen_win_attn_vstride(
    META_INFO, query_ptr, key_ptr, value_ptr, output_ptr,
    sm_scale,
    MW: tl.constexpr, S: tl.constexpr, SMU: tl.constexpr,
    DIM: tl.constexpr, PAD_DIM: tl.constexpr, NUM_HEADS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr, IS_SEQ_LAYOUT: tl.constexpr,
    V_STRIDE_FACTOR: tl.constexpr,   # 1 = contiguous, 3 = from qkv3[:,2]
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
        _base_case = tl.sum(tl.load(META_INFO + tl.arange(0, 1) + _base_case_offsets))
        _cur_blocks = tl.sum(tl.load(META_INFO + tl.arange(0, 1) + _base_case_offsets + 1))
        if pid_offset < num_blocks + _cur_blocks and flag == 0:
            flag = 1; cur_base_offsets = _base_case_offsets
            cur_case_blocks = _cur_blocks; prev_case_num_blocks = num_blocks
            cur_base_case = _base_case
        _base_case_offsets = _base_case_offsets + _base_case * INFO_LEN + 2
        num_blocks += _cur_blocks

    cur_context_length = tl.sum(tl.load(META_INFO + tl.arange(0, 1) + cur_base_offsets + 1 + 4))
    cur_pid_offsets  = pid_offset - prev_case_num_blocks
    blocks_per_ctx   = cur_case_blocks // cur_context_length
    cur_ctx_blocks = 0; ctx = -1
    for c in range(cur_context_length):
        if cur_ctx_blocks <= cur_pid_offsets and cur_pid_offsets < cur_ctx_blocks + blocks_per_ctx:
            ctx = c
        cur_ctx_blocks += blocks_per_ctx
    cur_blocks_offset_in_ctx = cur_pid_offsets % blocks_per_ctx

    for b in range(cur_base_case):
        cur_base_ptr = tl.sum(tl.load(META_INFO + tl.arange(0, 1) + cur_base_offsets + 1 + INFO_LEN*b + 1))
        cur_base_H   = tl.sum(tl.load(META_INFO + tl.arange(0, 1) + cur_base_offsets + 1 + INFO_LEN*b + 2))
        cur_base_W   = tl.sum(tl.load(META_INFO + tl.arange(0, 1) + cur_base_offsets + 1 + INFO_LEN*b + 3))
        cur_region_W = tl.sum(tl.load(META_INFO + tl.arange(0, 1) + cur_base_offsets + 1 + INFO_LEN*b + 7))
        win_tokens   = cur_base_H * cur_base_W * SMU
        n_ctx_W = cur_region_W // cur_base_W
        ctx_h = ctx // n_ctx_W; ctx_w = ctx % n_ctx_W
        h_off = ctx_h * cur_base_H; w_off = ctx_w * cur_base_W
        h_base = cur_base_ptr // MW;  w_base = cur_base_ptr % MW

        offs_d = tl.arange(0, PAD_DIM)
        q_block_start = cur_blocks_offset_in_ctx * BLOCK_SIZE
        offs_q = tl.arange(0, BLOCK_SIZE) + q_block_start
        q_mc = offs_q // SMU; q_sub = offs_q % SMU
        q_mr = q_mc // cur_base_W; q_mcl = q_mc % cur_base_W
        q_merged = (h_base + h_off + q_mr) * MW + (w_base + w_off + q_mcl)
        q_abs    = q_merged * SMU + q_sub

        q_valid = (offs_q[:, None] < win_tokens) & (offs_d[None, :] < DIM)
        q_ptrs  = query_ptr + q_abs[:, None] * NUM_HEADS * DIM + head_offset * DIM + offs_d[None, :]
        q = tl.load(q_ptrs, mask=q_valid, other=0.0)

        m_i = tl.zeros([BLOCK_SIZE], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        acc  = tl.zeros([BLOCK_SIZE, PAD_DIM], dtype=tl.float32)
        qk_scale = sm_scale * 1.44269504

        n_kv_blocks = tl.cdiv(win_tokens, BLOCK_SIZE)
        for kv_blk in range(n_kv_blocks):
            offs_kv = tl.arange(0, BLOCK_SIZE) + kv_blk * BLOCK_SIZE
            kv_mc = offs_kv // SMU; kv_sub = offs_kv % SMU
            kv_mr = kv_mc // cur_base_W; kv_mcl = kv_mc % cur_base_W
            kv_merged = (h_base + h_off + kv_mr) * MW + (w_base + w_off + kv_mcl)
            kv_abs    = kv_merged * SMU + kv_sub

            kv_valid = (offs_kv[:, None] < win_tokens) & (offs_d[None, :] < DIM)
            k_ptrs = key_ptr   + kv_abs[:, None] * NUM_HEADS * DIM + head_offset * DIM + offs_d[None, :]
            # v uses V_STRIDE_FACTOR — 1 for contiguous, 3 for qkv3[:,2]
            v_ptrs = value_ptr + kv_abs[:, None] * (V_STRIDE_FACTOR * NUM_HEADS * DIM) + head_offset * DIM + offs_d[None, :]
            k = tl.load(k_ptrs, mask=kv_valid, other=0.0)
            v = tl.load(v_ptrs, mask=kv_valid, other=0.0)

            qk = tl.dot(q, tl.trans(k)) * qk_scale
            qk = tl.where((offs_kv < win_tokens)[None, :], qk, float("-inf"))
            m_j = tl.maximum(m_i, tl.max(qk, axis=1))
            p   = tl.math.exp2(qk - m_j[:, None])
            l_j = tl.sum(p, axis=1)
            alpha = tl.math.exp2(m_i - m_j)
            acc   = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = m_j;  l_i = l_i * alpha + l_j

        acc = acc / l_i[:, None]
        q_mask_out = (offs_q[:, None] < win_tokens) & (offs_d[None, :] < DIM)
        o_ptrs = output_ptr + q_abs[:, None] * NUM_HEADS * DIM + head_offset * DIM + offs_d[None, :]
        tl.store(o_ptrs, acc.to(output_ptr.dtype.element_ty), mask=q_mask_out)


def qwen_window_attn_zerocopy(q, k, v_nc, meta, MW, smu, pbk, BLOCK_SIZE, num_warps, num_stages):
    """v_nc: non-contiguous [S, H, D] from qkv3[:,2] with stride_S = 3*H*D."""
    S, NH, D = q.shape
    PAD_DIM  = triton.next_power_of_2(D)
    out      = torch.empty_like(q)
    # v_stride_S = 3*H*D  → V_STRIDE_FACTOR = 3
    _qwen_win_attn_vstride[(NH, pbk)](
        meta, q, k, v_nc, out, D**-0.5,
        MW=MW, S=S, SMU=smu, DIM=D, PAD_DIM=PAD_DIM,
        NUM_HEADS=NH, BLOCK_SIZE=BLOCK_SIZE,
        IS_SEQ_LAYOUT=True, V_STRIDE_FACTOR=3,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out


# ── fake linear & rotary (for isolated path timing) ───────────────────────────
def make_fake_block(S, device):
    """Returns (qkv_weight, proj_weight, rotary_cos, rotary_sin)."""
    qkv_w  = torch.randn(3*C, C, dtype=DTYPE, device=device)
    proj_w = torch.randn(C, C, dtype=DTYPE, device=device)
    # cos/sin [S, HD] (simplified — half head_dim)
    cos = torch.randn(S, HD, dtype=DTYPE, device=device)
    sin = torch.randn(S, HD, dtype=DTYPE, device=device)
    return qkv_w, proj_w, cos, sin


def timed(fn, warmup, repeats):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(repeats): fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / repeats


# ── per-resolution benchmark ──────────────────────────────────────────────────
def bench(res):
    H = W = res // 14
    S = H * W
    MW = W // SMS

    BS, NW, NS = (64, 4, 2) if S <= 4096 else (32, 1, 2)
    meta = gen.build(H, W, BS, device=DEVICE)
    pbk  = gen.per_batch_blocks(H, W, BS)

    # inputs shared across all runs
    normed   = torch.randn(S, C, dtype=DTYPE, device=DEVICE)
    qkv_w, proj_w, cos, sin = make_fake_block(S, DEVICE)
    pos_emb  = (cos, sin)
    n_wins   = (H//SMS // WMS) * (W//SMS // WMS)
    cu       = torch.arange(0, (n_wins+1)*WIN_TOK, WIN_TOK, dtype=torch.int32, device=DEVICE)

    # ── FA2 path ──────────────────────────────────────────────────────────────
    def run_fa2_path():
        x = torch.nn.functional.linear(normed, qkv_w)          # [S, 3C]
        qkv = x.reshape(S, 3, NH, HD).permute(1, 0, 2, 3)     # [3, S, H, D]
        q, k, v = qkv.unbind(0)                                 # non-cont [S,H,D]
        q, k = apply_rotary_pos_emb_vision(q, k, pos_emb[0], pos_emb[1])
        # flatten: q,k contiguous after rotary; v still non-cont → copy
        out = flash_attn_varlen_func(
            q.reshape(S, NH, HD), k.reshape(S, NH, HD), v.contiguous().reshape(S, NH, HD),
            cu, cu, max_seqlen_q=WIN_TOK, max_seqlen_k=WIN_TOK,
            dropout_p=0.0, causal=False,
        )
        _ = torch.nn.functional.linear(out.reshape(S, C), proj_w)

    # ── GUIDE BSHD (1 copy for v) ─────────────────────────────────────────────
    def run_bshd_1copy():
        x    = torch.nn.functional.linear(normed, qkv_w)       # [S, 3C]
        qkv3 = x.reshape(S, 3, NH, HD)                         # [S, 3, H, D]
        q, k = apply_rotary_pos_emb_vision(qkv3[:, 0], qkv3[:, 1], pos_emb[0], pos_emb[1])
        v    = qkv3[:, 2].contiguous()                          # 1 copy
        out  = qwen_window_attention(q, k, v, meta, MW, SMU, pbk,
                                     BLOCK_SIZE=BS, seq_layout=True,
                                     num_warps=NW, num_stages=NS)
        _ = torch.nn.functional.linear(out.reshape(S, C), proj_w)

    # ── GUIDE BSHD zero-copy (v non-contiguous, V_STRIDE_FACTOR=3) ───────────
    def run_bshd_zerocopy():
        x    = torch.nn.functional.linear(normed, qkv_w)       # [S, 3C]
        qkv3 = x.reshape(S, 3, NH, HD)                         # [S, 3, H, D]
        q, k = apply_rotary_pos_emb_vision(qkv3[:, 0], qkv3[:, 1], pos_emb[0], pos_emb[1])
        v    = qkv3[:, 2]                                       # zero copy — stride_S = 3*H*D
        out  = qwen_window_attn_zerocopy(q, k, v, meta, MW, SMU, pbk,
                                         BLOCK_SIZE=BS, num_warps=NW, num_stages=NS)
        _ = torch.nn.functional.linear(out.reshape(S, C), proj_w)

    # warmup all
    for _ in range(WARMUP):
        run_fa2_path(); run_bshd_1copy(); run_bshd_zerocopy()
    torch.cuda.synchronize()

    t_fa2  = timed(run_fa2_path,       WARMUP, REPEATS)
    t_1cp  = timed(run_bshd_1copy,     WARMUP, REPEATS)
    t_0cp  = timed(run_bshd_zerocopy,  WARMUP, REPEATS)

    # v copy cost alone
    def v_copy_only():
        x    = torch.nn.functional.linear(normed, qkv_w)
        qkv3 = x.reshape(S, 3, NH, HD)
        _    = qkv3[:, 2].contiguous()
    t_vcopy = timed(v_copy_only, WARMUP, REPEATS)

    return {
        'res': res, 'S': S, 'n_wins': n_wins, 'BS': BS,
        'fa2_ms':   t_fa2,
        'bshd_ms':  t_1cp,
        'zero_ms':  t_0cp,
        'vcopy_ms': t_vcopy,
        'zero_vs_bshd': t_0cp / t_1cp,
        'bshd_vs_fa2':  t_1cp / t_fa2,
        'zero_vs_fa2':  t_0cp / t_fa2,
    }


def main():
    print(f'Device: {torch.cuda.get_device_name(0)}')
    print(f'Measuring 1 window-attn block (QKV-proj + layout + attn + out-proj)')
    print(f'n_win_blocks measured = 1 block; ×{N_WIN_BLOCKS} for full ViT estimate')
    print()
    print('Paths:')
    print('  FA2     : reshape+permute → rotary → v.contiguous() → FA2-varlen → proj')
    print('  BSHD    : reshape         → rotary → v.contiguous() → Triton(seq=T) → proj   [1 copy]')
    print('  ZERO-CP : reshape         → rotary → v (no copy)    → Triton(VSF=3) → proj   [0 copy]')
    print()

    results = []
    for res in [448, 896, 1344, 1792]:
        print(f'  {res}px...', flush=True, end=' ')
        r = bench(res)
        results.append(r)
        print(f'FA2={r["fa2_ms"]:.4f}ms  BSHD={r["bshd_ms"]:.4f}ms  ZERO={r["zero_ms"]:.4f}ms  '
              f'vcopy={r["vcopy_ms"]:.4f}ms')

    print()
    print('='*78)
    print(f'  {"res":>6} | {"FA2":>9} | {"BSHD(1cp)":>10} | {"ZERO-CP":>9} | '
          f'{"v-copy":>7} | {"BSHD/FA2":>9} | {"ZERO/FA2":>9} | {"ZERO/BSHD":>10}')
    print('  '+'-'*78)
    for r in results:
        print(f'  {r["res"]:>4}px | {r["fa2_ms"]:>7.4f}ms | {r["bshd_ms"]:>8.4f}ms | '
              f'{r["zero_ms"]:>7.4f}ms | {r["vcopy_ms"]:>5.4f}ms | '
              f'{r["bshd_vs_fa2"]:>7.3f}x  | {r["zero_vs_fa2"]:>7.3f}x  | {r["zero_vs_bshd"]:>8.3f}x')

    print()
    print('  ×28 blocks estimate (full ViT window-attn segment):')
    print(f'  {"res":>6} | {"FA2 ×28":>10} | {"BSHD ×28":>10} | {"ZERO ×28":>10} | {"v-copy ×28":>12}')
    print('  '+'-'*50)
    for r in results:
        fa28   = r['fa2_ms']  * 28
        b28    = r['bshd_ms'] * 28
        z28    = r['zero_ms'] * 28
        vc28   = r['vcopy_ms']* 28
        print(f'  {r["res"]:>4}px | {fa28:>8.2f}ms | {b28:>8.2f}ms | {z28:>8.2f}ms | {vc28:>10.2f}ms')


if __name__ == '__main__':
    main()
