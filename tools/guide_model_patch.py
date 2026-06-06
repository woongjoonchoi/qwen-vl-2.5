"""
GUIDE vision encoder patch for Qwen2.5-VL.

Replaces model.visual.forward with a version that:
  - skips window_index materialization (no hidden/RoPE reorder)
  - uses Triton descriptor window attention for window layers
  - keeps FA2 (or whatever _attn_implementation is set) for full-attention layers
  - skips reverse_restore (output is row-major, same logical content as baseline)

Usage:
    from tools.guide_model_patch import patch_model_with_guide
    patch_model_with_guide(model)   # in-place, reversible via unpatch_model
    # now model.generate(...) uses the GUIDE path
"""
import types
import torch
import torch.nn.functional as F

from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    apply_rotary_pos_emb_vision,
)

_ORIG_FORWARD_ATTR = "_guide_orig_visual_forward"


def _build_guide_forward(model):
    """Return a bound forward function for model.visual that uses GUIDE."""
    import sys
    sys.path.insert(0, "/workspace")
    sys.path.insert(0, "/workspace/tools")
    from tools.qwen_descriptor import QwenDescriptor, get_per_batch_blocks
    from kernel_triton.qwen_window_attention.triton_ops_qwen import qwen_window_attention

    vc = model.config.vision_config
    num_heads = vc.num_heads
    head_dim  = vc.hidden_size // num_heads
    rws       = vc.window_size // vc.patch_size
    sms       = vc.spatial_merge_size
    smu       = sms ** 2
    fullatt   = set(vc.fullatt_block_indexes)

    gen = QwenDescriptor(num_heads, rws, sms)

    # descriptor cache keyed by (H, W, BLOCK_SIZE) — config varies by resolution
    _desc_cache: dict = {}

    def _kernel_cfg(S_raw: int):
        """Resolution-adaptive kernel config.
        Sweep result (H200, window_size=64tok):
          S<=4096  (<=896px ): BS=64, warps=4, stages=2
          S> 4096  (>896px  ): BS=32, warps=1, stages=2
        BS=32 beats BS=64 at high-res because n_kv_blocks=2 enables
        better software-pipeline utilisation vs n_kv_blocks=1 (BS=64).
        """
        if S_raw <= 4096:
            return 64, 4, 2
        else:
            return 32, 1, 2

    def guide_visual_forward(self, hidden_states: torch.Tensor,
                             grid_thw: torch.Tensor, **kwargs) -> torch.Tensor:
        device = hidden_states.device

        # ① patch embed
        hidden = self.patch_embed(hidden_states)   # [S_raw, C]

        # ② RoPE — row-major (no window_index gather)
        rope = self.rot_pos_emb(grid_thw)
        emb  = torch.cat((rope, rope), dim=-1)
        pos_emb = (emb.cos(), emb.sin())

        # ③ cu_seqlens for full-attention layers (per image boundaries)
        cu_seqlens = F.pad(
            (grid_thw[:, 1] * grid_thw[:, 2] * grid_thw[:, 0])
            .cumsum(0, dtype=torch.int32), (1, 0), value=0)

        # ④ descriptor — build once per (H, W, BLOCK_SIZE), reuse across layers
        T, H, W = grid_thw[0].tolist()
        S_raw   = int(H * W)
        BLOCK_SIZE, NUM_WARPS, NUM_STAGES = _kernel_cfg(S_raw)
        cache_key = (H, W, BLOCK_SIZE)
        if cache_key not in _desc_cache:
            meta = gen.build(H, W, BLOCK_SIZE, device=device)
            pbk  = gen.per_batch_blocks(H, W, BLOCK_SIZE)
            MW   = W // sms
            _desc_cache[cache_key] = (meta, pbk, MW)
        guide_meta, guide_pbk, MW = _desc_cache[cache_key]
        guide_meta = guide_meta.to(device)

        # ⑤ block loop
        for li, blk in enumerate(self.blocks):
            normed  = blk.norm1(hidden)
            seq_len, C = normed.shape

            if li in fullatt:
                # full attention — unchanged (FA2 / SDPA / eager per config)
                attn_out = blk.attn(normed, cu_seqlens=cu_seqlens,
                                    position_embeddings=pos_emb, **kwargs)
            else:
                # window attention — GUIDE Triton descriptor kernel
                # qkv3: [S, 3, nh, hd]  contiguous — no permute needed
                qkv3 = blk.attn.qkv(normed).reshape(seq_len, 3, num_heads, head_dim)
                cos, sin = pos_emb
                # apply_rotary_pos_emb_vision returns new contiguous [S, nh, hd] tensors
                q, k = apply_rotary_pos_emb_vision(qkv3[:, 0], qkv3[:, 1], cos, sin)
                # v: one contiguous copy [S, nh, hd] (strides from qkv3 slice are non-natural)
                v = qkv3[:, 2].contiguous()
                # seq_layout=True: kernel reads [S, nh, hd] directly — no transpose/unsqueeze
                out = qwen_window_attention(
                    q, k, v, guide_meta, MW, smu, guide_pbk,
                    BLOCK_SIZE=BLOCK_SIZE, seq_layout=True,
                    num_warps=NUM_WARPS, num_stages=NUM_STAGES)  # adaptive config
                # out: [S, nh, hd] contiguous — reshape is a view, no copy
                attn_out = out.reshape(seq_len, C)
                attn_out = blk.attn.proj(attn_out)

            hidden = hidden + attn_out
            hidden = hidden + blk.mlp(blk.norm2(hidden))

        # ⑥ merger (PatchMerger) on row-major hidden — no reverse_restore
        hidden = self.merger(hidden)
        return hidden

    return guide_visual_forward


def patch_model_with_guide(model) -> None:
    """In-place: replace model.visual.forward with the GUIDE Triton path."""
    vis = model.visual
    if hasattr(vis, _ORIG_FORWARD_ATTR):
        return   # already patched
    setattr(vis, _ORIG_FORWARD_ATTR, vis.forward)
    vis.forward = types.MethodType(_build_guide_forward(model), vis)


def unpatch_model(model) -> None:
    """Restore original model.visual.forward."""
    vis = model.visual
    orig = getattr(vis, _ORIG_FORWARD_ATTR, None)
    if orig is not None:
        vis.forward = orig
        delattr(vis, _ORIG_FORWARD_ATTR)
