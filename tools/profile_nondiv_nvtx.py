#!/usr/bin/env python3
"""
FA2 vs GUIDE VE breakdown with NVTX annotations for nsys profiling.

Usage:
  nsys profile --trace=cuda,nvtx --capture-range=cudaProfilerApi \
    --output=/workspace/output/nsys_nondiv/fa2_guide \
    docker run ... python3 tools/profile_nondiv_nvtx.py \
      --mode fa2|guide \
      --image-path /workspace/data/raw/docvqa/images/pybv0228_81.png \
      --warmup 5 --profile-iters 10
"""
import argparse, sys
from pathlib import Path

sys.path.insert(0, "/workspace")
sys.path.insert(0, "/workspace/tools")

import torch
import torch.nn.functional as F
import torch.cuda.nvtx as nvtx


# ── helpers ────────────────────────────────────────────────────────────────────
def cudaProfilerStart():
    torch.cuda.cudart().cudaProfilerStart()

def cudaProfilerStop():
    torch.cuda.cudart().cudaProfilerStop()


# ── load model ────────────────────────────────────────────────────────────────
def load_fa2(model_path):
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    proc  = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cuda",
        local_files_only=True, attn_implementation="flash_attention_2")
    model.eval()
    return model, proc


def load_guide(model_path):
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from tools.guide_model_patch import patch_model_with_guide
    proc  = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cuda",
        local_files_only=True, attn_implementation="flash_attention_2")
    patch_model_with_guide(model)
    model.eval()
    return model, proc


# ── FA2 instrumented forward ──────────────────────────────────────────────────
def fa2_forward_nvtx(vis, pv, gth):
    import types
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        apply_rotary_pos_emb_vision)

    prefix = "FA2"
    fullatt = set(vis.fullatt_block_indexes)

    # patch_embed
    nvtx.range_push(f"{prefix}.patch_embed")
    hidden = vis.patch_embed(pv)
    nvtx.range_pop()

    rotary_pos_emb = vis.rot_pos_emb(gth)

    # get_window_index
    nvtx.range_push(f"{prefix}.get_window_index")
    window_index, cu_window_seqlens = vis.get_window_index(gth)
    cu_window_seqlens = torch.tensor(cu_window_seqlens,
                                     device=hidden.device, dtype=torch.int32)
    cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)
    nvtx.range_pop()

    seq_len, _ = hidden.size()

    # scatter (partition)
    nvtx.range_push(f"{prefix}.scatter_hidden_rope")
    hidden = hidden.reshape(seq_len // vis.spatial_merge_unit,
                            vis.spatial_merge_unit, -1)
    hidden = hidden[window_index, :, :]
    hidden = hidden.reshape(seq_len, -1)
    rotary_pos_emb = rotary_pos_emb.reshape(
        seq_len // vis.spatial_merge_unit, vis.spatial_merge_unit, -1)
    rotary_pos_emb = rotary_pos_emb[window_index, :, :]
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    nvtx.range_pop()

    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    pos_emb = (emb.cos(), emb.sin())

    cu_seqlens = torch.repeat_interleave(
        gth[:, 1] * gth[:, 2], gth[:, 0]).cumsum(dim=0, dtype=torch.int32)
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    # blocks
    for li, blk in enumerate(vis.blocks):
        is_full = li in fullatt
        label = "full_attn" if is_full else "window_attn"
        nvtx.range_push(f"{prefix}.{label}.block{li:02d}")
        cu_now = cu_seqlens if is_full else cu_window_seqlens
        hidden = blk(hidden, cu_seqlens=cu_now,
                     position_embeddings=pos_emb)
        nvtx.range_pop()

    # merger
    nvtx.range_push(f"{prefix}.merger")
    hidden = vis.merger(hidden)
    nvtx.range_pop()

    # reverse scatter
    nvtx.range_push(f"{prefix}.reverse_scatter")
    reverse_indices = torch.argsort(window_index)
    hidden = hidden[reverse_indices, :]
    nvtx.range_pop()

    return hidden


# ── GUIDE instrumented forward ────────────────────────────────────────────────
def guide_forward_nvtx(model, pv, gth):
    from tools.qwen_descriptor import QwenDescriptor
    from kernel_triton.qwen_window_attention.triton_ops_qwen import qwen_window_attention
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        apply_rotary_pos_emb_vision)

    vis = model.visual
    vc  = model.config.vision_config
    num_heads = vc.num_heads
    head_dim  = vc.hidden_size // num_heads
    rws       = vc.window_size // vc.patch_size
    sms       = vc.spatial_merge_size
    smu       = sms ** 2
    fullatt   = set(vc.fullatt_block_indexes)
    BLOCK_SIZE = 64; NUM_WARPS = 4; NUM_STAGES = 3
    DEVICE = pv.device

    gen = QwenDescriptor(num_heads, rws, sms)
    prefix = "GUIDE"

    # patch_embed
    nvtx.range_push(f"{prefix}.patch_embed")
    hidden = vis.patch_embed(pv)
    nvtx.range_pop()

    rope = vis.rot_pos_emb(gth)
    emb  = torch.cat((rope, rope), dim=-1)
    pos_emb = (emb.cos(), emb.sin())
    cu_seqlens = F.pad(
        (gth[:, 1] * gth[:, 2] * gth[:, 0]).cumsum(0, dtype=torch.int32),
        (1, 0), value=0)

    # descriptor build
    nvtx.range_push(f"{prefix}.descriptor_build")
    T, H, W = gth[0].tolist()
    meta = gen.build(H, W, BLOCK_SIZE, device=DEVICE)
    pbk  = gen.per_batch_blocks(H, W, BLOCK_SIZE)
    MW2  = W // sms
    nvtx.range_pop()

    # blocks
    for li, blk in enumerate(vis.blocks):
        is_full = li in fullatt
        label = "full_attn" if is_full else "window_attn"
        nvtx.range_push(f"{prefix}.{label}.block{li:02d}")

        normed = blk.norm1(hidden)
        seq_len, C = normed.shape

        if is_full:
            attn_out = blk.attn(normed, cu_seqlens=cu_seqlens,
                                position_embeddings=pos_emb)
        else:
            qkv = (blk.attn.qkv(normed)
                   .reshape(seq_len, 3, num_heads, head_dim)
                   .permute(1, 0, 2, 3))
            q, k, v = qkv.unbind(0)
            q, k = apply_rotary_pos_emb_vision(q, k, pos_emb[0], pos_emb[1])
            q = q.transpose(0,1).unsqueeze(0).contiguous()
            k = k.transpose(0,1).unsqueeze(0).contiguous()
            v = v.transpose(0,1).unsqueeze(0).contiguous()
            out = qwen_window_attention(
                q, k, v, meta, MW2, smu, pbk,
                BLOCK_SIZE=BLOCK_SIZE, seq_layout=False,
                num_warps=NUM_WARPS, num_stages=NUM_STAGES)
            attn_out = out[0].transpose(0,1).reshape(seq_len, C)
            attn_out = blk.attn.proj(attn_out)

        hidden = hidden + attn_out
        hidden = hidden + blk.mlp(blk.norm2(hidden))
        nvtx.range_pop()

    # merger
    nvtx.range_push(f"{prefix}.merger")
    hidden = vis.merger(hidden)
    nvtx.range_pop()

    return hidden


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["fa2","guide","both"])
    ap.add_argument("--model-path", default="/workspace/models/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--image-path", required=True)
    ap.add_argument("--warmup",        type=int, default=5)
    ap.add_argument("--profile-iters", type=int, default=10)
    args = ap.parse_args()

    from PIL import Image
    from qwen_vl_utils import process_vision_info

    modes = ["fa2","guide"] if args.mode == "both" else [args.mode]

    for mode in modes:
        print(f"\n[{mode.upper()}] loading model...", flush=True)
        if mode == "fa2":
            model, proc = load_fa2(args.model_path)
        else:
            model, proc = load_guide(args.model_path)

        # prepare inputs
        img = Image.open(args.image_path).convert("RGB")
        msgs = [{"role":"user","content":[
            {"type":"image","image":img},
            {"type":"text","text":"What is shown?"}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        img_in, vid_in = process_vision_info(msgs)
        inputs = proc(text=[text], images=img_in, videos=vid_in,
                      padding=True, return_tensors="pt")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        pv  = inputs["pixel_values"].to(torch.bfloat16)
        gth = inputs["image_grid_thw"]
        T, H, W = gth[0].tolist()
        print(f"  grid H={H} W={W} → merged {H//2}×{W//2}", flush=True)

        forward_fn = fa2_forward_nvtx if mode == "fa2" else \
                     lambda pv, gth: guide_forward_nvtx(model, pv, gth)

        vis = model.visual

        # warmup (no profiling)
        print(f"  warmup {args.warmup} iters...", flush=True)
        with torch.no_grad():
            for _ in range(args.warmup):
                if mode == "fa2":
                    fa2_forward_nvtx(vis, pv, gth)
                else:
                    guide_forward_nvtx(model, pv, gth)
        torch.cuda.synchronize()

        # profiling
        print(f"  profiling {args.profile_iters} iters (cudaProfilerStart)...", flush=True)
        cudaProfilerStart()
        with torch.no_grad():
            for i in range(args.profile_iters):
                nvtx.range_push(f"{mode.upper()}.iter{i:02d}")
                if mode == "fa2":
                    fa2_forward_nvtx(vis, pv, gth)
                else:
                    guide_forward_nvtx(model, pv, gth)
                nvtx.range_pop()
        torch.cuda.synchronize()
        cudaProfilerStop()
        print(f"  done.", flush=True)


if __name__ == "__main__":
    main()
