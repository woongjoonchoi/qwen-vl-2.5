#!/usr/bin/env python3
"""
Non-divisible grid: FA2 vs GUIDE ViT attention breakdown profiling.

Measures per-operation GPU time for the vision encoder forward pass:
  FA2  : patch_embed | get_window_index | gather_hidden/rope |
         window_attn(×28) | full_attn(×4) | merger | reverse_gather
  GUIDE: patch_embed | descriptor_build |
         guide_window_attn(×28) | full_attn(×4) | merger

Usage:
  python3 tools/profile_nondiv_attn_breakdown.py \
      --model-path /workspace/models/Qwen2.5-VL-3B-Instruct \
      --image-path /workspace/data/raw/docvqa/images/nkbl0226_1.png \
      --warmup 3 --runs 5
"""
import argparse, sys, time
from pathlib import Path

import torch
import torch.nn.functional as F


# ── helpers ────────────────────────────────────────────────────────────────────

def ev():
    e = torch.cuda.Event(enable_timing=True)
    return e

def ms(s, e):
    torch.cuda.synchronize()
    return s.elapsed_time(e)

def record_pair():
    s, e2 = ev(), ev()
    return s, e2


# ── FA2 instrumented forward ───────────────────────────────────────────────────

def make_fa2_timed_forward(vis_module):
    """Return a timed version of vis_module.forward (FA2 path)."""
    import types
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        apply_rotary_pos_emb_vision)

    def timed_forward(self, hidden_states, grid_thw, **kwargs):
        timings = {}
        def rec(name):
            s2, e2 = record_pair()
            s2.record()
            return s2, e2, name

        # patch_embed
        s, e2, n = rec("patch_embed")
        hidden_states = self.patch_embed(hidden_states)
        e2.record(); torch.cuda.synchronize()
        timings[n] = ms(s, e2)

        # rotary
        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        # get_window_index
        s, e2, n = rec("get_window_index")
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        e2.record(); torch.cuda.synchronize()
        timings[n] = ms(s, e2)

        cu_window_seqlens = torch.tensor(
            cu_window_seqlens, device=hidden_states.device, dtype=torch.int32)
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        seq_len, _ = hidden_states.size()

        # gather hidden + rope (the scatter / reorder step)
        s, e2, n = rec("scatter_hidden_rope")
        hidden_states = hidden_states.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        e2.record(); torch.cuda.synchronize()
        timings[n] = ms(s, e2)

        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        # blocks
        fullatt = set(self.fullatt_block_indexes)
        win_times, full_times = [], []
        for li, blk in enumerate(self.blocks):
            s, e2 = record_pair()
            s.record()
            if li in fullatt:
                cu_now = cu_seqlens
            else:
                cu_now = cu_window_seqlens
            hidden_states = blk(hidden_states, cu_seqlens=cu_now,
                                position_embeddings=position_embeddings, **kwargs)
            e2.record(); torch.cuda.synchronize()
            t = ms(s, e2)
            if li in fullatt:
                full_times.append(t)
            else:
                win_times.append(t)

        timings["window_attn_blocks"] = win_times
        timings["full_attn_blocks"]   = full_times

        # merger
        s, e2, n = rec("merger")
        hidden_states = self.merger(hidden_states)
        e2.record(); torch.cuda.synchronize()
        timings[n] = ms(s, e2)

        # reverse scatter
        s, e2, n = rec("reverse_scatter")
        reverse_indices = torch.argsort(window_index)
        hidden_states   = hidden_states[reverse_indices, :]
        e2.record(); torch.cuda.synchronize()
        timings[n] = ms(s, e2)

        self._last_timings = timings
        return hidden_states

    vis_module.forward = types.MethodType(timed_forward, vis_module)


# ── GUIDE instrumented forward ─────────────────────────────────────────────────

def make_guide_timed_forward(model):
    import types
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        apply_rotary_pos_emb_vision)
    sys.path.insert(0, "/workspace"); sys.path.insert(0, "/workspace/tools")
    from tools.qwen_descriptor import QwenDescriptor

    from kernel_triton.qwen_window_attention.triton_ops_qwen import qwen_window_attention

    vis = model.visual
    vc  = model.config.vision_config
    num_heads  = vc.num_heads
    head_dim   = vc.hidden_size // num_heads
    rws        = vc.window_size // vc.patch_size
    sms        = vc.spatial_merge_size
    smu        = sms ** 2
    fullatt    = set(vc.fullatt_block_indexes)
    BLOCK_SIZE = 64; NUM_WARPS = 4; NUM_STAGES = 3

    gen = QwenDescriptor(num_heads, rws, sms)
    _cache = {}

    def timed_forward(self, hidden_states, grid_thw, **kwargs):
        timings = {}
        device = hidden_states.device

        def rec(name):
            s2, e2 = record_pair()
            s2.record()
            return s2, e2, name

        # patch_embed
        s, e2, n = rec("patch_embed")
        hidden = self.patch_embed(hidden_states)
        e2.record(); torch.cuda.synchronize()
        timings[n] = ms(s, e2)

        rope = self.rot_pos_emb(grid_thw)
        emb  = torch.cat((rope, rope), dim=-1)
        pos_emb = (emb.cos(), emb.sin())

        cu_seqlens = F.pad(
            (grid_thw[:, 1] * grid_thw[:, 2] * grid_thw[:, 0])
            .cumsum(0, dtype=torch.int32), (1, 0), value=0)

        # descriptor build
        T, H, W = grid_thw[0].tolist()
        key = (H, W)
        s, e2, n = rec("descriptor_build")
        if key not in _cache:
            meta = gen.build(H, W, BLOCK_SIZE, device=device)
            pbk  = gen.per_batch_blocks(H, W, BLOCK_SIZE)
            MW2  = W // sms
            _cache[key] = (meta, pbk, MW2)
        guide_meta, guide_pbk, MW2 = _cache[key]
        guide_meta = guide_meta.to(device)
        e2.record(); torch.cuda.synchronize()
        timings[n] = ms(s, e2)

        # blocks
        win_times, full_times = [], []
        for li, blk in enumerate(self.blocks):
            s, e2 = record_pair()
            s.record()
            normed = blk.norm1(hidden)
            seq_len2, C = normed.shape

            if li in fullatt:
                attn_out = blk.attn(normed, cu_seqlens=cu_seqlens,
                                    position_embeddings=pos_emb, **kwargs)
            else:
                qkv = (blk.attn.qkv(normed)
                       .reshape(seq_len2, 3, num_heads, head_dim)
                       .permute(1, 0, 2, 3))
                q, k, v = qkv.unbind(0)
                cos, sin = pos_emb
                from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
                    apply_rotary_pos_emb_vision)
                q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
                q = q.transpose(0,1).unsqueeze(0).contiguous()
                k = k.transpose(0,1).unsqueeze(0).contiguous()
                v = v.transpose(0,1).unsqueeze(0).contiguous()
                out = qwen_window_attention(
                    q, k, v, guide_meta, MW2, smu, guide_pbk,
                    BLOCK_SIZE=BLOCK_SIZE, seq_layout=False,
                    num_warps=NUM_WARPS, num_stages=NUM_STAGES)
                attn_out = out[0].transpose(0,1).reshape(seq_len2, C)
                attn_out = blk.attn.proj(attn_out)

            hidden = hidden + attn_out
            hidden = hidden + blk.mlp(blk.norm2(hidden))

            e2.record(); torch.cuda.synchronize()
            t = ms(s, e2)
            if li in fullatt:
                full_times.append(t)
            else:
                win_times.append(t)

        timings["window_attn_blocks"] = win_times
        timings["full_attn_blocks"]   = full_times

        # merger
        s, e2, n = rec("merger")
        hidden = self.merger(hidden)
        e2.record(); torch.cuda.synchronize()
        timings[n] = ms(s, e2)

        self._last_timings = timings
        return hidden

    vis.forward = types.MethodType(timed_forward, vis)


# ── load model ────────────────────────────────────────────────────────────────

def load_model(model_path, mode):
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    proc = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    kwargs = dict(torch_dtype=torch.bfloat16, device_map="cuda",
                  local_files_only=True, attn_implementation="flash_attention_2")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, **kwargs)
    model.eval()
    if mode == "guide":
        make_guide_timed_forward(model)
    else:
        make_fa2_timed_forward(model.visual)
    return model, proc


# ── run one forward ───────────────────────────────────────────────────────────

def run_forward(model, proc, img_path, question="What is shown?"):
    from PIL import Image
    from qwen_vl_utils import process_vision_info
    img = Image.open(img_path).convert("RGB")
    msgs = [{"role":"user","content":[
        {"type":"image","image":img},{"type":"text","text":question}]}]
    text  = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    img_in, vid_in = process_vision_info(msgs)
    inputs = proc(text=[text], images=img_in, videos=vid_in,
                  padding=True, return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    # only run the visual encoder (no generate)
    pv  = inputs["pixel_values"].to(model.visual.patch_embed.proj.weight.dtype)
    gth = inputs["image_grid_thw"]

    with torch.no_grad():
        _ = model.visual(pv, gth)

    T, H, W = gth[0].tolist()
    return model.visual._last_timings, int(H), int(W)


# ── report ────────────────────────────────────────────────────────────────────

def report(label, timings, H, W, sms=2, smu=4):
    MH, MW = H // sms, W // sms
    win_list  = timings["window_attn_blocks"]
    full_list = timings["full_attn_blocks"]
    t_win_total  = sum(win_list)
    t_full_total = sum(full_list)

    scatter = timings.get("scatter_hidden_rope", 0)
    reverse = timings.get("reverse_scatter",     0)
    desc    = timings.get("descriptor_build",    0)
    pe      = timings.get("patch_embed",         0)
    gwidx   = timings.get("get_window_index",    0)
    merger  = timings.get("merger",              0)

    total = pe + gwidx + scatter + t_win_total + t_full_total + merger + reverse + desc

    print(f"\n{'='*62}")
    print(f"  {label}  |  grid_thw H={H} W={W}  →  merged {MH}×{MW}")
    print(f"{'='*62}")
    def row(name, t, note=""):
        pct = 100*t/total if total>0 else 0
        print(f"  {name:<28} {t:>8.2f} ms  {pct:>5.1f}%  {note}")

    row("patch_embed",            pe)
    if gwidx:
        row("get_window_index",       gwidx)
    if scatter:
        row("scatter (hidden+rope)",  scatter,  "← partition overhead")
    if desc > 0.05:
        row("descriptor_build",       desc,     "(cached=0 if warmed)")
    row("window_attn ×28",         t_win_total)
    for i, t in enumerate(win_list):
        print(f"    block {i:2d}            {t:>8.2f} ms")
    row("full_attn  ×4",           t_full_total)
    for i, t in enumerate(full_list):
        print(f"    block {[7,15,23,31][i]:2d}            {t:>8.2f} ms")
    row("merger",                  merger)
    if reverse:
        row("reverse_scatter",        reverse,  "← partition overhead")
    print(f"  {'─'*54}")
    row("TOTAL VE",                total)
    overhead = scatter + reverse + gwidx
    if overhead > 0:
        print(f"\n  partition total (scatter+reverse+get_idx): "
              f"{overhead:.2f} ms  ({100*overhead/total:.1f}% of VE)")
    attn_total = t_win_total + t_full_total
    print(f"  attention total (win+full): "
          f"{attn_total:.2f} ms  ({100*attn_total/total:.1f}% of VE)")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/workspace/models/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--image-path", required=True)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--runs",   type=int, default=5)
    ap.add_argument("--question", default="What is shown in this document?")
    args = ap.parse_args()

    print(f"\nLoading models...")
    fa2_model,  proc = load_model(args.model_path, "fa2")
    guide_model, _   = load_model(args.model_path, "guide")
    print("Models loaded.")

    def run_avg(model, n_warmup, n_runs):
        from collections import defaultdict
        import copy
        # warmup
        for _ in range(n_warmup):
            t, H, W = run_forward(model, proc, args.image_path, args.question)
        # collect
        agg = defaultdict(list)
        for _ in range(n_runs):
            t, H, W = run_forward(model, proc, args.image_path, args.question)
            for k, v in t.items():
                agg[k].append(v)
        # average (scalars and lists separately)
        avg = {}
        for k, vlist in agg.items():
            if isinstance(vlist[0], list):
                avg[k] = [sum(x[i] for x in vlist)/n_runs
                           for i in range(len(vlist[0]))]
            else:
                avg[k] = sum(vlist) / n_runs
        return avg, H, W

    print(f"\nRunning FA2   (warmup={args.warmup}, runs={args.runs})...")
    fa2_t, H, W = run_avg(fa2_model, args.warmup, args.runs)
    report("FA2", fa2_t, H, W)

    print(f"\nRunning GUIDE (warmup={args.warmup}, runs={args.runs})...")
    guide_t, H, W = run_avg(guide_model, args.warmup, args.runs)
    report("GUIDE", guide_t, H, W)

    # diff summary
    def get_total(t):
        win = sum(t["window_attn_blocks"])
        full = sum(t["full_attn_blocks"])
        sc = t.get("scatter_hidden_rope", 0)
        rv = t.get("reverse_scatter", 0)
        gw = t.get("get_window_index", 0)
        desc = t.get("descriptor_build", 0)
        pe = t.get("patch_embed", 0)
        mg = t.get("merger", 0)
        return pe + gw + sc + win + full + mg + rv + desc

    fa2_total   = get_total(fa2_t)
    guide_total = get_total(guide_t)
    fa2_attn    = sum(fa2_t["window_attn_blocks"])   + sum(fa2_t["full_attn_blocks"])
    guide_attn  = sum(guide_t["window_attn_blocks"]) + sum(guide_t["full_attn_blocks"])
    fa2_part    = fa2_t.get("scatter_hidden_rope",0) + fa2_t.get("reverse_scatter",0) + fa2_t.get("get_window_index",0)

    print(f"\n{'='*62}")
    print(f"  SUMMARY  (avg over {args.runs} runs)")
    print(f"{'='*62}")
    print(f"  {'':28}  {'FA2':>8}  {'GUIDE':>8}  {'ratio':>7}")
    print(f"  {'─'*54}")
    print(f"  {'VE total':28}  {fa2_total:>8.2f}  {guide_total:>8.2f}  {fa2_total/guide_total:>6.3f}x")
    print(f"  {'attention (win+full)':28}  {fa2_attn:>8.2f}  {guide_attn:>8.2f}  {fa2_attn/guide_attn:>6.3f}x")
    print(f"  {'partition overhead (FA2)':28}  {fa2_part:>8.2f}  {'N/A':>8}         ")
    print(f"  {'partition % of FA2 VE':28}  {100*fa2_part/fa2_total:>7.1f}%")
    print(f"{'='*62}")


if __name__ == "__main__":
    main()
