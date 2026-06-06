#!/usr/bin/env python3
"""
Stage 12: baseline segment vs GUIDE replacement segment (nsys NVTX).

Emits NVTX ranges (each wraps `profile_iters` ops + a CUDA sync):

  one-time layout (baseline only):
    baseline.layout.hidden_reorder      row-major -> window-major gather
    baseline.layout.rope_reorder        RoPE gather to window-major
    baseline.layout.reverse_restore     window-major -> row-major scatter

  per selected window layer:
    baseline.attn_core.layerXX          batched SDPA over window-major Q/K/V
    guide.attn_core.layerXX.blockBEST   GUIDE descriptor Triton kernel (row-major)

Comparison A (attention-core only): baseline.attn_core vs guide.attn_core.
Comparison B (reorder-free replacement): GUIDE removes the one-time layout cost;
the parser reports raw layout cost and the per-window-layer amortized cost.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--target-layers", default="0,1,8,16,24")
    ap.add_argument("--best-config", required=True)
    ap.add_argument("--resize-px", type=int, default=896)
    ap.add_argument("--profile-iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--precision", default="bf16")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(Path(__file__).parent))
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from tools.qwen_descriptor import QwenDescriptor, get_per_batch_blocks
    from kernel_triton.qwen_window_attention.triton_ops_qwen import qwen_window_attention

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
             "fp32": torch.float32}[args.precision]

    from transformers import AutoConfig
    vc = AutoConfig.from_pretrained(args.model_path, local_files_only=True).vision_config
    num_heads = vc.num_heads
    head_dim = vc.hidden_size // num_heads
    hidden = vc.hidden_size
    rws = vc.window_size // vc.patch_size
    sms = vc.spatial_merge_size
    smu = sms ** 2
    patch = vc.patch_size
    depth = vc.depth
    num_window_layers = depth - len(vc.fullatt_block_indexes)

    best = json.loads(Path(args.best_config).read_text())
    BS = int(best.get("block_size", 32))
    NW = int(best.get("num_warps", 4))
    NS = int(best.get("num_stages", 3))

    H = W = args.resize_px // patch
    S = H * W
    MW = W // sms
    win_len = rws * rws
    n_win = S // win_len
    scale = head_dim ** -0.5
    layers = [int(x) for x in args.target_layers.split(",")]

    gen = QwenDescriptor(num_heads, rws, sms)
    expanded_wi = gen.emit_offsets(H, W, BS).to("cuda")
    meta = gen.build(H, W, BS, device="cuda")
    pbk = get_per_batch_blocks(H, W, num_heads, rws, sms, BS)

    # row-major tensors
    hidden_rm = torch.randn(S, hidden, device="cuda", dtype=dtype)
    rope_rm = torch.randn(S, head_dim, device="cuda", dtype=dtype)
    Qr = torch.randn(1, num_heads, S, head_dim, device="cuda", dtype=torch.float32).to(dtype)
    Kr = torch.randn(1, num_heads, S, head_dim, device="cuda", dtype=torch.float32).to(dtype)
    Vr = torch.randn(1, num_heads, S, head_dim, device="cuda", dtype=torch.float32).to(dtype)
    # window-major Q/K/V for the baseline core
    Qw = Qr[:, :, expanded_wi].reshape(num_heads, n_win, win_len, head_dim)
    Kw = Kr[:, :, expanded_wi].reshape(num_heads, n_win, win_len, head_dim)
    Vw = Vr[:, :, expanded_wi].reshape(num_heads, n_win, win_len, head_dim)
    O_wm = torch.empty_like(Qw)

    PI, WU = args.profile_iters, args.warmup

    def baseline_attn():
        # batched SDPA over [nh, n_win, win_len, hd] (sdpa backend)
        return F.scaled_dot_product_attention(Qw, Kw, Vw)

    def guide_attn():
        return qwen_window_attention(Qr, Kr, Vr, meta, MW, smu, pbk, BLOCK_SIZE=BS,
                                     seq_layout=False, num_warps=NW, num_stages=NS)

    def run_range(name, fn):
        for _ in range(WU):
            fn()
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_push(name)
        for _ in range(PI):
            fn()
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()

    info = {
        "H": H, "W": W, "S": S, "num_heads": num_heads, "head_dim": head_dim,
        "num_windows": n_win, "tokens_per_window": win_len,
        "block_size": BS, "num_warps": NW, "num_stages": NS,
        "profile_iters": PI, "num_window_layers": num_window_layers,
        "precision": args.precision, "target_layers": layers,
    }
    print(f"[Stage12] {info}", flush=True)

    torch.cuda.nvtx.range_push("segment_compare.total")

    # one-time layout (baseline)
    run_range("baseline.layout.hidden_reorder",
              lambda: hidden_rm.reshape(S // smu, smu, hidden)[
                  expanded_wi[::smu] // smu].reshape(-1, hidden))
    run_range("baseline.layout.rope_reorder", lambda: rope_rm[expanded_wi])
    run_range("baseline.layout.reverse_restore",
              lambda: torch.empty_like(Qr).index_copy_(2, expanded_wi, Qr))

    # per-layer attention cores
    for li in layers:
        run_range(f"baseline.attn_core.layer{li:02d}", baseline_attn)
        run_range(f"guide.attn_core.layer{li:02d}.block{BS}", guide_attn)

    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()

    (out / "segment_info.json").write_text(json.dumps(info, indent=2))
    print("[Stage12] segment compare done", flush=True)


if __name__ == "__main__":
    main()
