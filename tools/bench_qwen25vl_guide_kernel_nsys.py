#!/usr/bin/env python3
"""
Stage 11: Nsight Systems block-size performance sweep for the GUIDE kernel.

Emits one NVTX range per (BLOCK_SIZE, NUM_WARPS, NUM_STAGES) candidate:
    guide.block_sweep.block{bs}.warps{nw}.stages{ns}
Inside each range, `profile_iters` kernel invocations run, then a CUDA sync.
nsys `nvtx_gpu_proj_trace` then attributes projected GPU time to each range.

Also runs a per-candidate correctness check (kernel vs fp32 torch reference) so
that winner selection can be restricted to correctness-passing candidates.

Modes:
  synthetic : fixed H/W (args), synthetic random Q/K/V at model dims.
  captured  : real Qwen vision shape (square resize), synthetic Q/K/V at that
              shape/dtype (kernel-level perf depends on shape+dtype, not values).
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


def build_reference(Qr, Kr, Vr, expanded_wi, win_len, scale):
    """fp32 baseline: gather row-major -> window-major, batched SDPA, scatter."""
    nh, S, hd = Qr.shape
    Qw = Qr[:, expanded_wi].float().reshape(nh, -1, win_len, hd)
    Kw = Kr[:, expanded_wi].float().reshape(nh, -1, win_len, hd)
    Vw = Vr[:, expanded_wi].float().reshape(nh, -1, win_len, hd)
    sc = torch.matmul(Qw, Kw.transpose(-1, -2)) * scale
    Ow = torch.matmul(F.softmax(sc, dim=-1), Vw).reshape(nh, S, hd)
    O = torch.empty_like(Ow)
    O[:, expanded_wi] = Ow
    return O   # row-major fp32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--mode", choices=["synthetic", "captured"], default="captured")
    ap.add_argument("--target-layer", type=int, default=8)
    ap.add_argument("--resize-px", type=int, default=896)
    ap.add_argument("--H", type=int, default=64)
    ap.add_argument("--W", type=int, default=64)
    ap.add_argument("--block-sizes", default="16,32,64")
    ap.add_argument("--num-warps", default="4,8")
    ap.add_argument("--num-stages", default="3,4")
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
    rws = vc.window_size // vc.patch_size
    sms = vc.spatial_merge_size
    smu = sms ** 2
    patch = vc.patch_size

    if args.mode == "captured":
        H = W = args.resize_px // patch
    else:
        H, W = args.H, args.W
    S = H * W
    MW = W // sms
    win_len = rws * rws
    scale = head_dim ** -0.5
    gen = QwenDescriptor(num_heads, rws, sms)

    expanded_wi = gen.emit_offsets(H, W, 32).to("cuda")
    Q = torch.randn(1, num_heads, S, head_dim, device="cuda", dtype=torch.float32).to(dtype)
    K = torch.randn(1, num_heads, S, head_dim, device="cuda", dtype=torch.float32).to(dtype)
    V = torch.randn(1, num_heads, S, head_dim, device="cuda", dtype=torch.float32).to(dtype)
    O_ref = build_reference(Q[0], K[0], V[0], expanded_wi, win_len, scale)

    block_sizes = [int(x) for x in args.block_sizes.split(",")]
    num_warps_l = [int(x) for x in args.num_warps.split(",")]
    num_stages_l = [int(x) for x in args.num_stages.split(",")]

    candidates = [(bs, nw, ns) for bs in block_sizes
                  for nw in num_warps_l for ns in num_stages_l]

    info = {
        "mode": args.mode, "target_layer": args.target_layer,
        "H": H, "W": W, "S": S, "num_heads": num_heads, "head_dim": head_dim,
        "num_windows": S // win_len, "tokens_per_window": win_len,
        "precision": args.precision, "profile_iters": args.profile_iters,
    }

    # Pre-build descriptors + correctness check (outside profiled ranges)
    metas = {}
    pbks = {}
    correctness = {}
    for bs in block_sizes:
        metas[bs] = gen.build(H, W, bs, device="cuda")
        pbks[bs] = get_per_batch_blocks(H, W, num_heads, rws, sms, bs)
    for (bs, nw, ns) in candidates:
        try:
            O = qwen_window_attention(Q, K, V, metas[bs], MW, smu, pbks[bs],
                                      BLOCK_SIZE=bs, seq_layout=False,
                                      num_warps=nw, num_stages=ns)[0]
            rel = ((O.float() - O_ref).norm() / O_ref.norm().clamp(min=1e-8)).item()
            correctness[f"block{bs}.warps{nw}.stages{ns}"] = {
                "rel_l2_error": rel, "status": "PASS" if rel <= 1e-2 else "FAIL"}
        except Exception as e:
            correctness[f"block{bs}.warps{nw}.stages{ns}"] = {
                "status": f"ERROR: {str(e)[:60]}", "rel_l2_error": -1}

    torch.cuda.synchronize()
    print(f"[Stage11] {info}", flush=True)

    # ── Profiled sweep ────────────────────────────────────────────────────────
    torch.cuda.nvtx.range_push("guide.block_sweep.total")
    for (bs, nw, ns) in candidates:
        tag = f"block{bs}.warps{nw}.stages{ns}"
        if correctness[tag].get("status") != "PASS":
            continue  # do not profile non-correct candidates
        meta, pbk = metas[bs], pbks[bs]
        # warmup (outside range)
        for _ in range(args.warmup):
            qwen_window_attention(Q, K, V, meta, MW, smu, pbk, BLOCK_SIZE=bs,
                                  seq_layout=False, num_warps=nw, num_stages=ns)
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_push(f"guide.block_sweep.{tag}")
        for _ in range(args.profile_iters):
            qwen_window_attention(Q, K, V, meta, MW, smu, pbk, BLOCK_SIZE=bs,
                                  seq_layout=False, num_warps=nw, num_stages=ns)
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()

    (out / "bench_info.json").write_text(json.dumps(info, indent=2))
    (out / "candidates_correctness.json").write_text(json.dumps(correctness, indent=2))
    print("[Stage11] sweep done", flush=True)


if __name__ == "__main__":
    main()
