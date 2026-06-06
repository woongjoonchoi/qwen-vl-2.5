#!/usr/bin/env python3
"""
Stage 7: Python Reference Attention Correctness

Compares baseline window attention (gather via expanded_window_index)
vs GUIDE descriptor attention (direct gather/scatter).
No Triton — pure PyTorch.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


def scaled_dot_product_attention_window(Q, K, V, scale):
    """Standard scaled dot-product attention for a single window."""
    scores = torch.matmul(Q, K.transpose(-1, -2)) * scale
    probs  = F.softmax(scores, dim=-1)
    return torch.matmul(probs, V)


def baseline_attention(Q, K, V, expanded_wi, rws, smu, num_heads):
    """
    Baseline: gather Q/K/V via expanded_window_index, per-window attention,
    scatter back.
    Q/K/V: [num_heads, S_raw, head_dim]
    expanded_wi: [S_raw] indices (window-major order)
    Returns: [num_heads, S_raw, head_dim]
    """
    S_raw = Q.shape[1]
    head_dim = Q.shape[-1]
    scale = head_dim ** -0.5
    win_len = rws * rws

    # Gather into window-major order
    Q_w = Q[:, expanded_wi, :]   # [nh, S_raw, D]
    K_w = K[:, expanded_wi, :]
    V_w = V[:, expanded_wi, :]

    # Process each window
    n_windows = S_raw // win_len
    O_w = torch.zeros_like(Q_w)
    for wi in range(n_windows):
        s = wi * win_len
        e = s + win_len
        q = Q_w[:, s:e, :]  # [nh, win_len, D]
        k = K_w[:, s:e, :]
        v = V_w[:, s:e, :]
        o = scaled_dot_product_attention_window(q, k, v, scale)
        O_w[:, s:e, :] = o

    # Handle boundary (if S_raw not divisible by win_len)
    rem = S_raw % win_len
    if rem > 0:
        s = n_windows * win_len
        q = Q_w[:, s:, :]
        k = K_w[:, s:, :]
        v = V_w[:, s:, :]
        o = scaled_dot_product_attention_window(q, k, v, scale)
        O_w[:, s:, :] = o

    # Scatter back to row-major order
    O = torch.zeros_like(Q)
    O[:, expanded_wi, :] = O_w
    return O


def descriptor_attention(Q, K, V, desc_offsets, win_offsets_list, num_heads):
    """
    GUIDE: descriptor-guided gather/scatter.
    desc_offsets: list of window groups (each group = list of offsets)
    Q/K/V: [num_heads, S_raw, head_dim]
    Returns: [num_heads, S_raw, head_dim]
    """
    head_dim = Q.shape[-1]
    scale = head_dim ** -0.5
    O = torch.zeros_like(Q)

    for win_offsets in win_offsets_list:
        offs = torch.tensor(win_offsets, device=Q.device, dtype=torch.long)
        q = Q[:, offs, :]   # [nh, win_len, D]
        k = K[:, offs, :]
        v = V[:, offs, :]
        o = scaled_dot_product_attention_window(q, k, v, scale)
        O[:, offs, :] += o

    return O


def compute_metrics(baseline, guide):
    """Compute error metrics between two tensors."""
    diff = (baseline - guide).abs()
    return {
        "max_abs_error":    diff.max().item(),
        "mean_abs_error":   diff.mean().item(),
        "rel_l2_error":     (diff.norm() / baseline.norm().clamp(min=1e-8)).item(),
        "cosine_sim":       F.cosine_similarity(
                                baseline.reshape(-1), guide.reshape(-1), dim=0).item(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir",  required=True)
    parser.add_argument("--dtype",       default="fp32,bf16")
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(Path(__file__).parent))
    from qwen_descriptor import QwenDescriptor

    # Load config
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(args.model_path, local_files_only=True)
    vc  = cfg.vision_config
    num_heads  = vc.num_heads
    head_dim   = vc.hidden_size // num_heads
    rws        = vc.window_size // vc.patch_size
    sms        = vc.spatial_merge_size
    smu        = sms ** 2
    patch_size = vc.patch_size

    gen = QwenDescriptor(num_heads=num_heads, raw_window_size=rws,
                         spatial_merge_size=sms)

    results = []
    dtypes = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
    test_dtypes = [d.strip() for d in args.dtype.split(",") if d.strip() in dtypes]

    # Test cases: (H, W) in raw-patch units
    test_cases = [
        ("H32W32_1win",  8,  8),   # single window
        ("H32W32",      32, 32),   # 16 windows
        ("H64W64",      64, 64),   # 64 windows (res=896)
    ]

    for case_name, H, W in test_cases:
        for dtype_name in test_dtypes:
            dtype = dtypes[dtype_name]
            print(f"\n[Stage7] case={case_name} H={H} W={W} dtype={dtype_name}", flush=True)

            S_raw = H * W
            BLOCK_SIZE = min(32, rws * rws)

            # Build descriptor & window groups
            win_groups = gen.emit_window_groups(H, W, BLOCK_SIZE)

            # Build expanded_window_index
            H_m = H // sms
            W_m = W // sms
            rws_m = rws // sms
            n_win_H = (H_m + rws_m - 1) // rws_m
            n_win_W = (W_m + rws_m - 1) // rws_m
            merged_idx = []
            for wh in range(n_win_H):
                for ww in range(n_win_W):
                    for mh in range(rws_m):
                        for mw in range(rws_m):
                            h = wh * rws_m + mh
                            w = ww * rws_m + mw
                            if h < H_m and w < W_m:
                                merged_idx.append(h * W_m + w)
            window_index = torch.tensor(merged_idx, dtype=torch.long, device="cuda")
            expanded_wi = (
                window_index[:, None] * smu +
                torch.arange(smu, device="cuda")[None, :]
            ).reshape(-1)

            # Synthetic Q/K/V
            if dtype in (torch.float16, torch.bfloat16):
                Q = torch.randn(num_heads, S_raw, head_dim, device="cuda", dtype=torch.float32).to(dtype)
                K = torch.randn(num_heads, S_raw, head_dim, device="cuda", dtype=torch.float32).to(dtype)
                V = torch.randn(num_heads, S_raw, head_dim, device="cuda", dtype=torch.float32).to(dtype)
            else:
                Q = torch.randn(num_heads, S_raw, head_dim, device="cuda", dtype=dtype)
                K = torch.randn(num_heads, S_raw, head_dim, device="cuda", dtype=dtype)
                V = torch.randn(num_heads, S_raw, head_dim, device="cuda", dtype=dtype)

            # Run baseline
            with torch.no_grad():
                O_base = baseline_attention(Q, K, V, expanded_wi, rws, smu, num_heads)
                O_guide = descriptor_attention(Q, K, V, None, win_groups, num_heads)

            # Compute metrics
            # Move both to fp32 for comparison
            metrics = compute_metrics(
                O_base.float().cpu(),
                O_guide.float().cpu()
            )

            atol = 1e-4 if dtype == torch.float32 else 1e-2
            pass_check = metrics["max_abs_error"] <= atol

            print(f"  max_abs_err={metrics['max_abs_error']:.6f}  "
                  f"rel_l2={metrics['rel_l2_error']:.6f}  "
                  f"atol={atol}  PASS={pass_check}", flush=True)

            result = {
                "case": case_name, "H": H, "W": W, "dtype": dtype_name,
                "num_heads": num_heads, "head_dim": head_dim,
                **{k: round(v, 8) for k, v in metrics.items()},
                "atol": atol,
                "status": "PASS" if pass_check else "FAIL",
            }
            results.append(result)

    # Save
    if results:
        with open(out / "python_reference_correctness.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader(); w.writerows(results)

    all_pass = all(r["status"] == "PASS" for r in results)
    status = {"status": "PASS" if all_pass else "FAIL",
              "passing": sum(1 for r in results if r["status"] == "PASS"),
              "total": len(results)}
    (out / "stage_status.json").write_text(json.dumps(status, indent=2))

    print(f"\n[Stage7] Result: {status['status']}  {status['passing']}/{status['total']}")
    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
