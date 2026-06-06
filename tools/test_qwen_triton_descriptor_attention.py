#!/usr/bin/env python3
"""
Stage 9: Triton Synthetic Correctness

Tests Triton GUIDE kernel against Python reference on synthetic Q/K/V.
Cases: 8x8, 16x16, 32x32, 32x48 in raw patch units.
Block sizes: 16, 32, 64.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


def compute_metrics(ref, out):
    ref_f = ref.float().cpu()
    out_f = out.float().cpu()
    diff = (ref_f - out_f).abs()
    return {
        "max_abs_error":  diff.max().item(),
        "mean_abs_error": diff.mean().item(),
        "rel_l2_error":   (diff.norm() / ref_f.norm().clamp(min=1e-8)).item(),
        "cosine_sim":     F.cosine_similarity(ref_f.reshape(-1), out_f.reshape(-1), dim=0).item(),
    }


def python_reference_attention(Q, K, V, win_groups, head_dim):
    """Pure Python attention used as reference."""
    scale = head_dim ** -0.5
    O = torch.zeros_like(Q)
    for win_offs in win_groups:
        offs = torch.tensor(win_offs, device=Q.device, dtype=torch.long)
        q = Q[:, offs, :]
        k = K[:, offs, :]
        v = V[:, offs, :]
        scores = torch.matmul(q, k.transpose(-1, -2)) * scale
        probs  = F.softmax(scores, dim=-1)
        o      = torch.matmul(probs, v)
        O[:, offs, :] += o
    return O


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path",  required=True)
    parser.add_argument("--output-dir",  required=True)
    parser.add_argument("--cases",       default="8x8,16x16,32x32,32x48")
    parser.add_argument("--block-sizes", default="16,32,64")
    parser.add_argument("--dtypes",      default="fp16,bf16")
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(Path(__file__).parent))
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from tools.qwen_descriptor import QwenDescriptor, get_per_batch_blocks
    from kernel_triton.qwen_window_attention.triton_ops_qwen import qwen_window_attention

    # Load config
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(args.model_path, local_files_only=True)
    vc = cfg.vision_config
    num_heads = vc.num_heads
    head_dim  = vc.hidden_size // num_heads
    rws       = vc.window_size // vc.patch_size
    sms       = vc.spatial_merge_size
    smu       = sms ** 2

    gen = QwenDescriptor(num_heads=num_heads, raw_window_size=rws,
                         spatial_merge_size=sms)

    block_sizes = [int(b) for b in args.block_sizes.split(",")]
    dtypes_map  = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    test_dtypes = [d.strip() for d in args.dtypes.split(",") if d.strip() in dtypes_map]

    # Parse cases
    test_cases = []
    for case in args.cases.split(","):
        h, w = (int(x) for x in case.strip().split("x"))
        test_cases.append((f"{h}x{w}", h, w))

    # Also test small debug dimensions
    debug_nh = 2
    debug_hd = 16
    test_cases_debug = [("debug_8x8", 8, 8)]

    results = []

    def run_case(case_name, H, W, nh, hd, dtype, BS, is_debug=False):
        S = H * W
        if S % BS != 0 and BS > S:
            return None  # skip

        # Skip if BLOCK_SIZE > tokens_per_window (no valid kernel launch)
        if BS > rws * rws:
            return {"case": case_name, "H": H, "W": W, "dtype": dtype_name,
                    "BLOCK_SIZE": BS, "status": "SKIPPED_BS_GT_WIN", "max_abs_error": -1}

        try:
            if dtype in (torch.float16, torch.bfloat16):
                Q = torch.randn(1, nh, S, hd, device="cuda", dtype=torch.float32).to(dtype)
                K = torch.randn(1, nh, S, hd, device="cuda", dtype=torch.float32).to(dtype)
                V = torch.randn(1, nh, S, hd, device="cuda", dtype=torch.float32).to(dtype)
            else:
                Q = torch.randn(1, nh, S, hd, device="cuda", dtype=dtype)
                K = torch.randn(1, nh, S, hd, device="cuda", dtype=dtype)
                V = torch.randn(1, nh, S, hd, device="cuda", dtype=dtype)

            # Build descriptor
            meta = gen.build(H, W, BS, device="cuda")
            pbk  = get_per_batch_blocks(H, W, nh, rws, sms, BS)
            win_groups = gen.emit_window_groups(H, W, BS)
            MW = W // sms

            # Python reference computed in fp32 (true answer); kernel keeps its
            # native dtype. This isolates kernel correctness from input rounding.
            with torch.no_grad():
                O_ref = python_reference_attention(
                    Q[0].float(), K[0].float(), V[0].float(), win_groups, hd)

            # Triton kernel
            with torch.no_grad():
                O_tri = qwen_window_attention(Q, K, V, meta, MW, smu, pbk,
                                               BLOCK_SIZE=BS, seq_layout=False)

            # Compare. Tolerances reflect mantissa width:
            #   fp32: 1e-4, fp16 (10-bit): 1e-2, bf16 (7-bit): 2.5e-2.
            # rel_l2 is the dtype-robust primary metric.
            metrics = compute_metrics(O_ref.unsqueeze(0), O_tri)
            atol = {torch.float32: 1e-4, torch.float16: 1e-2,
                    torch.bfloat16: 2.5e-2}[dtype]
            pass_check = (metrics["max_abs_error"] <= atol
                          and metrics["rel_l2_error"] <= 1e-2)

            return {
                "case": case_name, "H": H, "W": W, "dtype": dtype_name,
                "BLOCK_SIZE": BS, "num_heads": nh, "head_dim": hd,
                "is_debug": is_debug,
                **{k: round(v, 8) for k, v in metrics.items()},
                "atol": atol,
                "status": "PASS" if pass_check else "FAIL",
            }
        except Exception as e:
            return {
                "case": case_name, "H": H, "W": W, "dtype": dtype_name,
                "BLOCK_SIZE": BS, "status": f"ERROR: {str(e)[:80]}",
                "max_abs_error": -1,
            }

    # Debug cases first
    for case_name, H, W in test_cases_debug:
        for BS in block_sizes:
            for dtype_name in test_dtypes:
                dtype = dtypes_map[dtype_name]
                r = run_case(case_name, H, W, debug_nh, debug_hd, dtype, BS, is_debug=True)
                if r:
                    results.append(r)
                    print(f"[debug] {case_name} BS={BS} {dtype_name}: {r.get('status')} max_err={r.get('max_abs_error', '?')}", flush=True)

    # Model-config cases
    for case_name, H, W in test_cases:
        for BS in block_sizes:
            for dtype_name in test_dtypes:
                dtype = dtypes_map[dtype_name]
                r = run_case(case_name, H, W, num_heads, head_dim, dtype, BS)
                if r:
                    results.append(r)
                    print(f"[model] {case_name} BS={BS} {dtype_name}: {r.get('status')} max_err={r.get('max_abs_error', '?')}", flush=True)

    if results:
        with open(out / "triton_synthetic_correctness.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()), extrasaction="ignore")
            w.writeheader(); w.writerows(results)

    passing   = sum(1 for r in results if r.get("status") == "PASS")
    errored   = sum(1 for r in results if str(r.get("status","")).startswith("ERROR"))
    total     = len(results)
    all_pass  = passing == (total - errored) and errored == 0

    status = {"status": "PASS" if all_pass else "PARTIAL" if passing > 0 else "FAIL",
              "passing": passing, "total": total, "errors": errored}
    (out / "stage_status.json").write_text(json.dumps(status, indent=2))
    print(f"\n[Stage9] {status['status']}  {passing}/{total} pass  {errored} errors")
    if not all_pass:
        for r in results:
            if r.get("status") not in ("PASS", "SKIPPED_BS_GT_WIN"):
                print(f"  FAIL/ERR: {r}", flush=True)
        if errored == total:
            sys.exit(1)


if __name__ == "__main__":
    main()
