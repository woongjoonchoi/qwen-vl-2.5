#!/usr/bin/env python3
"""
Stage 6: Offset / Window-Length / RoPE Equivalence Tests

Authoritative comparison: the baseline window_index / cu_window_seqlens are
obtained from the *real* Qwen2.5-VL `get_window_index` method (not a
reconstruction).  The GUIDE descriptor (Option B, merged-cell grid) must
reproduce the exact `expanded_window_index`.

Tests:
  6.1 expanded_window_index offsets   == descriptor emit_offsets
  6.2 cu_window_seqlens raw lengths   == descriptor window lengths
  6.3 RoPE gather (descriptor offsets)== baseline RoPE reorder
  6.4 arange visual debug
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import torch


def get_real_get_window_index(vc):
    """Return a bound-style callable replicating the model's get_window_index,
    using the real method from transformers with a lightweight shim that only
    carries the config attributes the method reads."""
    from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as M
    fn = M.Qwen2_5_VisionTransformerPretrainedModel.get_window_index

    class Shim:
        spatial_merge_size = vc.spatial_merge_size
        patch_size = vc.patch_size
        window_size = vc.window_size
        spatial_merge_unit = vc.spatial_merge_size ** 2

    shim = Shim()
    return lambda grid_thw: fn(shim, grid_thw)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cases", default="divisible")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(Path(__file__).parent))
    from qwen_descriptor import QwenDescriptor

    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(args.model_path, local_files_only=True)
    vc = cfg.vision_config
    num_heads = vc.num_heads
    patch_size = vc.patch_size
    sms = vc.spatial_merge_size
    smu = sms ** 2
    rws = vc.window_size // patch_size          # raw window size (=8)

    get_window_index = get_real_get_window_index(vc)
    gen = QwenDescriptor(num_heads=num_heads, raw_window_size=rws,
                         spatial_merge_size=sms)

    # ── Build test cases ──────────────────────────────────────────────────────
    test_cases = []
    if "divisible" in args.cases:
        for res in [448, 896, 1344]:
            Hp = res // patch_size
            Wp = res // patch_size
            test_cases.append((f"res{res}_H{Hp}W{Wp}", Hp, Wp))
    if "boundary" in args.cases:
        test_cases.append(("boundary_H30W45", 30, 45))

    results = []
    for case_name, H, W in test_cases:
        # Authoritative baseline from the real model
        grid_thw = torch.tensor([[1, H, W]])
        window_index, cu_window_seqlens = get_window_index(grid_thw)
        window_index = window_index.long()
        cu = torch.tensor(cu_window_seqlens, dtype=torch.long)
        expanded_window_index = (
            window_index[:, None] * smu
            + torch.arange(smu)[None, :]
        ).reshape(-1)                                    # [S_raw]
        S_raw = H * W
        hf_lengths = (cu[1:] - cu[:-1]).tolist()
        # drop trailing zero-length windows that HF can emit at grid edges
        hf_lengths = [l for l in hf_lengths if l > 0]

        for BLOCK_SIZE in [16, 32, 64]:
            print(f"\n[Test 6] case={case_name} H={H} W={W} BS={BLOCK_SIZE}", flush=True)

            try:
                desc_offsets = gen.emit_offsets(H, W, BLOCK_SIZE).long()
                build_ok = True
            except NotImplementedError as e:
                print(f"  SKIP (descriptor): {e}", flush=True)
                build_ok = False
                desc_offsets = torch.empty(0, dtype=torch.long)

            # 6.1 offset order equality
            offset_order_equal = (
                build_ok
                and desc_offsets.numel() == expanded_window_index.numel()
                and torch.equal(desc_offsets, expanded_window_index)
            )

            # token-set per window
            tpw = rws * rws
            def wins(t):
                return [sorted(t[i * tpw:(i + 1) * tpw].tolist())
                        for i in range(t.numel() // tpw)]
            set_equal = (build_ok
                         and sorted(wins(desc_offsets)) == sorted(wins(expanded_window_index)))

            # 6.2 lengths
            desc_groups = gen.emit_window_groups(H, W, BLOCK_SIZE) if build_ok else []
            desc_lengths = [len(g) for g in desc_groups]
            length_equal = (desc_lengths == hf_lengths)

            # 6.3 RoPE gather equivalence (use arange as rope payload proxy)
            rope_dim = 8
            rope = torch.arange(S_raw)[:, None].repeat(1, rope_dim).float()
            hf_rope = rope.reshape(S_raw // smu, smu, rope_dim)[window_index] \
                          .reshape(S_raw, rope_dim)
            rope_equal = (build_ok
                          and torch.equal(rope[desc_offsets], hf_rope))

            # 6.4 arange visual debug
            x = torch.arange(S_raw)
            debug_match = (build_ok
                           and torch.equal(x[desc_offsets], x[expanded_window_index]))

            status = "PASS" if (offset_order_equal and length_equal and rope_equal) \
                else ("SKIP" if not build_ok else "FAIL")
            print(f"  offset_order_equal={offset_order_equal} set_equal={set_equal} "
                  f"length_equal={length_equal} rope_equal={rope_equal} → {status}", flush=True)

            # debug dump
            dbg = f"Case {case_name} H={H} W={W} BS={BLOCK_SIZE}\n"
            for wi in range(min(3, len(desc_groups))):
                base_win = sorted(expanded_window_index[wi * tpw:(wi + 1) * tpw].tolist())
                dbg += f"Window {wi}: desc={sorted(desc_groups[wi])[:8]}...\n"
                dbg += f"         base={base_win[:8]}...\n"
            (out / f"first_windows_debug_{case_name}_bs{BLOCK_SIZE}.txt").write_text(dbg)

            results.append({
                "case": case_name, "H": H, "W": W, "BLOCK_SIZE": BLOCK_SIZE,
                "S_raw": S_raw, "n_windows": len(desc_groups),
                "offset_order_equal": offset_order_equal,
                "window_token_set_equal": set_equal,
                "length_equal": length_equal,
                "rope_equal": rope_equal,
                "debug_match": debug_match,
                "status": status,
            })

    # CSV
    with open(out / "offset_equivalence_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader(); w.writerows(results)

    # RoPE / length CSVs (split views for the plan's required outputs)
    with open(out / "window_length_equivalence.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["case", "BLOCK_SIZE", "length_equal"])
        for r in results:
            w.writerow([r["case"], r["BLOCK_SIZE"], r["length_equal"]])
    with open(out / "rope_equivalence_summary.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["case", "BLOCK_SIZE", "rope_equal"])
        for r in results:
            w.writerow([r["case"], r["BLOCK_SIZE"], r["rope_equal"]])

    considered = [r for r in results if r["status"] != "SKIP"]
    all_pass = bool(considered) and all(r["status"] == "PASS" for r in considered)
    status = {"status": "PASS" if all_pass else "FAIL",
              "num_cases": len(results),
              "passing": sum(1 for r in results if r["status"] == "PASS"),
              "skipped": sum(1 for r in results if r["status"] == "SKIP")}
    (out / "stage_status.json").write_text(json.dumps(status, indent=2))
    print(f"\n[Stage6] overall: {status['status']}  "
          f"{status['passing']}/{len(results)} pass  ({status['skipped']} skipped)")

    if not all_pass:
        for r in results:
            if r["status"] == "FAIL":
                print(f"  FAIL: {r}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
