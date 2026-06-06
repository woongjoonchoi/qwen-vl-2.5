#!/usr/bin/env python3
"""
Qwen2.5-VL baseline profiling script.
Modes: inference_smoke | app_timing | nsys_profile | structure_inspect
"""
import argparse
import csv
import json
import math
import os
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import torch


# ─── helpers ────────────────────────────────────────────────────────────────

def ts():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def log(msg):
    print(f"[{ts()}] {msg}", flush=True)


DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


@contextmanager
def nvtx_range(name: str, enabled: bool = True):
    if enabled:
        torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        if enabled:
            torch.cuda.nvtx.range_pop()


def make_gpu_timer():
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    return s, e


def elapsed_ms(start_ev, end_ev):
    return start_ev.elapsed_time(end_ev)


# ─── model loader ────────────────────────────────────────────────────────────

def load_model_and_processor(model_path: str, precision: str,
                              device: str = "cuda", attn_impl: str = "sdpa"):
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    dtype = DTYPE_MAP[precision]
    log(f"Loading processor from {model_path} ...")
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)

    log(f"Loading model from {model_path} (dtype={precision}, attn={attn_impl}) ...")
    load_kwargs = dict(torch_dtype=dtype, device_map=device, local_files_only=True)
    if attn_impl in ("fa2", "guide"):
        load_kwargs["attn_implementation"] = "flash_attention_2"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, **load_kwargs)
    model.eval()

    if attn_impl == "guide":
        import sys; sys.path.insert(0, str(Path(__file__).parent))
        from guide_model_patch import patch_model_with_guide
        patch_model_with_guide(model)
        log("GUIDE patch applied.")

    impl = model.visual.blocks[0].attn.config._attn_implementation
    log(f"Model loaded. {sum(p.numel() for p in model.parameters())/1e9:.2f}B  attn_impl={impl}")
    return model, processor


# ─── request builder ─────────────────────────────────────────────────────────

def build_request(image_path: str, question: str, processor,
                  target_resolution: int, device: str):
    from qwen_vl_utils import process_vision_info
    from PIL import Image

    # Resize image to target_resolution x target_resolution for deterministic token counts
    img = Image.open(image_path).convert("RGB")
    if target_resolution > 0:
        # Round to nearest multiple of 28 (patch_size * merge_size = 14 * 2)
        res = int(round(target_resolution / 28)) * 28
        img = img.resize((res, res), Image.LANCZOS)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": question},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    return {k: v.to(device) for k, v in inputs.items()}


def count_visual_tokens(inputs, model):
    """Count visual tokens before and after merger."""
    pv = inputs.get("pixel_values", None)
    grid = inputs.get("image_grid_thw", None)
    if pv is None or grid is None:
        return 0, 0
    # Before merger: total patches
    tokens_before = pv.shape[0]  # each row is one patch
    # After merger: tokens_before / (merge_size^2)
    merge = getattr(model.visual, "spatial_merge_size", 2)
    if hasattr(model.visual, "merger"):
        # merger halves spatial dims twice → /4
        tokens_after = tokens_before // (merge * merge)
    else:
        tokens_after = tokens_before
    return tokens_before, tokens_after


# ─── timing hooks ────────────────────────────────────────────────────────────

class RequestProfiler:
    """
    Attaches hooks to model for fine-grained GPU timing.
    Usage:
        with RequestProfiler(model, use_nvtx=True) as prof:
            outputs = model.generate(...)
        timings = prof.get_timings()
    """

    def __init__(self, model, use_nvtx: bool = False):
        self.model = model
        self.use_nvtx = use_nvtx
        self._hooks = []
        self._events = {}
        self._llm_call = 0
        self._block_events = {}

    def __enter__(self):
        m = self.model
        cfg = m.config.vision_config
        fullatt_blocks = set(cfg.fullatt_block_indexes)
        depth = cfg.depth

        def mk(name):
            s, e = make_gpu_timer()
            self._events[name] = (s, e)

        # Visual encoder total
        mk("visual_encoder")
        def pre_ve(mod, args):
            self._events["visual_encoder"][0].record()
            if self.use_nvtx:
                torch.cuda.nvtx.range_push("visual_encoder")
        def post_ve(mod, args, out):
            if self.use_nvtx:
                torch.cuda.nvtx.range_pop()
            self._events["visual_encoder"][1].record()
        self._hooks.append(m.visual.register_forward_pre_hook(pre_ve))
        self._hooks.append(m.visual.register_forward_hook(post_ve))

        # Visual merger
        if hasattr(m.visual, "merger"):
            mk("visual_merger")
            def pre_mg(mod, args):
                self._events["visual_merger"][0].record()
                if self.use_nvtx:
                    torch.cuda.nvtx.range_push("visual_merger")
            def post_mg(mod, args, out):
                if self.use_nvtx:
                    torch.cuda.nvtx.range_pop()
                self._events["visual_merger"][1].record()
            self._hooks.append(m.visual.merger.register_forward_pre_hook(pre_mg))
            self._hooks.append(m.visual.merger.register_forward_hook(post_mg))

        # Patch embedding
        if hasattr(m.visual, "patch_embed"):
            mk("patch_embed")
            def pre_pe(mod, args):
                self._events["patch_embed"][0].record()
                if self.use_nvtx:
                    torch.cuda.nvtx.range_push("vit.patch_embed")
            def post_pe(mod, args, out):
                if self.use_nvtx:
                    torch.cuda.nvtx.range_pop()
                self._events["patch_embed"][1].record()
            self._hooks.append(m.visual.patch_embed.register_forward_pre_hook(pre_pe))
            self._hooks.append(m.visual.patch_embed.register_forward_hook(post_pe))

        # ViT blocks: window_attn_total and full_attn_total
        mk("window_attn_total")
        mk("full_attn_total")
        self._wa_list = []  # (start_ev, end_ev)
        self._fa_list = []

        if hasattr(m.visual, "blocks"):
            for i, block in enumerate(m.visual.blocks):
                bs, be = make_gpu_timer()
                label = "full" if i in fullatt_blocks else "win"
                bname = f"vit_block_{i:02d}_{label}"
                self._block_events[bname] = (bs, be)
                if label == "full":
                    self._fa_list.append((bs, be))
                else:
                    self._wa_list.append((bs, be))

                is_full = i in fullatt_blocks
                nvtx_name = f"vit.blocks.{i:02d}.{'full_attn' if is_full else 'window_attn'}"

                def make_hooks(s_ev, e_ev, nname, is_fa):
                    def pre_b(mod, args):
                        s_ev.record()
                        if self.use_nvtx:
                            torch.cuda.nvtx.range_push(nname)
                    def post_b(mod, args, out):
                        if self.use_nvtx:
                            torch.cuda.nvtx.range_pop()
                        e_ev.record()
                    return pre_b, post_b

                pre_b, post_b = make_hooks(bs, be, nvtx_name, is_full)
                self._hooks.append(block.register_forward_pre_hook(pre_b))
                self._hooks.append(block.register_forward_hook(post_b))

        # LLM model (first call = prefill, subsequent = decode)
        mk("llm_prefill")
        self._llm_call = 0

        def pre_llm(mod, args, kwargs):
            if self._llm_call == 0:
                self._events["llm_prefill"][0].record()
                if self.use_nvtx:
                    torch.cuda.nvtx.range_push("llm_prefill")

        def post_llm(mod, args, kwargs, out):
            if self._llm_call == 0:
                if self.use_nvtx:
                    torch.cuda.nvtx.range_pop()
                self._events["llm_prefill"][1].record()
                self._llm_call += 1

        self._hooks.append(m.model.register_forward_pre_hook(pre_llm, with_kwargs=True))
        self._hooks.append(m.model.register_forward_hook(post_llm, with_kwargs=True))

        return self

    def __exit__(self, *_):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def get_timings(self):
        """Return dict of timing in ms. Call after torch.cuda.synchronize()."""
        result = {}
        for name, (s, e) in self._events.items():
            try:
                result[f"{name}_ms"] = s.elapsed_time(e)
            except Exception:
                result[f"{name}_ms"] = -1.0

        # Aggregate window/full attention totals
        # (block hooks may not fire for GUIDE path — guard with try/except)
        try:
            result["window_attn_total_ms"] = sum(
                s.elapsed_time(e) for s, e in self._wa_list)
        except Exception:
            result["window_attn_total_ms"] = -1.0
        try:
            result["full_attn_total_ms"] = sum(
                s.elapsed_time(e) for s, e in self._fa_list)
        except Exception:
            result["full_attn_total_ms"] = -1.0

        result["per_block_ms"] = {}
        for bname, (s, e) in self._block_events.items():
            try:
                result["per_block_ms"][bname] = s.elapsed_time(e)
            except Exception:
                result["per_block_ms"][bname] = -1.0

        return result


class TTFTRecorder:
    """Records GPU timestamp when first token logits are computed."""
    from transformers import LogitsProcessor

    def __init__(self, use_nvtx: bool = False):
        self.event = torch.cuda.Event(enable_timing=True)
        self.recorded = False
        self.use_nvtx = use_nvtx

    def __call__(self, input_ids, scores):
        if not self.recorded:
            self.event.record()
            self.recorded = True
            if self.use_nvtx:
                torch.cuda.nvtx.range_push("decode_loop")
        return scores


# ─── single-sample profiling ─────────────────────────────────────────────────

def profile_one_sample(
    model, processor, sample: dict, max_new_tokens: int,
    resolution: int, device: str, use_nvtx: bool = False, run_id: str = ""
) -> dict:
    from transformers import LogitsProcessorList

    image_path = sample["image_path"]
    question = sample.get("question_or_prompt", "Describe this image.")

    # ── image preprocess (CPU) ──
    t_pre0 = time.perf_counter()
    with nvtx_range("image_preprocess", use_nvtx):
        inputs = build_request(image_path, question, processor, resolution, device)
    torch.cuda.synchronize()
    t_pre1 = time.perf_counter()
    image_preprocess_ms = (t_pre1 - t_pre0) * 1000  # includes processor

    # Token counts
    tokens_before, tokens_after = count_visual_tokens(inputs, model)
    num_input_text_tokens = (inputs["input_ids"].shape[1] - tokens_after
                              if tokens_after > 0 else inputs["input_ids"].shape[1])

    # ── generate with hooks ──
    ttft_rec = TTFTRecorder(use_nvtx=use_nvtx)
    logits_proc = LogitsProcessorList([ttft_rec])

    ev_gen_start = torch.cuda.Event(enable_timing=True)
    ev_gen_end = torch.cuda.Event(enable_timing=True)

    with RequestProfiler(model, use_nvtx=use_nvtx) as prof:
        torch.cuda.synchronize()
        t_gen0 = time.perf_counter()
        ev_gen_start.record()

        with nvtx_range("request_total", use_nvtx):
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    logits_processor=logits_proc,
                    return_dict_in_generate=False,
                )

        ev_gen_end.record()
        torch.cuda.synchronize()
        t_gen1 = time.perf_counter()

    hook_timings = prof.get_timings()
    if use_nvtx:
        # Close decode_loop range if it was opened
        if ttft_rec.recorded:
            torch.cuda.nvtx.range_pop()

    # ── derive metrics ──
    e2e_ms = ev_gen_start.elapsed_time(ev_gen_end)
    ttft_ms = ev_gen_start.elapsed_time(ttft_rec.event) if ttft_rec.recorded else e2e_ms
    decode_loop_ms = max(e2e_ms - ttft_ms, 0.0)

    in_len = inputs["input_ids"].shape[1]
    actual_gen = output_ids.shape[1] - in_len
    actual_gen = max(actual_gen, 0)

    tpot_ms = decode_loop_ms / max(actual_gen - 1, 1) if actual_gen > 1 else decode_loop_ms
    request_output_tok_s = actual_gen / (e2e_ms / 1000) if e2e_ms > 0 else 0.0
    decode_only_tok_s = max(actual_gen - 1, 0) / (decode_loop_ms / 1000) if decode_loop_ms > 0 else 0.0

    ve_ms = hook_timings.get("visual_encoder_ms", -1.0)
    vm_ms = hook_timings.get("visual_merger_ms", -1.0)
    lp_ms = hook_timings.get("llm_prefill_ms", -1.0)
    request_total_ms = image_preprocess_ms + e2e_ms

    f_vision = ve_ms / e2e_ms if ve_ms > 0 and e2e_ms > 0 else 0.0
    f_visual_prefill = (ve_ms + lp_ms) / e2e_ms if (ve_ms > 0 and lp_ms > 0 and e2e_ms > 0) else 0.0
    f_decode = decode_loop_ms / e2e_ms if e2e_ms > 0 else 0.0

    # Decode generated text
    generated_text = processor.batch_decode(
        output_ids[:, in_len:], skip_special_tokens=True
    )[0] if actual_gen > 0 else ""

    return {
        "run_id": run_id or str(uuid.uuid4())[:8],
        "sample_id": sample.get("sample_id", ""),
        "benchmark": sample.get("benchmark", ""),
        "image_path": image_path,
        "num_input_text_tokens": num_input_text_tokens,
        "num_visual_tokens_before_merger": tokens_before,
        "num_visual_tokens_after_merger": tokens_after,
        "actual_generated_tokens": actual_gen,
        "image_preprocess_ms": round(image_preprocess_ms, 3),
        "processor_ms": round(image_preprocess_ms, 3),  # combined
        "visual_encoder_ms": round(ve_ms, 3),
        "visual_merger_ms": round(vm_ms, 3),
        "patch_embed_ms": round(hook_timings.get("patch_embed_ms", -1.0), 3),
        "window_attn_total_ms": round(hook_timings.get("window_attn_total_ms", -1.0), 3),
        "full_attn_total_ms": round(hook_timings.get("full_attn_total_ms", -1.0), 3),
        "llm_prefill_ms": round(lp_ms, 3),
        "ttft_ms": round(ttft_ms, 3),
        "decode_loop_ms": round(decode_loop_ms, 3),
        "tpot_ms": round(tpot_ms, 3),
        "e2e_ms": round(e2e_ms, 3),
        "request_total_ms": round(request_total_ms, 3),
        "request_output_tok_s": round(request_output_tok_s, 3),
        "decode_only_tok_s": round(decode_only_tok_s, 3),
        "f_vision": round(f_vision, 6),
        "f_visual_prefill": round(f_visual_prefill, 6),
        "f_decode": round(f_decode, 6),
        "per_block_ms": hook_timings.get("per_block_ms", {}),
        "generated_text": generated_text[:256],
        "status": "OK",
        "error_message": "",
    }


# ─── modes ───────────────────────────────────────────────────────────────────

def mode_inference_smoke(args, model, processor, manifest):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"[inference_smoke] output_dir={out_dir}")

    results = []
    errors = []
    samples = [s for s in manifest if args.benchmark == "any" or s["benchmark"] == args.benchmark]
    samples = samples[:args.num_samples]

    if not samples:
        log(f"WARNING: no samples found for benchmark={args.benchmark}")
        samples = manifest[:args.num_samples]

    for i, sample in enumerate(samples):
        log(f"  sample {i+1}/{len(samples)}: {sample['sample_id']}")
        try:
            rec = profile_one_sample(
                model, processor, sample,
                max_new_tokens=args.max_new_tokens,
                resolution=args.resolution,
                device="cuda",
                use_nvtx=False,
            )
            results.append(rec)
            log(f"    e2e={rec['e2e_ms']:.1f}ms  gen={rec['actual_generated_tokens']}tok  "
                f"ve={rec['visual_encoder_ms']:.1f}ms  f_vision={rec['f_vision']:.3f}")
        except Exception as e:
            log(f"    ERROR: {e}")
            errors.append({"sample_id": sample["sample_id"], "error": str(e)})

    # Save outputs
    with open(out_dir / "smoke_generated_outputs.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps({k: v for k, v in r.items() if k != "per_block_ms"}) + "\n")

    with open(out_dir / "smoke_timing.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps({
                "sample_id": r["sample_id"],
                "e2e_ms": r["e2e_ms"],
                "visual_encoder_ms": r["visual_encoder_ms"],
                "ttft_ms": r["ttft_ms"],
            }) + "\n")

    model_report = {
        "model_path": args.model_path,
        "model_load": "PASS",
        "processor_load": "PASS",
        "num_successful_samples": len(results),
        "num_errors": len(errors),
        "errors": errors,
    }
    with open(out_dir / "model_load_report.json", "w") as f:
        json.dump(model_report, f, indent=2)

    passed = len(results) >= 1 and all(r["actual_generated_tokens"] > 0 for r in results)
    status = {"status": "PASS" if passed else "FAIL",
              "num_successful_samples": len(results),
              "generated_text_non_empty": all(len(r["generated_text"]) > 0 for r in results)}
    with open(out_dir / "stage_status.json", "w") as f:
        json.dump(status, f, indent=2)

    log(f"[inference_smoke] {'PASS' if passed else 'FAIL'}: {len(results)}/{len(samples)} samples OK")
    if not passed:
        sys.exit(1)


def mode_app_timing(args, model, processor, manifest):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"[app_timing] output_dir={out_dir}")

    benchmark = args.benchmark if args.benchmark != "any" else None
    samples = [s for s in manifest if benchmark is None or s["benchmark"] == benchmark]
    if not samples:
        samples = manifest
    samples = samples[:args.num_samples]

    log(f"Samples selected: {len(samples)}")

    # Warmup
    log(f"Warmup {args.warmup} iters ...")
    for i in range(args.warmup):
        s = samples[i % len(samples)]
        try:
            profile_one_sample(model, processor, s,
                                max_new_tokens=args.max_new_tokens,
                                resolution=args.resolution,
                                device="cuda", use_nvtx=False)
        except Exception as e:
            log(f"  warmup {i}: {e}")

    # Measured
    log(f"Measuring {args.measured_iters} iters ...")
    all_records = []
    for i in range(args.measured_iters):
        s = samples[i % len(samples)]
        try:
            r = profile_one_sample(
                model, processor, s,
                max_new_tokens=args.max_new_tokens,
                resolution=args.resolution,
                device="cuda", use_nvtx=False,
                run_id=f"iter{i:03d}",
            )
            all_records.append(r)
            if i % 5 == 0:
                log(f"  iter {i}: e2e={r['e2e_ms']:.1f}ms ve={r['visual_encoder_ms']:.1f}ms "
                    f"ttft={r['ttft_ms']:.1f}ms tpot={r['tpot_ms']:.2f}ms f_v={r['f_vision']:.3f}")
        except Exception as e:
            log(f"  iter {i}: ERROR {e}")

    if not all_records:
        log("FAIL: no successful measurements")
        sys.exit(1)

    # Save raw
    attn_impl_tag = getattr(args, "attn_impl", "sdpa")
    with open(out_dir / "timing_raw.jsonl", "w") as f:
        for r in all_records:
            row = {k: v for k, v in r.items() if k != "per_block_ms"}
            row["attn_impl"] = attn_impl_tag
            row["benchmark"] = benchmark or "mixed"
            row["resolution"] = args.resolution
            row["max_new_tokens"] = args.max_new_tokens
            f.write(json.dumps(row) + "\n")

    # Compute summary stats
    def mean(vals):
        return sum(vals) / len(vals) if vals else 0.0

    def median(vals):
        s = sorted(vals)
        n = len(s)
        return s[n // 2] if n else 0.0

    def p99(vals):
        s = sorted(vals)
        idx = min(int(len(s) * 0.99), len(s) - 1)
        return s[idx] if s else 0.0

    metrics = ["e2e_ms", "visual_encoder_ms", "visual_merger_ms", "llm_prefill_ms",
               "ttft_ms", "decode_loop_ms", "tpot_ms", "request_total_ms",
               "request_output_tok_s", "decode_only_tok_s", "f_vision",
               "f_visual_prefill", "f_decode", "actual_generated_tokens",
               "num_visual_tokens_before_merger", "num_visual_tokens_after_merger"]

    summary = {}
    for m in metrics:
        vals = [r[m] for r in all_records if isinstance(r.get(m), (int, float)) and r[m] >= 0]
        if vals:
            summary[f"mean_{m}"] = round(mean(vals), 4)
            summary[f"median_{m}"] = round(median(vals), 4)
            summary[f"p99_{m}"] = round(p99(vals), 4)
            summary[f"n_{m}"] = len(vals)

    summary["benchmark"] = benchmark or "mixed"
    summary["resolution"] = args.resolution
    summary["max_new_tokens"] = args.max_new_tokens
    summary["num_measured_iters"] = len(all_records)
    summary["attn_impl"] = getattr(args, "attn_impl", "sdpa")

    # Save summary CSV
    with open(out_dir / "timing_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    # Save generated outputs
    with open(out_dir / "generated_outputs.jsonl", "w") as f:
        for r in all_records:
            f.write(json.dumps({
                "sample_id": r["sample_id"],
                "generated_text": r["generated_text"],
                "actual_generated_tokens": r["actual_generated_tokens"],
            }) + "\n")

    # Report
    with open(out_dir / "app_timing_report.md", "w") as f:
        f.write("# App-Level Timing Report\n\n")
        f.write(f"- benchmark: {benchmark}\n")
        f.write(f"- resolution: {args.resolution}\n")
        f.write(f"- max_new_tokens: {args.max_new_tokens}\n")
        f.write(f"- measured_iters: {len(all_records)}\n\n")
        f.write("## Key Metrics (mean)\n\n")
        for m in ["e2e_ms", "visual_encoder_ms", "ttft_ms", "tpot_ms", "f_vision", "decode_only_tok_s"]:
            v = summary.get(f"mean_{m}", "N/A")
            f.write(f"- `{m}`: {v}\n")

    status = {
        "status": "PASS",
        "request_total_ms": summary.get("mean_request_total_ms"),
        "visual_encoder_ms": summary.get("mean_visual_encoder_ms"),
        "e2e_ms": summary.get("mean_e2e_ms"),
        "ttft_ms": summary.get("mean_ttft_ms"),
        "f_vision": summary.get("mean_f_vision"),
    }
    with open(out_dir / "stage_status.json", "w") as f:
        json.dump(status, f, indent=2)

    log(f"[app_timing] DONE: mean_e2e={summary.get('mean_e2e_ms', '?')}ms "
        f"mean_f_vision={summary.get('mean_f_vision', '?')}")


def mode_nsys_profile(args, model, processor, manifest):
    """Run with NVTX ranges + cudaProfilerApi for nsys capture."""
    import ctypes

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"[nsys_profile] output_dir={out_dir}")

    benchmark = args.benchmark if args.benchmark != "any" else None
    samples = [s for s in manifest if benchmark is None or s["benchmark"] == benchmark]
    if not samples:
        samples = manifest
    samples = samples[:args.num_samples]

    # Warmup (without profiler capture)
    log(f"Warmup {args.warmup} iters (no profiler) ...")
    for i in range(args.warmup):
        s = samples[i % len(samples)]
        try:
            profile_one_sample(model, processor, s,
                                max_new_tokens=args.max_new_tokens,
                                resolution=args.resolution,
                                device="cuda", use_nvtx=False)
        except Exception as e:
            log(f"  warmup {i}: {e}")

    torch.cuda.synchronize()
    log("Starting cudaProfilerStart() ...")
    torch.cuda.cudart().cudaProfilerStart()

    results = []
    log(f"Profiling {args.profile_iters} iters ...")
    for i in range(args.profile_iters):
        s = samples[i % len(samples)]
        torch.cuda.nvtx.range_push(f"iteration_{i:02d}")
        try:
            r = profile_one_sample(
                model, processor, s,
                max_new_tokens=args.max_new_tokens,
                resolution=args.resolution,
                device="cuda", use_nvtx=True,
                run_id=f"prof{i:02d}",
            )
            results.append(r)
            log(f"  profile iter {i}: e2e={r['e2e_ms']:.1f}ms ve={r['visual_encoder_ms']:.1f}ms")
        except Exception as e:
            log(f"  profile iter {i}: ERROR {e}")
        finally:
            torch.cuda.nvtx.range_pop()

    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    log("cudaProfilerStop()")

    # Save results
    with open(out_dir / "nsys_profile_results.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps({k: v for k, v in r.items() if k != "per_block_ms"}) + "\n")

    status = {"status": "PASS" if results else "FAIL", "num_profiled": len(results)}
    with open(out_dir / "stage_status.json", "w") as f:
        json.dump(status, f, indent=2)

    log(f"[nsys_profile] DONE: {len(results)} iterations profiled")


def mode_structure_inspect(args, model, processor, manifest):
    """Inspect visual encoder structure."""
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = model.config.vision_config
    fullatt = set(cfg.fullatt_block_indexes)
    depth = cfg.depth

    log(f"ViT depth: {depth}")
    log(f"Full attention blocks: {sorted(fullatt)}")
    log(f"Window size: {cfg.window_size}")
    log(f"Patch size: {cfg.patch_size}")
    log(f"Spatial merge size: {cfg.spatial_merge_size}")
    log(f"Hidden size: {cfg.hidden_size}")
    log(f"Num heads: {cfg.num_heads}")

    blocks_info = []
    for i in range(depth):
        btype = "full_attention" if i in fullatt else "window_attention"
        blocks_info.append({"block_idx": i, "attention_type": btype})

    with open(out_dir / "block_type_map.json", "w") as f:
        json.dump(blocks_info, f, indent=2)

    # Visual token counts by resolution
    resolutions = [448, 896, 1344, 1792]
    token_shape_rows = []
    for res in resolutions:
        snap = int(round(res / 28)) * 28
        h_patches = snap // cfg.patch_size
        w_patches = snap // cfg.patch_size
        tokens_before = h_patches * w_patches
        tokens_after = tokens_before // (cfg.spatial_merge_size ** 2)
        token_shape_rows.append({
            "resolution_target": res,
            "actual_resolution": snap,
            "h_patches": h_patches,
            "w_patches": w_patches,
            "tokens_before_merger": tokens_before,
            "tokens_after_merger": tokens_after,
        })
        log(f"  res={res}: {tokens_before} → {tokens_after} tokens (after merger)")

    with open(out_dir / "visual_token_shape_by_resolution.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(token_shape_rows[0].keys()))
        w.writeheader()
        w.writerows(token_shape_rows)

    structure = {
        "depth": depth,
        "fullatt_block_indexes": sorted(fullatt),
        "window_size": cfg.window_size,
        "patch_size": cfg.patch_size,
        "spatial_merge_size": cfg.spatial_merge_size,
        "hidden_size": cfg.hidden_size,
        "num_heads": cfg.num_heads,
        "out_hidden_size": cfg.out_hidden_size,
    }
    with open(out_dir / "visual_encoder_structure.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(structure.keys()))
        w.writeheader()
        w.writerow(structure)

    with open(out_dir / "structure_report.md", "w") as f:
        f.write("# Qwen2.5-VL Visual Encoder Structure\n\n")
        for k, v in structure.items():
            f.write(f"- **{k}**: {v}\n")
        f.write("\n## Visual Token Count by Resolution\n\n")
        f.write("| Resolution | Actual | Before Merger | After Merger |\n")
        f.write("|---|---|---|---|\n")
        for row in token_shape_rows:
            f.write(f"| {row['resolution_target']} | {row['actual_resolution']} "
                    f"| {row['tokens_before_merger']} | {row['tokens_after_merger']} |\n")

    status = {"status": "PASS", "depth": depth, "fullatt_blocks": sorted(fullatt)}
    with open(out_dir / "stage_status.json", "w") as f:
        json.dump(status, f, indent=2)

    log("[structure_inspect] DONE")


# ─── main ────────────────────────────────────────────────────────────────────

def load_manifest(manifest_path: str, benchmark: str = "any"):
    samples = []
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            s = json.loads(line)
            if benchmark == "any" or s.get("benchmark") == benchmark:
                samples.append(s)
    return samples


def save_environment_snapshot(out_dir: Path, model_path: str, args):
    info = {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_count": torch.cuda.device_count(),
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "model_path": model_path,
    }
    try:
        import transformers
        info["transformers_version"] = transformers.__version__
    except ImportError:
        pass
    with open(out_dir / "environment_snapshot.txt", "w") as f:
        for k, v in info.items():
            f.write(f"{k}: {v}\n")
    return info


def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-VL baseline profiler")
    parser.add_argument("--mode", required=True,
                        choices=["inference_smoke", "app_timing", "nsys_profile",
                                 "structure_inspect"])
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--benchmark", default="docvqa")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--resolution", type=int, default=896)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--precision", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--measured-iters", type=int, default=10)
    parser.add_argument("--profile-iters", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attn-impl", default="sdpa",
                        choices=["sdpa", "fa2", "guide"],
                        help="sdpa=HF default, fa2=FlashAttention-2, guide=GUIDE Triton")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save command
    with open(out_dir / "run_command.txt", "w") as f:
        f.write(" ".join(sys.argv) + "\n")

    # Save environment snapshot
    env = save_environment_snapshot(out_dir, args.model_path, args)
    log(f"GPU: {env.get('gpu_names', ['?'])[0] if env.get('gpu_names') else '?'}")

    # Load manifest
    manifest = load_manifest(args.manifest,
                              benchmark="any" if args.mode == "structure_inspect" else args.benchmark)
    # For inference_smoke, accept any benchmark if the requested one is empty
    if not manifest and args.mode != "structure_inspect":
        manifest = load_manifest(args.manifest, benchmark="any")
    log(f"Manifest: {len(manifest)} samples (benchmark={args.benchmark})")

    # Load model
    attn_impl = getattr(args, "attn_impl", "sdpa")
    model, processor = load_model_and_processor(args.model_path, args.precision,
                                                 attn_impl=attn_impl)

    # Save config snapshot
    import yaml  # may not be available
    try:
        with open(out_dir / "stage_config.yaml", "w") as f:
            yaml.dump(vars(args), f)
    except ImportError:
        with open(out_dir / "stage_config.yaml", "w") as f:
            json.dump(vars(args), f, indent=2)

    # Dispatch mode
    if args.mode == "inference_smoke":
        mode_inference_smoke(args, model, processor, manifest)
    elif args.mode == "app_timing":
        mode_app_timing(args, model, processor, manifest)
    elif args.mode == "nsys_profile":
        mode_nsys_profile(args, model, processor, manifest)
    elif args.mode == "structure_inspect":
        mode_structure_inspect(args, model, processor, manifest)


if __name__ == "__main__":
    main()
