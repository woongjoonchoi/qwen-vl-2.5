#!/usr/bin/env python3
"""
FA2 vs GUIDE comparative inference worker.

Runs a shard of benchmark_manifest_500_docker.jsonl and measures:
  VE latency, TTFT, TPOT, E2E generation time, accuracy (ANLS / relaxed_acc).

Usage (one GPU worker):
  python run_comparative_inference.py \
    --mode fa2|guide \
    --manifest /path/manifest.jsonl \
    --output-dir /path/out \
    --start-idx 0 --end-idx 125 \
    --gpu-id 0 --max-new-tokens 32 --precision bf16
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F


# ── text metrics ──────────────────────────────────────────────────────────────
import re, unicodedata

def normalize(s):
    s = unicodedata.normalize("NFKD", s).lower()
    s = re.sub(r"[^\w\s]", "", s).strip()
    return s

def anls(pred, gts, thr=0.5):
    from difflib import SequenceMatcher
    p = normalize(pred)
    best = 0.0
    for g in (gts if isinstance(gts, list) else [gts]):
        g = normalize(str(g))
        if max(len(p), len(g)) == 0:
            sim = 1.0
        else:
            sim = SequenceMatcher(None, p, g).ratio()
        score = sim if sim >= thr else 0.0
        best = max(best, score)
    return best

def relaxed_acc(pred, gts, tol=0.05):
    p = normalize(pred)
    for g in (gts if isinstance(gts, list) else [gts]):
        g = normalize(str(g))
        try:
            pv, gv = float(p.replace(",","")), float(g.replace(",",""))
            if gv == 0: match = abs(pv) < tol
            else: match = abs(pv - gv) / abs(gv) <= tol
            if match: return 1.0
        except ValueError:
            pass
        if p == g: return 1.0
    return 0.0


# ── model loading ─────────────────────────────────────────────────────────────
def load_model(model_path, mode, device):
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    dtype = torch.bfloat16
    proc = AutoProcessor.from_pretrained(model_path, local_files_only=True)

    print(f"[worker] loading model mode={mode} on {device}", flush=True)
    if mode == "guide":
        # Load with FA2 (used for full-attention layers)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=dtype, device_map=device,
            local_files_only=True, attn_implementation="flash_attention_2")
        # Patch vision encoder with GUIDE Triton kernel
        sys.path.insert(0, "/workspace"); sys.path.insert(0, "/workspace/tools")
        from tools.guide_model_patch import patch_model_with_guide
        patch_model_with_guide(model)
        print("[worker] GUIDE patch applied", flush=True)
    else:  # fa2
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=dtype, device_map=device,
            local_files_only=True, attn_implementation="flash_attention_2")

    model.eval()
    attn_impl = model.visual.blocks[0].attn.config._attn_implementation
    print(f"[worker] attn_impl={attn_impl}", flush=True)
    return model, proc


# ── single-sample inference + timing ─────────────────────────────────────────
def run_one(model, proc, sample: dict, max_new_tokens: int, device: str) -> dict:
    from transformers import LogitsProcessorList

    sys.path.insert(0, "/workspace/tools")
    from profile_qwen25vl_baseline import RequestProfiler, TTFTRecorder, make_gpu_timer

    img_path = sample["image_path"]
    question = sample.get("question_or_prompt", "")
    answer   = sample.get("answer_or_label", "")
    benchmark = sample.get("benchmark", "")

    # ── image load + processor ─────────────────────────────────────────────
    from PIL import Image
    from qwen_vl_utils import process_vision_info
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        return {"sample_id": sample.get("sample_id"), "status": f"IMG_ERR:{e}"}

    msgs = [{"role":"user","content":[
        {"type":"image","image":img},
        {"type":"text","text":question}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    img_in, vid_in = process_vision_info(msgs)
    inputs = proc(text=[text], images=img_in, videos=vid_in,
                  padding=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # ── grid dims & boundary flag ──────────────────────────────────────────
    grid_thw = inputs.get("image_grid_thw", None)
    if grid_thw is not None and grid_thw.shape[0] > 0:
        _T, _H, _W = grid_thw[0].cpu().tolist()
        _MH, _MW = int(_H) // 2, int(_W) // 2   # merge_size=2
        _is_boundary = (_MH % 4 != 0) or (_MW % 4 != 0)
        _n_vis_tok = _MH * _MW
    else:
        _H = _W = _MH = _MW = 0; _is_boundary = False; _n_vis_tok = 0
    _H, _W, _MH, _MW = int(_H), int(_W), int(_MH), int(_MW)

    # ── generate with timing hooks ─────────────────────────────────────────
    ttft_rec = TTFTRecorder(use_nvtx=False)
    lp_list  = LogitsProcessorList([ttft_rec])

    ev_start = torch.cuda.Event(enable_timing=True)
    ev_end   = torch.cuda.Event(enable_timing=True)

    with RequestProfiler(model, use_nvtx=False) as prof:
        torch.cuda.synchronize()
        ev_start.record()
        with torch.no_grad():
            out_ids = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=False, logits_processor=lp_list)
        ev_end.record()
        torch.cuda.synchronize()

    timings = prof.get_timings()

    # ── decode output ──────────────────────────────────────────────────────
    in_len = inputs["input_ids"].shape[1]
    pred_ids = out_ids[0, in_len:]
    pred_text = proc.decode(pred_ids, skip_special_tokens=True).strip()
    n_out = len(pred_ids)

    # ── derived timings ────────────────────────────────────────────────────
    total_ms   = ev_start.elapsed_time(ev_end)
    ve_ms      = timings.get("visual_encoder_ms", -1)
    prefill_ms = timings.get("llm_prefill_ms", -1)
    ttft_ms    = (ev_start.elapsed_time(ttft_rec.event)
                  if ttft_rec.recorded else -1)
    decode_ms  = max(0, total_ms - ttft_ms) if ttft_ms > 0 else -1
    tpot_ms    = decode_ms / max(n_out - 1, 1) if (decode_ms > 0 and n_out > 1) else -1
    tokens_per_s = n_out / (total_ms / 1000) if total_ms > 0 else -1

    # ── accuracy ──────────────────────────────────────────────────────────
    gts = answer if isinstance(answer, list) else [answer]
    if benchmark == "docvqa":
        score = anls(pred_text, gts)
        metric_name = "anls"
    elif benchmark == "chartqa":
        score = relaxed_acc(pred_text, gts)
        metric_name = "relaxed_acc"
    else:
        score = -1; metric_name = "none"

    return {
        "sample_id":    sample.get("sample_id"),
        "benchmark":    benchmark,
        "pred_text":    pred_text,
        "answer":       answer,
        metric_name:    round(score, 4),
        "ve_ms":        round(ve_ms, 3),
        "ttft_ms":      round(ttft_ms, 3),
        "prefill_ms":   round(prefill_ms, 3),
        "decode_ms":    round(decode_ms, 3),
        "tpot_ms":      round(tpot_ms, 3),
        "total_gen_ms": round(total_ms, 3),
        "tokens_per_s": round(tokens_per_s, 2),
        "n_out_tokens": n_out,
        "window_attn_total_ms": round(timings.get("window_attn_total_ms", -1), 3),
        "full_attn_total_ms":   round(timings.get("full_attn_total_ms", -1), 3),
        "grid_h": _H, "grid_w": _W,
        "merged_h": _MH, "merged_w": _MW,
        "n_visual_tokens": _n_vis_tok,
        "is_boundary": _is_boundary,
        "status": "OK",
    }


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode",           required=True, choices=["fa2","guide"])
    ap.add_argument("--manifest",       required=True)
    ap.add_argument("--output-dir",     required=True)
    ap.add_argument("--gpu-id",         type=int, default=0)
    ap.add_argument("--start-idx",      type=int, default=0)
    ap.add_argument("--end-idx",        type=int, default=9999)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--precision",      default="bf16")
    ap.add_argument("--model-path",     default="/workspace/models/Qwen2.5-VL-3B-Instruct")
    args = ap.parse_args()

    device = f"cuda:{args.gpu_id}"
    out    = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # log setup
    log_f = open(out / "progress.log", "a", buffering=1)
    def log(msg):
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        line = f"[{ts}] [gpu{args.gpu_id}] {msg}"
        print(line, flush=True); log_f.write(line + "\n")

    samples = [json.loads(l) for l in open(args.manifest)]
    shard = samples[args.start_idx : args.end_idx]
    log(f"mode={args.mode} shard=[{args.start_idx},{args.end_idx}) n={len(shard)}")

    model, proc = load_model(args.model_path, args.mode, device)

    results_path = out / f"results_gpu{args.gpu_id}.jsonl"
    rf = open(results_path, "w", buffering=1)

    ok = err = 0
    for i, sample in enumerate(shard):
        global_idx = args.start_idx + i
        try:
            r = run_one(model, proc, sample, args.max_new_tokens, device)
        except Exception as e:
            r = {"sample_id": sample.get("sample_id"),
                 "benchmark": sample.get("benchmark",""),
                 "status": f"ERROR:{str(e)[:120]}"}
            traceback.print_exc()
        rf.write(json.dumps(r) + "\n")
        ok  += r.get("status") == "OK"
        err += r.get("status", "").startswith("ERROR")
        if (i + 1) % 10 == 0 or i == 0:
            log(f"[{i+1}/{len(shard)}] ok={ok} err={err} "
                f"sample={r.get('sample_id')} ve={r.get('ve_ms',-1):.1f}ms "
                f"ttft={r.get('ttft_ms',-1):.1f}ms score={r.get('anls', r.get('relaxed_acc','?'))}")

    rf.close()
    log(f"DONE ok={ok} err={err} -> {results_path}")


if __name__ == "__main__":
    main()
