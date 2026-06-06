#!/usr/bin/env python3
"""
Qwen2.5-VL Evaluation Pipeline
Benchmarks: DocVQA (ANLS), ChartQA (Relaxed Accuracy), OCRBench v2 (Exact Match)
"""
import argparse
import csv
import difflib
import json
import os
import re
import sys
import time
from pathlib import Path

import torch


# ─── Metrics ─────────────────────────────────────────────────────────────────

def normalize_text(s: str) -> str:
    """Lowercase, strip punctuation/spaces."""
    s = s.strip().lower()
    s = re.sub(r'\s+', ' ', s)
    return s


def anls_score(pred: str, gts: list, threshold: float = 0.5) -> float:
    """Average Normalized Levenshtein Similarity for DocVQA."""
    pred_n = normalize_text(pred)
    best = 0.0
    for gt in gts:
        gt_n = normalize_text(str(gt))
        if len(pred_n) == 0 and len(gt_n) == 0:
            score = 1.0
        elif len(pred_n) == 0 or len(gt_n) == 0:
            score = 0.0
        else:
            # Use SequenceMatcher as approximation of NL distance
            sim = difflib.SequenceMatcher(None, pred_n, gt_n).ratio()
            score = sim
        best = max(best, score)
    return best if best >= threshold else 0.0


def relaxed_accuracy(pred: str, gts: list, tolerance: float = 0.05) -> bool:
    """ChartQA relaxed accuracy: exact match + numeric tolerance."""
    pred_n = normalize_text(pred)
    # Strip trailing period/spaces
    pred_n = pred_n.rstrip('.').strip()
    for gt in gts:
        gt_n = normalize_text(str(gt)).rstrip('.').strip()
        if pred_n == gt_n:
            return True
        # Check if pred contains gt (for verbose answers)
        if gt_n in pred_n or pred_n in gt_n:
            return True
        # Numeric tolerance
        try:
            p_num = float(pred_n.replace('%', '').replace(',', '').replace('$', ''))
            g_num = float(gt_n.replace('%', '').replace(',', '').replace('$', ''))
            denom = max(abs(g_num), 1e-6)
            if abs(p_num - g_num) / denom <= tolerance:
                return True
        except (ValueError, TypeError):
            pass
    return False


def exact_match(pred: str, gts: list) -> bool:
    """Exact match after normalization."""
    pred_n = normalize_text(pred)
    for gt in gts:
        gt_n = normalize_text(str(gt))
        if pred_n == gt_n:
            return True
        if gt_n in pred_n or pred_n in gt_n:
            return True
    return False


# ─── Model ────────────────────────────────────────────────────────────────────

def load_model(model_path: str, precision: str = "bf16", device: str = "cuda",
               attn_impl: str = "sdpa"):
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    import sys
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
    print(f"[eval] Loading processor from {model_path}", flush=True)
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    print(f"[eval] Loading model ({precision}, attn={attn_impl})...", flush=True)
    load_kw = dict(torch_dtype=dtype, device_map=device, local_files_only=True)
    if attn_impl in ("fa2", "guide"):
        load_kw["attn_implementation"] = "flash_attention_2"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, **load_kw)
    model.eval()
    if attn_impl == "guide":
        sys.path.insert(0, str(Path(__file__).parent))
        from guide_model_patch import patch_model_with_guide
        patch_model_with_guide(model)
        print("[eval] GUIDE patch applied.", flush=True)
    print(f"[eval] Model loaded: {sum(p.numel() for p in model.parameters())/1e9:.2f}B params", flush=True)
    return model, processor


def run_inference(model, processor, image, question: str,
                  max_new_tokens: int = 32, device: str = "cuda") -> str:
    """Single-sample inference with short-answer prompt."""
    from qwen_vl_utils import process_vision_info
    from PIL import Image as PILImage

    if isinstance(image, (str, Path)):
        img = PILImage.open(str(image)).convert("RGB")
    else:
        img = image.convert("RGB")

    messages = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text",  "text": question},
    ]}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    in_len = inputs["input_ids"].shape[1]
    return processor.batch_decode(out_ids[:, in_len:], skip_special_tokens=True)[0].strip()


# ─── DocVQA ───────────────────────────────────────────────────────────────────

DOCVQA_PROMPT = (
    "Answer the question using a single word or short phrase. "
    "Do not add explanation.\nQuestion: {question}"
)


def load_docvqa(data_root: str, split: str = "val", num_samples: int = 500):
    ann_map = {"val": "val_v1.0_withQT.json", "train": "train_v1.0_withQT.json", "test": "test_v1.0.json"}
    ann_path = Path(data_root) / "docvqa" / "annotations" / ann_map[split]
    img_root = Path(data_root) / "docvqa" / "images"

    with open(ann_path) as f:
        data = json.load(f)["data"]

    samples = []
    for item in data:
        img_fname = os.path.basename(item["image"])
        img_path = img_root / img_fname
        if not img_path.exists():
            continue
        samples.append({
            "sample_id": f"docvqa_{item['questionId']}",
            "image_path": str(img_path),
            "question": item["question"],
            "answers": item.get("answers", []),
        })
        if len(samples) >= num_samples:
            break

    return samples


def eval_docvqa(model, processor, data_root: str, num_samples: int,
                max_new_tokens: int, output_dir: Path, device: str,
                shard_idx: int = 0, num_shards: int = 1, shard_sfx: str = ""):
    samples = load_docvqa(data_root, "val", num_samples)
    if num_shards > 1:
        samples = [s for i, s in enumerate(samples) if i % num_shards == shard_idx]
    print(f"[docvqa] {len(samples)} samples", flush=True)

    results = []
    anls_scores = []

    for i, s in enumerate(samples):
        prompt = DOCVQA_PROMPT.format(question=s["question"])
        try:
            pred = run_inference(model, processor, s["image_path"],
                                  prompt, max_new_tokens, device)
        except Exception as e:
            pred = ""
            print(f"  [WARN] sample {i}: {e}", flush=True)

        score = anls_score(pred, s["answers"])
        anls_scores.append(score)

        results.append({
            "sample_id": s["sample_id"],
            "question": s["question"],
            "pred": pred,
            "gt": s["answers"],
            "anls": round(score, 4),
        })

        if (i + 1) % 50 == 0 or i < 5:
            running_mean = sum(anls_scores) / len(anls_scores)
            print(f"  [{i+1}/{len(samples)}] ANLS={running_mean:.4f} | "
                  f"pred={pred[:40]!r} | gt={s['answers'][0] if s['answers'] else '?'}",
                  flush=True)

    mean_anls = sum(anls_scores) / len(anls_scores) if anls_scores else 0
    print(f"[docvqa] Final ANLS: {mean_anls:.4f} (n={len(anls_scores)})", flush=True)

    _save_results(output_dir, f"docvqa{shard_sfx}", results,
                  {"benchmark": "docvqa", "split": "val", "shard": shard_sfx,
                   "num_samples": len(results), "mean_anls": round(mean_anls, 4),
                   "metric": "ANLS"})
    return mean_anls


# ─── ChartQA ─────────────────────────────────────────────────────────────────

CHARTQA_PROMPT = (
    "Answer with a single word or number. Do not explain.\nQuestion: {question}"
)


def load_chartqa(data_root: str, split: str = "val", num_samples: int = 500):
    import pandas as pd
    data_dir = Path(data_root) / "chartqa" / "data"
    img_save_dir = Path(data_root) / "chartqa" / "_extracted_images"
    img_save_dir.mkdir(exist_ok=True)

    parquet_files = sorted(data_dir.glob(f"{split}-*.parquet"))
    if not parquet_files:
        parquet_files = sorted(data_dir.glob("*.parquet"))

    samples = []
    for pf in parquet_files:
        df = pd.read_parquet(pf)
        for i, row in df.iterrows():
            if len(samples) >= num_samples:
                break
            img_data = row.get("image", None)
            img_path = None
            if img_data is not None:
                if isinstance(img_data, dict) and "bytes" in img_data:
                    img_bytes = img_data["bytes"]
                elif isinstance(img_data, bytes):
                    img_bytes = img_data
                else:
                    img_bytes = None
                if img_bytes:
                    img_file = img_save_dir / f"{pf.stem}_{i}.png"
                    if not img_file.exists():
                        img_file.write_bytes(img_bytes)
                    img_path = str(img_file)

            if img_path is None:
                continue

            label = row.get("label", row.get("answer", ["?"]))
            # Parquet label이 numpy array로 저장되어 ['Blue'] 형태로 옴
            # pandas가 numpy array를 그대로 반환 → list 변환
            if hasattr(label, 'tolist'):   # numpy array
                label = label.tolist()
            if isinstance(label, str):
                label = [label]
            elif not isinstance(label, list):
                label = [str(label)]
            # 각 원소도 numpy scalar일 수 있음
            label = [str(v) for v in label]

            # human_or_machine: 0=human, 1=machine (정수형)
            hom_raw = row.get("human_or_machine", -1)
            hom_int = int(hom_raw) if str(hom_raw).lstrip('-').isdigit() else -1
            split_type = "human" if hom_int == 0 else ("machine" if hom_int == 1 else str(hom_raw))

            samples.append({
                "sample_id": f"chartqa_{pf.stem}_{i}",
                "image_path": img_path,
                "question": str(row.get("query", row.get("question", ""))),
                "answers": label,
                "split_type": split_type,
            })
        if len(samples) >= num_samples:
            break

    return samples


def eval_chartqa(model, processor, data_root: str, num_samples: int,
                 max_new_tokens: int, output_dir: Path, device: str,
                 shard_idx: int = 0, num_shards: int = 1, shard_sfx: str = ""):
    samples = load_chartqa(data_root, "val", num_samples)
    if num_shards > 1:
        samples = [s for i, s in enumerate(samples) if i % num_shards == shard_idx]
    print(f"[chartqa] {len(samples)} samples", flush=True)

    results = []
    correct = 0
    human_correct = human_total = 0
    machine_correct = machine_total = 0

    for i, s in enumerate(samples):
        prompt = CHARTQA_PROMPT.format(question=s["question"])
        try:
            pred = run_inference(model, processor, s["image_path"],
                                  prompt, max_new_tokens, device)
        except Exception as e:
            pred = ""
            print(f"  [WARN] sample {i}: {e}", flush=True)

        hit = relaxed_accuracy(pred, s["answers"])
        if hit:
            correct += 1
        if s["split_type"] == "human":
            human_total += 1
            if hit: human_correct += 1
        else:
            machine_total += 1
            if hit: machine_correct += 1

        results.append({
            "sample_id": s["sample_id"],
            "question": s["question"],
            "pred": pred,
            "gt": s["answers"],
            "correct": hit,
            "split_type": s["split_type"],
        })

        if (i + 1) % 50 == 0 or i < 5:
            acc = correct / (i + 1)
            print(f"  [{i+1}/{len(samples)}] Acc={acc:.4f} | "
                  f"pred={pred[:30]!r} | gt={s['answers']}", flush=True)

    overall_acc = correct / len(results) if results else 0
    human_acc   = human_correct / human_total if human_total else 0
    machine_acc = machine_correct / machine_total if machine_total else 0

    print(f"[chartqa] Overall: {overall_acc:.4f}  Human: {human_acc:.4f}  Machine: {machine_acc:.4f}", flush=True)

    _save_results(output_dir, f"chartqa{shard_sfx}", results, {
        "benchmark": "chartqa", "split": "val", "shard": shard_sfx,
        "num_samples": len(results),
        "overall_relaxed_acc": round(overall_acc, 4),
        "human_relaxed_acc": round(human_acc, 4),
        "machine_relaxed_acc": round(machine_acc, 4),
        "metric": "relaxed_accuracy",
    })
    return overall_acc


# ─── OCRBench v2 ──────────────────────────────────────────────────────────────

OCRBENCH_PROMPT = (
    "Answer briefly with a word or short phrase. Do not explain.\nQuestion: {question}"
)


def load_ocrbench(data_root: str, num_samples: int = 500):
    import pandas as pd
    data_dir = Path(data_root) / "ocrbench_v2" / "data"
    img_save_dir = Path(data_root) / "ocrbench_v2" / "_extracted_images"
    img_save_dir.mkdir(exist_ok=True)

    samples = []
    for pf in sorted(data_dir.glob("test-*.parquet")):
        if len(samples) >= num_samples:
            break
        df = pd.read_parquet(pf)
        for i, row in df.iterrows():
            if len(samples) >= num_samples:
                break
            img_data = row.get("image", None)
            img_path = None
            if img_data is not None:
                if isinstance(img_data, dict) and "bytes" in img_data:
                    img_bytes = img_data["bytes"]
                elif isinstance(img_data, bytes):
                    img_bytes = img_data
                else:
                    img_bytes = None
                if img_bytes:
                    img_file = img_save_dir / f"{pf.stem}_{i}.png"
                    if not img_file.exists():
                        img_file.write_bytes(img_bytes)
                    img_path = str(img_file)

            if img_path is None:
                continue

            answers = row.get("answers", row.get("answer", []))
            if hasattr(answers, 'tolist'):
                answers = answers.tolist()
            if isinstance(answers, str):
                answers = [answers]
            elif not isinstance(answers, list):
                answers = [str(answers)]
            answers = [str(v) for v in answers]

            samples.append({
                "sample_id": f"ocr_{row.get('id', f'{pf.stem}_{i}')}",
                "image_path": img_path,
                "question": str(row.get("question", "")),
                "answers": answers,
                "dataset_name": str(row.get("dataset_name", "")),
                "type": str(row.get("type", "")),
            })

    return samples


def eval_ocrbench(model, processor, data_root: str, num_samples: int,
                  max_new_tokens: int, output_dir: Path, device: str,
                  shard_idx: int = 0, num_shards: int = 1, shard_sfx: str = ""):
    samples = load_ocrbench(data_root, num_samples)
    if num_shards > 1:
        samples = [s for i, s in enumerate(samples) if i % num_shards == shard_idx]
    print(f"[ocrbench_v2] {len(samples)} samples", flush=True)

    results = []
    correct_em = 0
    correct_anls = 0.0
    by_type: dict = {}

    for i, s in enumerate(samples):
        prompt = OCRBENCH_PROMPT.format(question=s["question"])
        try:
            pred = run_inference(model, processor, s["image_path"],
                                  prompt, max_new_tokens, device)
        except Exception as e:
            pred = ""
            print(f"  [WARN] sample {i}: {e}", flush=True)

        em   = exact_match(pred, s["answers"])
        a    = anls_score(pred, s["answers"])
        if em: correct_em += 1
        correct_anls += a

        dtype = s.get("type", "unknown")
        if dtype not in by_type:
            by_type[dtype] = {"correct": 0, "total": 0}
        by_type[dtype]["total"] += 1
        if em:
            by_type[dtype]["correct"] += 1

        results.append({
            "sample_id": s["sample_id"],
            "question": s["question"],
            "pred": pred,
            "gt": s["answers"],
            "exact_match": em,
            "anls": round(a, 4),
            "dataset_name": s.get("dataset_name", ""),
            "type": dtype,
        })

        if (i + 1) % 50 == 0 or i < 5:
            em_acc = correct_em / (i + 1)
            print(f"  [{i+1}/{len(samples)}] EM={em_acc:.4f} | "
                  f"pred={pred[:30]!r} | gt={s['answers'][:1]}", flush=True)

    em_acc   = correct_em / len(results) if results else 0
    anls_avg = correct_anls / len(results) if results else 0

    print(f"[ocrbench_v2] EM: {em_acc:.4f}  ANLS: {anls_avg:.4f}", flush=True)
    for t, v in sorted(by_type.items()):
        print(f"  type={t}: {v['correct']}/{v['total']} = {v['correct']/v['total']:.3f}")

    _save_results(output_dir, f"ocrbench_v2{shard_sfx}", results, {
        "benchmark": "ocrbench_v2", "split": "test", "shard": shard_sfx,
        "num_samples": len(results),
        "exact_match": round(em_acc, 4),
        "anls": round(anls_avg, 4),
        "metric": "exact_match+anls",
        "by_type": {t: round(v["correct"]/v["total"], 4) for t, v in by_type.items() if v["total"] > 0},
    })
    return em_acc


# ─── Save helpers ─────────────────────────────────────────────────────────────

def _save_results(output_dir: Path, benchmark: str, results: list, summary: dict):
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSONL predictions
    pred_path = output_dir / f"{benchmark}_predictions.jsonl"
    with open(pred_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Summary JSON
    summary_path = output_dir / f"{benchmark}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # CSV
    if results:
        csv_path = output_dir / f"{benchmark}_results.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[k for k in results[0] if k != "gt"])
            w.writeheader()
            for r in results:
                row = {k: v for k, v in r.items() if k != "gt"}
                row["gt_str"] = "|".join(str(g) for g in (r.get("gt") or []))
                w = csv.DictWriter(f, fieldnames=list(row.keys()))
                break
        # Redo properly
        csv_path = output_dir / f"{benchmark}_results.csv"
        fieldnames = list(results[0].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for r in results:
                row = dict(r)
                if "gt" in row:
                    row["gt"] = "|".join(str(g) for g in (row["gt"] or []))
                w.writerow(row)

    print(f"  → saved: {pred_path.name}, {summary_path.name}", flush=True)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-VL Evaluation Pipeline")
    parser.add_argument("--model-path",   required=True)
    parser.add_argument("--data-root",    required=True)
    parser.add_argument("--output-dir",   required=True)
    parser.add_argument("--benchmarks",   nargs="+",
                        default=["docvqa", "chartqa", "ocrbench_v2"],
                        choices=["docvqa", "chartqa", "ocrbench_v2"])
    parser.add_argument("--num-samples",  type=int, default=0,
                        help="Max samples per benchmark (0 = full dataset)")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--precision",    default="bf16",
                        choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--device",       default="cuda",
                        help="Device string, e.g. cuda, cuda:0, cuda:1")
    parser.add_argument("--shard-idx",    type=int, default=0,
                        help="Shard index for dataset splitting (0-based)")
    parser.add_argument("--num-shards",   type=int, default=1,
                        help="Total number of shards to split dataset into")
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--attn-impl",    default="sdpa",
                        choices=["sdpa", "fa2", "guide"],
                        help="sdpa=HF default, fa2=FlashAttention-2, guide=GUIDE Triton")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(out_dir / "eval_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    model, processor = load_model(args.model_path, args.precision, device=args.device,
                                   attn_impl=getattr(args, "attn_impl", "sdpa"))

    num_s = args.num_samples if args.num_samples > 0 else 999999
    all_metrics = {}

    shard_idx  = args.shard_idx
    num_shards = args.num_shards
    device     = args.device

    # shard suffix for output filenames (blank when no sharding)
    shard_sfx = f"_shard{shard_idx}of{num_shards}" if num_shards > 1 else ""

    for bm in args.benchmarks:
        print(f"\n{'='*60}", flush=True)
        print(f" Evaluating: {bm.upper()} "
              f"{'(shard %d/%d)' % (shard_idx, num_shards) if num_shards>1 else ''}", flush=True)
        print(f"{'='*60}", flush=True)
        bm_dir = out_dir / bm
        t0 = time.perf_counter()

        try:
            if bm == "docvqa":
                score = eval_docvqa(model, processor, args.data_root, num_s,
                                     args.max_new_tokens, bm_dir, device,
                                     shard_idx=shard_idx, num_shards=num_shards,
                                     shard_sfx=shard_sfx)
                all_metrics[bm] = {"ANLS": round(score, 4)}

            elif bm == "chartqa":
                score = eval_chartqa(model, processor, args.data_root, num_s,
                                      args.max_new_tokens, bm_dir, device,
                                      shard_idx=shard_idx, num_shards=num_shards,
                                      shard_sfx=shard_sfx)
                all_metrics[bm] = {"relaxed_acc": round(score, 4)}

            elif bm == "ocrbench_v2":
                score = eval_ocrbench(model, processor, args.data_root, num_s,
                                       args.max_new_tokens, bm_dir, device,
                                       shard_idx=shard_idx, num_shards=num_shards,
                                       shard_sfx=shard_sfx)
                all_metrics[bm] = {"exact_match": round(score, 4)}

        except Exception as e:
            print(f"  [ERROR] {bm}: {e}", flush=True)
            import traceback; traceback.print_exc()
            all_metrics[bm] = {"error": str(e)}

        elapsed = time.perf_counter() - t0
        print(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f} min)", flush=True)

    # Final summary
    print(f"\n{'='*60}", flush=True)
    print(" EVALUATION SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    for bm, metrics in all_metrics.items():
        print(f"  {bm:<15}: {metrics}", flush=True)

    summary_path = out_dir / "eval_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "model_path": args.model_path,
            "num_samples": args.num_samples,
            "max_new_tokens": args.max_new_tokens,
            "precision": args.precision,
            "results": all_metrics,
        }, f, indent=2)
    print(f"\n  Written: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
