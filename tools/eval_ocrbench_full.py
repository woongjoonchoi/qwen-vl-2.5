#!/usr/bin/env python3
"""
OCRBench v2 Full Evaluation — task-specific max_new_tokens + prompts
Tasks: 30 types, 10,000 samples
"""
import argparse, json, re, sys, time
from pathlib import Path
from collections import defaultdict

import torch


# ─── Task-specific config (max_new_tokens + prompt template) ─────────────────

TASK_CONFIG = {
    # ── OCR Core ──────────────────────────────────────────────────────────────
    "text recognition en": {
        "max_tokens": 64,
        "prompt": "Read and output all text in the image exactly as written.",
    },
    "fine-grained text recognition en": {
        "max_tokens": 64,
        "prompt": "Read the specific text indicated. Output only the text.",
    },
    "handwritten answer extraction cn": {
        "max_tokens": 64,
        "prompt": "读取图像中的手写文字，直接输出内容。",
    },
    "full-page OCR en": {
        "max_tokens": 768,
        "prompt": "Read all text in the image from top to bottom. Output the complete text.",
    },
    "full-page OCR cn": {
        "max_tokens": 768,
        "prompt": "读取图像中的所有文字，从上到下输出完整内容。",
    },
    "text counting en": {
        "max_tokens": 16,
        "prompt": "Count the number of text instances. Answer with a single number.",
    },
    "text spotting en": {
        "max_tokens": 128,
        "prompt": "Detect all text in the image. Output each text and its bounding box as: text: [x1,y1,x2,y2]",
    },
    "text grounding en": {
        "max_tokens": 32,
        "prompt": "Find the bounding box of the specified text. Output 4 numbers: x1 y1 x2 y2",
    },
    # ── Cognition & Reasoning ─────────────────────────────────────────────────
    "cognition VQA en": {
        "max_tokens": 32,
        "prompt": "Answer briefly with a word or short phrase.",
    },
    "cognition VQA cn": {
        "max_tokens": 32,
        "prompt": "用一个词或短语简短回答。",
    },
    "reasoning VQA en": {
        "max_tokens": 48,
        "prompt": "Answer the question based on the text in the image. Be concise.",
    },
    "reasoning VQA cn": {
        "max_tokens": 48,
        "prompt": "根据图像中的文字回答问题，简明扼要。",
    },
    "VQA with position en": {
        "max_tokens": 64,
        "prompt": "Answer the question. If asked for position, give bounding box coordinates.",
    },
    "math QA en": {
        "max_tokens": 48,
        "prompt": "Answer the math question. Give only the final numerical answer.",
    },
    "science QA en": {
        "max_tokens": 32,
        "prompt": "Answer with a single word, letter (A/B/C/D), or short phrase.",
    },
    # ── Document & Structured ─────────────────────────────────────────────────
    "document parsing en": {
        "max_tokens": 384,
        "prompt": "Parse and extract the main structured content from this document image.",
    },
    "document parsing cn": {
        "max_tokens": 384,
        "prompt": "解析并提取该文档图像中的主要结构化内容。",
    },
    "document classification en": {
        "max_tokens": 16,
        "prompt": "Classify this document. Output only the category name.",
    },
    "key information extraction en": {
        "max_tokens": 384,
        "prompt": (
            "Extract key-value pairs from the image. "
            "Output as a JSON object with the exact field names requested. "
            "Use lists for values."
        ),
    },
    "key information extraction cn": {
        "max_tokens": 384,
        "prompt": "从图像中提取键值对。输出为JSON对象，使用列表存储值。",
    },
    "key information mapping en": {
        "max_tokens": 384,
        "prompt": (
            "Map the keys to their corresponding values from the image. "
            "Output as a JSON object with lists for values."
        ),
    },
    "table parsing en": {
        "max_tokens": 768,
        "prompt": "Convert this table image to HTML format. Output the complete HTML table.",
    },
    "table parsing cn": {
        "max_tokens": 768,
        "prompt": "将此表格图像转换为HTML格式，输出完整的HTML表格。",
    },
    "formula recognition en": {
        "max_tokens": 128,
        "prompt": (
            "Recognize the mathematical formula in the image. "
            "Output ONLY the LaTeX code, with spaces between tokens. "
            "Example: a ^ { 2 } + b ^ { 2 } = c ^ { 2 }"
        ),
    },
    "formula recognition cn": {
        "max_tokens": 128,
        "prompt": (
            "识别图像中的数学公式，仅输出LaTeX代码，标记之间用空格分隔。"
            "示例：a ^ { 2 } + b ^ { 2 } = c ^ { 2 }"
        ),
    },
    # ── Scene & Chart ─────────────────────────────────────────────────────────
    "chart parsing en": {
        "max_tokens": 512,
        "prompt": "Convert the chart to a nested Python dict with keys: title, x_title, y_title, values.",
    },
    "APP agent en": {
        "max_tokens": 64,
        "prompt": "Answer briefly based on the UI/app screenshot.",
    },
    "diagram QA en": {
        "max_tokens": 32,
        "prompt": "Answer briefly with a word or short phrase.",
    },
    # ── Multilingual ─────────────────────────────────────────────────────────
    "text translation cn": {
        "max_tokens": 128,
        "prompt": "翻译图像中的文字。直接输出翻译结果。",
    },
    # ── Misc ─────────────────────────────────────────────────────────────────
    "ASCII art classification en": {
        "max_tokens": 16,
        "prompt": "What object does this ASCII art represent? Answer with one word.",
    },
}

DEFAULT_CONFIG = {"max_tokens": 64, "prompt": "Answer briefly."}


# ─── Metrics ──────────────────────────────────────────────────────────────────

def extract_gt(gt_raw):
    s = str(gt_raw[0]) if isinstance(gt_raw, list) else str(gt_raw)
    matches = re.findall(r"'([^']+)'", s)
    return matches if matches else [s.strip("[]'\"")]

def norm(s):
    return re.sub(r'\s+', ' ', s.strip().lower()).rstrip('.,')

def text_match(pred, gts):
    p = norm(pred)
    for g in extract_gt(gts):
        g = norm(g)
        if p == g or g in p or p in g: return True
        try:
            pn = float(p.replace('%','').replace(',','').replace('$',''))
            gn = float(g.replace('%','').replace(',','').replace('$',''))
            if abs(pn-gn)/max(abs(gn),1e-6) <= 0.05: return True
        except: pass
    return False

def json_match(pred, gts):
    try:
        obj = json.loads(re.sub(r'```[a-z]*\n?', '', pred).strip())
    except:
        try:
            obj = eval(pred.replace('```json','').replace('```','').strip())
        except:
            return text_match(pred, gts)
    if not isinstance(obj, dict):
        return text_match(pred, gts)
    gt_str = ' '.join(extract_gt(gts)).lower()
    for k, v in obj.items():
        v = (v[0] if isinstance(v, list) and v else str(v))
        if norm(str(v)) and norm(str(v)) in gt_str:
            return True
    return False

def latex_match(pred, gts):
    p = re.sub(r'[\s\\]+', '', pred.lower().replace('$$','').replace('$',''))
    for g in extract_gt(gts):
        g = re.sub(r'[\s\\]+', '', g.lower())
        if not g: continue
        if p == g: return True
        if len(p) > 4 and p in g: return True
        if len(g) > 4 and g in p: return True
    return False

def coord_match(pred, gts):
    nums_p = [float(x) for x in re.findall(r'\d+\.?\d*', pred)][:4]
    for g in extract_gt(gts):
        nums_g = [float(x) for x in re.findall(r'\d+\.?\d*', g)][:4]
        if len(nums_p) == 4 and len(nums_g) == 4:
            if all(abs(p-g)/max(abs(g),1) <= 0.15 for p,g in zip(nums_p,nums_g)):
                return True
    return False

def overlap_match(pred, gts, threshold=0.35):
    p_words = set(norm(pred).split())
    if not p_words: return False
    for g in extract_gt(gts):
        g_words = set(norm(g).split())
        if not g_words: continue
        overlap = len(p_words & g_words) / len(g_words)
        if overlap >= threshold: return True
    return False

TASK_METRIC = {
    "text recognition en": text_match,
    "fine-grained text recognition en": text_match,
    "handwritten answer extraction cn": text_match,
    "full-page OCR en": overlap_match,
    "full-page OCR cn": overlap_match,
    "text counting en": text_match,
    "text spotting en": text_match,
    "text grounding en": coord_match,
    "cognition VQA en": text_match,
    "cognition VQA cn": text_match,
    "reasoning VQA en": text_match,
    "reasoning VQA cn": text_match,
    "VQA with position en": text_match,
    "math QA en": text_match,
    "science QA en": text_match,
    "document parsing en": overlap_match,
    "document parsing cn": overlap_match,
    "document classification en": text_match,
    "key information extraction en": json_match,
    "key information extraction cn": json_match,
    "key information mapping en": json_match,
    "table parsing en": overlap_match,
    "table parsing cn": overlap_match,
    "formula recognition en": latex_match,
    "formula recognition cn": latex_match,
    "chart parsing en": json_match,
    "APP agent en": text_match,
    "diagram QA en": text_match,
    "text translation cn": text_match,
    "ASCII art classification en": text_match,
}


# ─── Model ────────────────────────────────────────────────────────────────────

def load_model(model_path, precision, device, attn_impl="sdpa"):
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
    print(f"[eval] Loading model ({precision}, attn={attn_impl}) on {device}", flush=True)
    proc = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    kw = dict(torch_dtype=dtype, device_map=device, local_files_only=True)
    if attn_impl in ("fa2", "guide"):
        kw["attn_implementation"] = "flash_attention_2"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, **kw)
    model.eval()
    if attn_impl == "guide":
        sys.path.insert(0, str(Path(__file__).parent))
        from guide_model_patch import patch_model_with_guide
        patch_model_with_guide(model)
        print("[eval] GUIDE patch applied.", flush=True)
    print(f"[eval] {sum(p.numel() for p in model.parameters())/1e9:.2f}B params", flush=True)
    return model, proc


def infer(model, proc, image_path, question, max_tokens, device):
    from qwen_vl_utils import process_vision_info
    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text",  "text": question},
    ]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    imgs, vids = process_vision_info(msgs)
    inp = proc(text=[text], images=imgs, videos=vids, padding=True, return_tensors="pt")
    inp = {k: v.to(device) for k, v in inp.items()}
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_tokens, do_sample=False)
    return proc.batch_decode(out[:, inp["input_ids"].shape[1]:],
                              skip_special_tokens=True)[0].strip()


# ─── Dataset loader ───────────────────────────────────────────────────────────

def load_ocrbench(data_root, num_samples, shard_idx, num_shards):
    import pandas as pd
    data_dir = Path(data_root) / "ocrbench_v2" / "data"
    img_dir  = Path(data_root) / "ocrbench_v2" / "_extracted_images"
    img_dir.mkdir(exist_ok=True)

    samples = []
    for pf in sorted(data_dir.glob("test-*.parquet")):
        df = pd.read_parquet(pf)
        for i, row in df.iterrows():
            img_data = row.get("image")
            if img_data is None: continue
            img_bytes = (img_data["bytes"] if isinstance(img_data, dict) and "bytes" in img_data
                         else img_data if isinstance(img_data, bytes) else None)
            if img_bytes is None: continue
            img_file = img_dir / f"{pf.stem}_{i}.png"
            if not img_file.exists():
                img_file.write_bytes(img_bytes)

            answers = row.get("answers", row.get("answer", []))
            if hasattr(answers, 'tolist'): answers = answers.tolist()
            if isinstance(answers, str): answers = [answers]
            elif not isinstance(answers, list): answers = [str(answers)]
            answers = [str(v) for v in answers]

            samples.append({
                "sample_id":    f"ocr_{row.get('id', f'{pf.stem}_{i}')}",
                "image_path":   str(img_file),
                "question":     str(row.get("question", "")),
                "answers":      answers,
                "type":         str(row.get("type", "unknown")),
                "dataset_name": str(row.get("dataset_name", "")),
            })
        if num_samples > 0 and len(samples) >= num_samples:
            break

    if num_shards > 1:
        samples = [s for i, s in enumerate(samples) if i % num_shards == shard_idx]

    return samples


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path",  required=True)
    parser.add_argument("--data-root",   required=True)
    parser.add_argument("--output-dir",  required=True)
    parser.add_argument("--num-samples", type=int, default=0)
    parser.add_argument("--precision",   default="bf16")
    parser.add_argument("--device",      default="cuda")
    parser.add_argument("--shard-idx",   type=int, default=0)
    parser.add_argument("--num-shards",  type=int, default=1)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--attn-impl",   default="sdpa",
                        choices=["sdpa", "fa2", "guide"])
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shard_sfx = f"_shard{args.shard_idx}of{args.num_shards}" if args.num_shards > 1 else ""

    samples = load_ocrbench(args.data_root, args.num_samples,
                             args.shard_idx, args.num_shards)
    n = len(samples)
    print(f"[eval] {n} samples (shard {args.shard_idx}/{args.num_shards})", flush=True)

    model, proc = load_model(args.model_path, args.precision, args.device,
                              attn_impl=getattr(args, "attn_impl", "sdpa"))

    results = []
    type_stats = defaultdict(lambda: {"c": 0, "t": 0})
    t0 = time.perf_counter()

    for i, s in enumerate(samples):
        task = s["type"]
        cfg = TASK_CONFIG.get(task, DEFAULT_CONFIG)
        question = cfg["prompt"] + "\n" + s["question"] if s["question"] else cfg["prompt"]
        max_tok = cfg["max_tokens"]

        try:
            pred = infer(model, proc, s["image_path"], question, max_tok, args.device)
        except Exception as e:
            pred = ""
            print(f"  [WARN] {i}: {e}", flush=True)

        metric_fn = TASK_METRIC.get(task, text_match)
        hit = metric_fn(pred, s["answers"])
        type_stats[task]["t"] += 1
        if hit: type_stats[task]["c"] += 1

        results.append({
            "sample_id": s["sample_id"], "type": task,
            "dataset_name": s["dataset_name"],
            "question": s["question"], "pred": pred,
            "gt": s["answers"], "correct": hit,
            "max_tokens_used": max_tok,
        })

        if (i + 1) % 100 == 0 or i < 3:
            elapsed = time.perf_counter() - t0
            total_c = sum(v["c"] for v in type_stats.values())
            acc = total_c / (i + 1)
            eta = elapsed / (i + 1) * (n - i - 1)
            print(f"  [{i+1}/{n}] acc={acc:.3f} | {task[:30]!r} | "
                  f"elapsed={elapsed/60:.1f}m ETA={eta/60:.1f}m", flush=True)

    # Save
    pred_path = out_dir / f"ocrbench_v2{shard_sfx}_predictions.jsonl"
    with open(pred_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Summary
    total_c = sum(v["c"] for v in type_stats.values())
    total_t = sum(v["t"] for v in type_stats.values())
    em_acc = total_c / total_t if total_t else 0

    by_type = {t: round(v["c"]/v["t"], 4) for t, v in type_stats.items() if v["t"] > 0}
    summary = {
        "num_samples": total_t, "exact_match": round(em_acc, 4),
        "shard": shard_sfx, "by_type": by_type,
    }
    (out_dir / f"ocrbench_v2{shard_sfx}_summary.json").write_text(
        json.dumps(summary, indent=2))

    print(f"\n[eval] EM={em_acc:.4f}  n={total_t}", flush=True)
    for t, v in sorted(type_stats.items(), key=lambda x: -x[1]["t"]):
        acc = v["c"]/v["t"] if v["t"] else 0
        print(f"  {t:<42} {v['c']:>4}/{v['t']:<4} = {acc*100:5.1f}%")


if __name__ == "__main__":
    main()
