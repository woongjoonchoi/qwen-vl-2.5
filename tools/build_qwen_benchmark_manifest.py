#!/usr/bin/env python3
"""
Build benchmark manifest from data/raw for Qwen2.5-VL baseline profiling.
Scans all supported benchmarks and outputs benchmark_manifest.jsonl.
"""
import argparse
import json
import os
import sys
from pathlib import Path


def log(msg):
    print(f"[manifest] {msg}", flush=True)


def find_image(image_name, search_dirs):
    """Find image file by name in a list of directories."""
    fname = os.path.basename(image_name)
    for d in search_dirs:
        p = Path(d) / fname
        if p.exists():
            return str(p)
    return None


def scan_docvqa(data_root, max_samples):
    """Scan DocVQA dataset."""
    base = Path(data_root) / "docvqa"
    ann_dir = base / "annotations"
    img_dir = base / "images"

    samples = []
    missing_images = 0

    for ann_file in ["val_v1.0_withQT.json", "train_v1.0_withQT.json", "test_v1.0.json"]:
        path = ann_dir / ann_file
        if not path.exists():
            continue
        split = ann_file.split("_")[0]
        log(f"  docvqa: reading {ann_file}")
        with open(path) as f:
            data = json.load(f)
        items = data.get("data", [])
        for item in items:
            if len(samples) >= max_samples:
                break
            img_rel = item.get("image", "")
            img_fname = os.path.basename(img_rel)
            img_path = find_image(img_fname, [str(img_dir), str(img_dir / split)])
            if img_path is None:
                missing_images += 1
                continue
            answers = item.get("answers", [])
            samples.append({
                "benchmark": "docvqa",
                "sample_id": f"docvqa_{item.get('questionId', len(samples))}",
                "image_path": img_path,
                "question_or_prompt": item.get("question", "Describe the document."),
                "answer_or_label": answers[0] if answers else None,
                "annotation_path": str(path),
                "source_file": str(path),
                "split": split,
            })
        if len(samples) >= max_samples:
            break

    if missing_images:
        log(f"  docvqa: {missing_images} samples missing images (skipped)")
    return samples


def scan_chartqa(data_root, max_samples):
    """Scan ChartQA dataset (parquet with embedded images)."""
    base = Path(data_root) / "chartqa" / "data"
    samples = []
    img_save_dir = Path(data_root) / "chartqa" / "_extracted_images"
    img_save_dir.mkdir(exist_ok=True)

    parquet_files = sorted(base.glob("val-*.parquet")) or sorted(base.glob("test-*.parquet"))
    if not parquet_files:
        log("  chartqa: no parquet files found")
        return samples

    try:
        import pandas as pd
    except ImportError:
        log("  chartqa: pandas not available, skipping")
        return samples

    for pf in parquet_files:
        if len(samples) >= max_samples:
            break
        log(f"  chartqa: reading {pf.name}")
        try:
            df = pd.read_parquet(pf)
        except Exception as e:
            log(f"  chartqa: failed to read {pf}: {e}")
            continue

        # Detect columns
        question_col = next((c for c in ["query", "question", "text", "Question"] if c in df.columns), None)
        answer_col = next((c for c in ["label", "answer", "Answer", "response"] if c in df.columns), None)
        image_col = next((c for c in ["image", "Image", "img"] if c in df.columns), None)

        if question_col is None:
            log(f"  chartqa: no question column found in {list(df.columns)}")
            continue

        for i, row in df.iterrows():
            if len(samples) >= max_samples:
                break
            question = str(row.get(question_col, "")) if question_col else "Describe the chart."
            answer = str(row.get(answer_col, "")) if answer_col else None

            img_path = None
            if image_col and image_col in row:
                img_data = row[image_col]
                if isinstance(img_data, dict) and "bytes" in img_data:
                    # Embedded image bytes
                    img_bytes = img_data["bytes"]
                    img_file = img_save_dir / f"chartqa_{i}.png"
                    if not img_file.exists():
                        img_file.write_bytes(img_bytes)
                    img_path = str(img_file)
                elif isinstance(img_data, bytes):
                    img_file = img_save_dir / f"chartqa_{i}.png"
                    if not img_file.exists():
                        img_file.write_bytes(img_data)
                    img_path = str(img_file)
                elif isinstance(img_data, str) and Path(img_data).exists():
                    img_path = img_data

            if img_path is None:
                continue

            samples.append({
                "benchmark": "chartqa",
                "sample_id": f"chartqa_{pf.stem}_{i}",
                "image_path": img_path,
                "question_or_prompt": question,
                "answer_or_label": answer,
                "annotation_path": str(pf),
                "source_file": str(pf),
                "split": "val" if "val" in pf.stem else "test",
            })

    return samples


def scan_ocrbench_v2(data_root, max_samples):
    """Scan OCRBench v2 dataset."""
    base = Path(data_root) / "ocrbench_v2" / "data"
    samples = []
    img_save_dir = Path(data_root) / "ocrbench_v2" / "_extracted_images"
    img_save_dir.mkdir(exist_ok=True)

    parquet_files = sorted(base.glob("test-*.parquet")) or sorted(base.glob("*.parquet"))
    if not parquet_files:
        log("  ocrbench_v2: no parquet files found")
        return samples

    try:
        import pandas as pd
    except ImportError:
        log("  ocrbench_v2: pandas not available, skipping")
        return samples

    for pf in parquet_files[:2]:  # limit files for speed
        if len(samples) >= max_samples:
            break
        try:
            df = pd.read_parquet(pf)
        except Exception as e:
            log(f"  ocrbench_v2: failed to read {pf}: {e}")
            continue

        question_col = next((c for c in ["question", "query", "text", "Question"] if c in df.columns), None)
        answer_col = next((c for c in ["answer", "Answer", "label", "response"] if c in df.columns), None)
        image_col = next((c for c in ["image", "Image", "img"] if c in df.columns), None)

        for i, row in df.iterrows():
            if len(samples) >= max_samples:
                break
            question = str(row.get(question_col, "Extract the text from this image.")) if question_col else "Extract the text from this image."
            answer = str(row.get(answer_col, "")) if answer_col else None

            img_path = None
            if image_col and image_col in row:
                img_data = row[image_col]
                if isinstance(img_data, dict) and "bytes" in img_data:
                    img_bytes = img_data["bytes"]
                    img_file = img_save_dir / f"ocr_{i}.png"
                    if not img_file.exists():
                        img_file.write_bytes(img_bytes)
                    img_path = str(img_file)
                elif isinstance(img_data, bytes):
                    img_file = img_save_dir / f"ocr_{i}.png"
                    if not img_file.exists():
                        img_file.write_bytes(img_data)
                    img_path = str(img_file)

            if img_path is None:
                continue

            samples.append({
                "benchmark": "ocr",
                "sample_id": f"ocr_{pf.stem}_{i}",
                "image_path": img_path,
                "question_or_prompt": question,
                "answer_or_label": answer,
                "annotation_path": str(pf),
                "source_file": str(pf),
                "split": "test",
            })

    return samples


def scan_refcocog(data_root, max_samples):
    """Scan RefCOCOg for grounding benchmark."""
    base = Path(data_root) / "refcocog"
    refer_dir = base / "refer"
    coco_dir = base / "coco"
    samples = []

    ann_file = refer_dir / "refcocog" / "refs(umd).p"
    if not ann_file.exists():
        ann_file = refer_dir / "refcocog" / "refs(google).p"
    if not ann_file.exists():
        # Try to find any annotation
        refs_files = list(refer_dir.rglob("refs*.p"))
        if refs_files:
            ann_file = refs_files[0]
        else:
            log("  refcocog: no annotation file found")
            return samples

    try:
        import pickle
        with open(ann_file, "rb") as f:
            refs = pickle.load(f)
    except Exception as e:
        log(f"  refcocog: failed to load refs: {e}")
        return samples

    # Find COCO images
    img_dirs = list(coco_dir.glob("*2014")) + list(coco_dir.glob("images"))
    if not img_dirs:
        img_dirs = [coco_dir]

    for ref in refs[:max_samples * 2]:
        if len(samples) >= max_samples:
            break
        if ref.get("split", "") not in ["val", "test", "testA", "testB"]:
            continue
        try:
            sentences = ref.get("sentences", [])
            if not sentences:
                continue
            question = f"Point out the location of: {sentences[0]['sent']}"
            img_id = ref.get("image_id", 0)
            img_fname = f"COCO_val2014_{str(img_id).zfill(12)}.jpg"
            img_path = None
            for d in img_dirs:
                p = d / img_fname
                if p.exists():
                    img_path = str(p)
                    break
                p = d / "val2014" / img_fname
                if p.exists():
                    img_path = str(p)
                    break

            if img_path is None:
                continue

            bbox = ref.get("bbox", None)
            samples.append({
                "benchmark": "grounding",
                "sample_id": f"refcocog_{ref.get('ref_id', len(samples))}",
                "image_path": img_path,
                "question_or_prompt": question,
                "answer_or_label": json.dumps(bbox) if bbox else None,
                "annotation_path": str(ann_file),
                "source_file": str(ann_file),
                "split": ref.get("split", "val"),
            })
        except Exception:
            continue

    return samples


def scan_visionarena(data_root, max_samples):
    """Scan VisionArena benchmark."""
    base = Path(data_root) / "visionarena_bench" / "data"
    samples = []
    img_save_dir = Path(data_root) / "visionarena_bench" / "_extracted_images"
    img_save_dir.mkdir(exist_ok=True)

    parquet_files = sorted(base.glob("train-*.parquet")) or sorted(base.glob("*.parquet"))
    if not parquet_files:
        log("  visionarena: no parquet files found")
        return samples

    try:
        import pandas as pd
    except ImportError:
        log("  visionarena: pandas not available")
        return samples

    for pf in parquet_files[:1]:
        if len(samples) >= max_samples:
            break
        try:
            df = pd.read_parquet(pf)
        except Exception as e:
            log(f"  visionarena: failed to read {pf}: {e}")
            continue

        # VisionArena conversation format
        for i, row in df.iterrows():
            if len(samples) >= max_samples:
                break
            try:
                conv = row.get("conversation", row.get("conversations", None))
                question = "Describe this image in detail."
                if conv:
                    if isinstance(conv, list):
                        for turn in conv:
                            if isinstance(turn, dict) and turn.get("role", "") in ["user", "human"]:
                                question = turn.get("content", question)
                                if isinstance(question, list):
                                    for part in question:
                                        if isinstance(part, dict) and part.get("type") == "text":
                                            question = part.get("text", question)
                                            break
                                break
                    elif isinstance(conv, str):
                        question = conv[:200]

                img_data = row.get("image", row.get("images", None))
                img_path = None

                if img_data is not None:
                    if isinstance(img_data, dict) and "bytes" in img_data:
                        img_bytes = img_data["bytes"]
                    elif isinstance(img_data, bytes):
                        img_bytes = img_data
                    elif isinstance(img_data, list) and len(img_data) > 0:
                        first = img_data[0]
                        if isinstance(first, dict):
                            img_bytes = first.get("bytes", None)
                        elif isinstance(first, bytes):
                            img_bytes = first
                        else:
                            img_bytes = None
                    else:
                        img_bytes = None

                    if img_bytes:
                        img_file = img_save_dir / f"va_{i}.png"
                        if not img_file.exists():
                            img_file.write_bytes(img_bytes)
                        img_path = str(img_file)

                if img_path is None:
                    continue

                samples.append({
                    "benchmark": "visual_chat",
                    "sample_id": f"visionarena_{i}",
                    "image_path": img_path,
                    "question_or_prompt": str(question)[:512],
                    "answer_or_label": None,
                    "annotation_path": str(pf),
                    "source_file": str(pf),
                    "split": "train",
                })
            except Exception:
                continue

    return samples


def main():
    parser = argparse.ArgumentParser(description="Build Qwen benchmark manifest from data/raw")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples-per-benchmark", type=int, default=5)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_root = args.data_root
    max_s = args.max_samples_per_benchmark

    log(f"Scanning data root: {data_root}")
    log(f"Max samples per benchmark: {max_s}")

    all_samples = []
    missing = []
    summary_rows = []

    scanners = [
        ("docvqa", scan_docvqa),
        ("chartqa", scan_chartqa),
        ("ocrbench_v2", scan_ocrbench_v2),
        ("refcocog", scan_refcocog),
        ("visionarena", scan_visionarena),
    ]

    for name, fn in scanners:
        log(f"Scanning {name}...")
        try:
            samples = fn(data_root, max_s)
            if samples:
                all_samples.extend(samples)
                benchmarks = set(s["benchmark"] for s in samples)
                log(f"  {name}: {len(samples)} samples (benchmarks: {benchmarks})")
                for bm in benchmarks:
                    bm_samples = [s for s in samples if s["benchmark"] == bm]
                    summary_rows.append({"benchmark": bm, "count": len(bm_samples), "status": "OK"})
            else:
                log(f"  {name}: 0 samples found")
                missing.append(name)
                summary_rows.append({"benchmark": name, "count": 0, "status": "MISSING_DATA"})
        except Exception as e:
            log(f"  {name}: ERROR - {e}")
            missing.append(name)
            summary_rows.append({"benchmark": name, "count": 0, "status": f"ERROR: {e}"})

    if not all_samples:
        log("ERROR: No samples found in any benchmark!")
        status = {"status": "FAIL", "message": "No samples found"}
    else:
        log(f"Total samples: {len(all_samples)}")
        status = {"status": "PASS", "total_samples": len(all_samples)}

    # Write manifest
    manifest_path = out_dir / "benchmark_manifest.jsonl"
    with open(manifest_path, "w") as f:
        for s in all_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    log(f"Written: {manifest_path}")

    # Write summary CSV
    import csv
    summary_path = out_dir / "benchmark_manifest_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["benchmark", "count", "status"])
        writer.writeheader()
        writer.writerows(summary_rows)
    log(f"Written: {summary_path}")

    # Write sample preview
    preview = all_samples[:10]
    preview_path = out_dir / "sample_preview.json"
    with open(preview_path, "w") as f:
        json.dump(preview, f, indent=2, ensure_ascii=False)

    # Write missing benchmarks
    missing_path = out_dir / "missing_benchmarks.json"
    with open(missing_path, "w") as f:
        json.dump(missing, f, indent=2)

    # Write stage status
    stage_status = {
        "status": status["status"],
        "total_samples": len(all_samples),
        "benchmarks_found": list(set(s["benchmark"] for s in all_samples)),
        "missing_benchmarks": missing,
    }
    with open(out_dir / "stage_status.json", "w") as f:
        json.dump(stage_status, f, indent=2)

    # Write report
    with open(out_dir / "data_manifest_report.md", "w") as f:
        f.write("# Data Manifest Report\n\n")
        f.write(f"- Data root: `{data_root}`\n")
        f.write(f"- Total samples: {len(all_samples)}\n\n")
        f.write("## Benchmark Summary\n\n")
        for row in summary_rows:
            f.write(f"- **{row['benchmark']}**: {row['count']} samples — {row['status']}\n")
        if missing:
            f.write(f"\n## Missing Benchmarks\n\n{missing}\n")

    if status["status"] != "PASS":
        log("FAIL: No samples found")
        sys.exit(1)

    log("DONE: manifest build complete")


if __name__ == "__main__":
    main()
