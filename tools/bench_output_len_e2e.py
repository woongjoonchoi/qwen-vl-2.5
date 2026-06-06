#!/usr/bin/env python3
"""
E2E latency vs output_length at 1792px resolution.
Forces generation of exactly N tokens via min_new_tokens=max_new_tokens=N.
Measures: ve_ms, e2e_ms, tpot_ms for FA2 or GUIDE.
"""
import argparse, json, sys, csv
from pathlib import Path
import torch

sys.path.insert(0, '/workspace')
sys.path.insert(0, '/workspace/tools')

MODEL_PATH     = '/workspace/models/Qwen2.5-VL-3B-Instruct'
MANIFEST       = '/workspace/output/qwen25vl_baseline/manifests/benchmark_manifest_500_docker.jsonl'
MAX_PIXELS     = 1792 * 1792          # forces ~1792px images
OUTPUT_LENGTHS = [32, 64, 128, 256, 512, 1024]
N_SAMPLES      = 20
WARMUP_ITERS   = 3
PROMPT         = "Describe this image comprehensively."


def load_model(mode: str):
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    proc = AutoProcessor.from_pretrained(
        MODEL_PATH, local_files_only=True,
        min_pixels=4 * 28 * 28, max_pixels=MAX_PIXELS)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map='cuda',
        local_files_only=True, attn_implementation='flash_attention_2')
    if mode == 'guide':
        from tools.guide_model_patch import patch_model_with_guide
        patch_model_with_guide(model)
    model.eval()
    return model, proc


def prepare_inputs(proc, image_path: str):
    from PIL import Image
    from qwen_vl_utils import process_vision_info
    img = Image.open(image_path).convert('RGB')
    msgs = [{'role': 'user', 'content': [
        {'type': 'image', 'image': img},
        {'type': 'text',  'text': PROMPT}]}]
    text   = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    img_in, _ = process_vision_info(msgs)
    inputs = proc(text=[text], images=img_in, padding=True, return_tensors='pt')
    inputs = {k: v.to('cuda') for k, v in inputs.items()}
    inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)
    return inputs


def cuda_ms(ev0, ev1):
    return ev0.elapsed_time(ev1)


def measure(model, inputs, n_tokens):
    """Returns (ve_ms, e2e_ms, n_gen)."""
    pv   = inputs['pixel_values']
    gthw = inputs['image_grid_thw']

    ev = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
    torch.cuda.synchronize()

    with torch.no_grad():
        ev[0].record()
        _ = model.visual(pv, grid_thw=gthw)
        ev[1].record()

        ev[2].record()
        out = model.generate(
            **inputs,
            max_new_tokens=n_tokens,
            min_new_tokens=n_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            use_cache=True,
        )
        ev[3].record()
        torch.cuda.synchronize()

    ve_ms  = cuda_ms(ev[0], ev[1])
    e2e_ms = cuda_ms(ev[2], ev[3])
    n_gen  = int(out.shape[1] - inputs['input_ids'].shape[1])
    return ve_ms, e2e_ms, n_gen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode',   required=True, choices=['fa2', 'guide'])
    ap.add_argument('--output', required=True)
    args = ap.parse_args()

    # load samples
    samples = []
    with open(MANIFEST) as f:
        for line in f:
            d = json.loads(line.strip())
            if Path(d['image_path']).exists():
                samples.append(d)
            if len(samples) == N_SAMPLES:
                break
    print(f'Loaded {len(samples)} samples', flush=True)

    # load model
    print(f'Loading [{args.mode}] model...', flush=True)
    model, proc = load_model(args.mode)

    # pre-process all inputs
    print('Processing images at 1792px...', flush=True)
    all_inputs = []
    for s in samples:
        inp  = prepare_inputs(proc, s['image_path'])
        T, H, W = inp['image_grid_thw'][0].tolist()
        all_inputs.append({
            'sample_id': s['sample_id'],
            'inputs':    inp,
            'n_vis':     int(H * W * T),
            'H': int(H), 'W': int(W),
        })
        print(f'  {s["sample_id"]}: grid {int(H)}×{int(W)} → {int(H*W*T)} tokens', flush=True)

    # warmup
    print(f'\nWarmup {WARMUP_ITERS} iters...', flush=True)
    with torch.no_grad():
        for _ in range(WARMUP_ITERS):
            measure(model, all_inputs[0]['inputs'], 64)
    torch.cuda.synchronize()

    # measure
    results = []
    for n_tok in OUTPUT_LENGTHS:
        print(f'\n=== output_length = {n_tok} ===', flush=True)
        for item in all_inputs:
            ve_ms, e2e_ms, n_gen = measure(model, item['inputs'], n_tok)
            llm_ms  = e2e_ms - ve_ms
            tpot_ms = llm_ms / max(n_gen, 1)
            row = {
                'mode':          args.mode,
                'sample_id':     item['sample_id'],
                'output_length': n_tok,
                'n_vis_tokens':  item['n_vis'],
                'grid_h':        item['H'],
                'grid_w':        item['W'],
                've_ms':         round(ve_ms,  3),
                'e2e_ms':        round(e2e_ms, 3),
                'llm_ms':        round(llm_ms, 3),   # prefill + decode
                'n_gen':         n_gen,
                'tpot_ms':       round(tpot_ms, 3),
            }
            results.append(row)
            print(f'  {item["sample_id"]}: ve={ve_ms:.1f}ms  e2e={e2e_ms:.1f}ms  n_gen={n_gen}  tpot={tpot_ms:.1f}ms', flush=True)

    # save
    fields = ['mode','sample_id','output_length','n_vis_tokens','grid_h','grid_w',
              've_ms','e2e_ms','llm_ms','n_gen','tpot_ms']
    with open(args.output, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)
    print(f'\nSaved {len(results)} rows → {args.output}', flush=True)


if __name__ == '__main__':
    main()
