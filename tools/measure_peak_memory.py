#!/usr/bin/env python3
"""Peak GPU memory measurement at each inference stage (1792px)."""
import torch, sys, json
from pathlib import Path
sys.path.insert(0, '/workspace')

MODEL_PATH = '/workspace/models/Qwen2.5-VL-3B-Instruct'
MANIFEST   = '/workspace/output/qwen25vl_baseline/manifests/benchmark_manifest_500_docker.jsonl'

def mb(b): return b / 1024**2
def snap(label):
    torch.cuda.synchronize()
    cur = mb(torch.cuda.memory_allocated())
    peak= mb(torch.cuda.max_memory_allocated())
    res = mb(torch.cuda.memory_reserved())
    print(f'  [{label:<30}]  cur={cur:7.0f}MB  peak={peak:7.0f}MB  reserved={res:7.0f}MB')
    return cur, peak

print('=' * 75)
print(' Peak GPU Memory Measurement — Qwen2.5-VL-3B @ 1792px')
print('=' * 75)
print(f'  GPU: {torch.cuda.get_device_name(0)}')
print(f'  Total VRAM: {mb(torch.cuda.get_device_properties(0).total_memory):.0f} MB '
      f'({mb(torch.cuda.get_device_properties(0).total_memory)/1024:.1f} GB)')
print()

# ── 1. before model load ──────────────────────────────────────────────────────
torch.cuda.reset_peak_memory_stats()
snap('before model load')

# ── 2. load model ─────────────────────────────────────────────────────────────
print()
print('  Loading model...')
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
proc  = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True,
            min_pixels=4*28*28, max_pixels=1792*1792)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_PATH, torch_dtype=torch.bfloat16, device_map='cuda',
            local_files_only=True, attn_implementation='flash_attention_2')
model.eval()
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
snap('after model load')

# ── 3. prepare 1792px input ───────────────────────────────────────────────────
print()
print('  Preparing 1792px input...')
from PIL import Image
from qwen_vl_utils import process_vision_info

# load first available sample
sample_path = None
with open(MANIFEST) as f:
    for line in f:
        d = json.loads(line.strip())
        if Path(d['image_path']).exists():
            sample_path = d['image_path']
            break
assert sample_path, 'No sample found'
print(f'  Image: {sample_path}')

img  = Image.open(sample_path).convert('RGB')
msgs = [{'role': 'user', 'content': [
    {'type': 'image', 'image': img},
    {'type': 'text',  'text': 'Describe this image comprehensively.'}]}]
text    = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
img_in, _ = process_vision_info(msgs)
inputs  = proc(text=[text], images=img_in, padding=True, return_tensors='pt')
inputs  = {k: v.to('cuda') for k, v in inputs.items()}
inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)

T, H, W = inputs['image_grid_thw'][0].tolist()
n_vis = int(H * W)
print(f'  Grid: {int(H)}×{int(W)} = {n_vis} raw tokens → {n_vis//4} merged tokens')
print(f'  Input IDs: {inputs["input_ids"].shape[1]} tokens')

torch.cuda.reset_peak_memory_stats()
snap('after input prep')

# ── 4. VIT forward only ───────────────────────────────────────────────────────
print()
print('  Running VIT forward...')
torch.cuda.reset_peak_memory_stats()
with torch.no_grad():
    _ = model.visual(inputs['pixel_values'], grid_thw=inputs['image_grid_thw'])
snap('after VIT forward (peak)')

# ── 5. full generate (prefill + decode 64 tokens) ─────────────────────────────
print()
print('  Running generate (prefill + decode 64 tokens)...')
torch.cuda.reset_peak_memory_stats()
with torch.no_grad():
    snap('before generate')
    out = model.generate(
        **inputs,
        max_new_tokens=64, min_new_tokens=64,
        do_sample=False, temperature=None, top_p=None, use_cache=True,
    )
n_gen = int(out.shape[1] - inputs['input_ids'].shape[1])
print(f'  Generated {n_gen} tokens')
_, peak_gen = snap('after generate (peak)')

# ── 6. decode only (longer) ───────────────────────────────────────────────────
print()
print('  Running generate (decode 256 tokens)...')
torch.cuda.reset_peak_memory_stats()
with torch.no_grad():
    out2 = model.generate(
        **inputs,
        max_new_tokens=256, min_new_tokens=256,
        do_sample=False, temperature=None, top_p=None, use_cache=True,
    )
snap('after generate 256tok (peak)')

# ── 7. summary ────────────────────────────────────────────────────────────────
print()
print('=' * 75)
print(' Summary')
print('=' * 75)
total_vram = mb(torch.cuda.get_device_properties(0).total_memory)
rows = [
    ('Model weights (cur after load)', mb(torch.cuda.memory_allocated())),
]
# re-run to get clean numbers
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
snap('after empty_cache')
print(f'  Total VRAM:  {total_vram:.0f} MB ({total_vram/1024:.1f} GB)')
print(f'  Utilization at peak generate: {peak_gen/total_vram*100:.1f}%')
