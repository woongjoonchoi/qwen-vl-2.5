#!/usr/bin/env python3
"""
Vision Encoder E2E Latency Comparison (896×896)

세 가지 경로를 vision encoder 단위로 비교:
  A. Baseline-SDPA   : HF 기본 경로, _attn_implementation="sdpa" (PyTorch Flash Attn)
  B. Baseline-Eager  : _attn_implementation="eager" (naive matmul)
  C. GUIDE           : window_index 재배열 없음, Triton descriptor 커널

측정 범위: patch_embed → 모든 blocks → merger
           (reverse_restore 포함, LLM decoder 제외)

방법: torch.cuda.Event GPU 타이밍, warmup=5, measure=20
"""
import sys, json, time
from pathlib import Path
import torch
import torch.nn.functional as F

sys.path.insert(0, '/workspace')
sys.path.insert(0, '/workspace/tools')

RESIZE_PX  = 896
BLOCK_SIZE = 64     # Stage-11 winner
WARMUP     = 5
MEASURE    = 20
PRECISION  = "bf16"


# ── 모델 로드 ──────────────────────────────────────────────────────────────────
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    apply_rotary_pos_emb_vision,
)
from qwen_vl_utils import process_vision_info
from PIL import Image
import numpy as np

print("[bench] loading model...", flush=True)
dtype  = torch.bfloat16
proc   = AutoProcessor.from_pretrained(
    '/workspace/models/Qwen2.5-VL-3B-Instruct', local_files_only=True)
model  = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    '/workspace/models/Qwen2.5-VL-3B-Instruct',
    torch_dtype=dtype, device_map='cuda', local_files_only=True)
model.eval()

vc         = model.config.vision_config
num_heads  = vc.num_heads
head_dim   = vc.hidden_size // num_heads
rws        = vc.window_size // vc.patch_size
sms        = vc.spatial_merge_size
smu        = sms ** 2
fullatt    = set(vc.fullatt_block_indexes)

from tools.qwen_descriptor import QwenDescriptor, get_per_batch_blocks
from kernel_triton.qwen_window_attention.triton_ops_qwen import qwen_window_attention

# ── 입력 준비 ──────────────────────────────────────────────────────────────────
img = Image.fromarray(
    np.random.randint(0, 255, (RESIZE_PX, RESIZE_PX, 3), dtype=np.uint8))
msgs = [{"role":"user","content":[
    {"type":"image","image":img},{"type":"text","text":"Describe."}]}]
text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
img_in, vid_in = process_vision_info(msgs)
inputs = proc(text=[text], images=img_in, videos=vid_in,
              padding=True, return_tensors="pt")
pixel_values   = inputs["pixel_values"].to("cuda").to(dtype)
image_grid_thw = inputs["image_grid_thw"].to("cuda")
T, H, W = image_grid_thw[0].tolist()
S = T * H * W
print(f"[bench] grid_thw={T}×{H}×{W}  S_raw={S}  dtype={dtype}", flush=True)
print(f"[bench] 28 window layers, 4 full layers  BLOCK_SIZE={BLOCK_SIZE}", flush=True)

# ── GUIDE descriptor (루프 밖 1회) ─────────────────────────────────────────────
gen       = QwenDescriptor(num_heads, rws, sms)
guide_meta = gen.build(H, W, BLOCK_SIZE, device="cuda")
guide_pbk  = gen.per_batch_blocks(H, W, BLOCK_SIZE)
MW         = W // sms
print(f"[bench] descriptor built: meta_len={len(guide_meta)}  pbk={guide_pbk}", flush=True)


# ── GPU 타이밍 유틸 ────────────────────────────────────────────────────────────
def gpu_time_ms(fn, warmup=WARMUP, measure=MEASURE):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    events = [(torch.cuda.Event(enable_timing=True),
               torch.cuda.Event(enable_timing=True)) for _ in range(measure)]
    for s, e in events:
        s.record(); fn(); e.record()
    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in events]
    return times


# ══════════════════════════════════════════════════════════════════════════════
# A. Baseline-SDPA
# ══════════════════════════════════════════════════════════════════════════════
print("\n[A] Baseline-SDPA  (PyTorch F.scaled_dot_product_attention)", flush=True)

def run_baseline_sdpa():
    with torch.no_grad():
        return model.visual(pixel_values, image_grid_thw)

times_sdpa = gpu_time_ms(run_baseline_sdpa)
mean_sdpa  = sum(times_sdpa) / len(times_sdpa)
print(f"  mean={mean_sdpa:.2f} ms  min={min(times_sdpa):.2f}  max={max(times_sdpa):.2f}", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# B. Baseline-Eager  (naive matmul, no flash)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[B] Baseline-Eager  (naive matmul softmax)", flush=True)

# 어텐션 구현을 eager로 임시 교체
for blk in model.visual.blocks:
    blk.attn.config._attn_implementation = "eager"

def run_baseline_eager():
    with torch.no_grad():
        return model.visual(pixel_values, image_grid_thw)

times_eager = gpu_time_ms(run_baseline_eager)
mean_eager  = sum(times_eager) / len(times_eager)
print(f"  mean={mean_eager:.2f} ms  min={min(times_eager):.2f}  max={max(times_eager):.2f}", flush=True)

# sdpa로 복원
for blk in model.visual.blocks:
    blk.attn.config._attn_implementation = "sdpa"


# ══════════════════════════════════════════════════════════════════════════════
# C. GUIDE  (Triton descriptor kernel for window layers)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[C] GUIDE  (Triton descriptor, no window_index materialization)", flush=True)

def run_guide():
    """
    Vision encoder forward — GUIDE 경로.
    1. hidden/RoPE를 row-major 유지 (window_index 재배열 없음)
    2. descriptor를 루프 밖에서 미리 구성 (guide_meta, 이미 빌드됨)
    3. window layer: Triton descriptor 커널 (gather+attend+scatter)
    4. full  layer : PyTorch SDPA (변경 없음)
    5. reverse_restore 없음
    """
    with torch.no_grad():
        vis = model.visual

        # ① patch embed
        hidden = vis.patch_embed(pixel_values)   # [S_raw, C]

        # ② RoPE — row-major 유지 (reorder 안 함)
        rope = vis.rot_pos_emb(image_grid_thw)   # [S_raw, D_rope/2]
        emb  = torch.cat((rope, rope), dim=-1)
        pos_emb = (emb.cos(), emb.sin())          # row-major

        # ③ full attention용 cu_seqlens
        cu_seqlens = F.pad(
            (image_grid_thw[:, 1] * image_grid_thw[:, 2] * image_grid_thw[:, 0])
            .cumsum(0, dtype=torch.int32), (1, 0), value=0)

        # ④ block 루프 — descriptor는 루프 밖에서 이미 빌드됨 (guide_meta)
        for li, blk in enumerate(vis.blocks):
            normed = blk.norm1(hidden)
            seq_len, C = normed.shape

            if li in fullatt:
                # full attention: SDPA 그대로
                attn_out = blk.attn(
                    normed, cu_seqlens=cu_seqlens, position_embeddings=pos_emb)
            else:
                # window attention: GUIDE Triton 커널
                qkv = blk.attn.qkv(normed) \
                          .reshape(seq_len, 3, num_heads, head_dim) \
                          .permute(1, 0, 2, 3)
                q, k, v = qkv.unbind(0)           # [S, nh, hd]

                cos, sin = pos_emb
                q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

                # [1, nh, S, hd]
                q = q.transpose(0, 1).unsqueeze(0).contiguous()
                k = k.transpose(0, 1).unsqueeze(0).contiguous()
                v = v.transpose(0, 1).unsqueeze(0).contiguous()

                out = qwen_window_attention(
                    q, k, v, guide_meta, MW, smu, guide_pbk,
                    BLOCK_SIZE=BLOCK_SIZE, seq_layout=False,
                    num_warps=4, num_stages=3)   # Stage-11 winner

                attn_out = out[0].transpose(0, 1).reshape(seq_len, C)
                attn_out = blk.attn.proj(attn_out)

            hidden = hidden + attn_out
            hidden = hidden + blk.mlp(blk.norm2(hidden))

        # ⑤ merger (row-major 순서로 입력)
        hidden = vis.merger(hidden)
        # reverse_restore 없음 — row-major 그대로 반환

        return hidden

times_guide = gpu_time_ms(run_guide)
mean_guide  = sum(times_guide) / len(times_guide)
print(f"  mean={mean_guide:.2f} ms  min={min(times_guide):.2f}  max={max(times_guide):.2f}", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# 결과 요약
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print(f"  Vision Encoder E2E  ({RESIZE_PX}×{RESIZE_PX}, bf16, H200)")
print(f"  grid {T}×{H}×{W}  S_raw={S}  heads={num_heads}  hd={head_dim}")
print(f"  {MEASURE} iters after {WARMUP} warmup")
print("="*60)
print(f"  A. Baseline-SDPA   {mean_sdpa:7.2f} ms   (1.00×)")
print(f"  B. Baseline-Eager  {mean_eager:7.2f} ms   ({mean_eager/mean_sdpa:.2f}×)")
print(f"  C. GUIDE (Triton)  {mean_guide:7.2f} ms   ({mean_guide/mean_sdpa:.2f}×)")
print("="*60)
print(f"  SDPA vs Eager speedup : {mean_eager/mean_sdpa:.2f}×")
print(f"  GUIDE vs SDPA         : {mean_sdpa/mean_guide:.2f}× ({'faster' if mean_guide < mean_sdpa else 'slower'})")
print(f"  GUIDE vs Eager        : {mean_eager/mean_guide:.2f}× ({'faster' if mean_guide < mean_eager else 'slower'})")

# JSON 저장
result = {
    "resize_px": RESIZE_PX, "grid": [T,H,W], "S_raw": S,
    "num_heads": num_heads, "head_dim": head_dim,
    "block_size": BLOCK_SIZE, "precision": PRECISION,
    "warmup": WARMUP, "measure": MEASURE,
    "baseline_sdpa_mean_ms":  round(mean_sdpa,  4),
    "baseline_eager_mean_ms": round(mean_eager, 4),
    "guide_mean_ms":          round(mean_guide, 4),
    "guide_vs_sdpa_speedup":  round(mean_sdpa / mean_guide, 4),
    "guide_vs_eager_speedup": round(mean_eager / mean_guide, 4),
    "sdpa_vs_eager_speedup":  round(mean_eager / mean_sdpa,  4),
}
out_path = Path('/workspace/output/qwen25vl_guide_kernel_level/summaries/ve_e2e_bench_896.json')
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(result, indent=2))
print(f"\n[bench] saved → {out_path}")
