# Session Context — GUIDE Kernel Optimization (Qwen2.5-VL-3B)

> 다른 디바이스/세션에서 작업을 이어갈 때 이 문서를 먼저 읽으세요.
> 코드/CSV는 git에 있고, 이 문서는 "왜 그렇게 했는지"를 담습니다.

## 환경 / 제약
- **Docker-first**: host 환경 건드리지 않음. 항상 컨테이너에서 실행.
  - 이미지: `qwen25vl-guide:latest` (flash-attn 포함), `qwen25vl-baseline:latest`
  - 실행 패턴:
    ```
    docker run --rm --gpus device=0 -v /home/wjchoi/Qwen3-VL:/workspace \
      -e HOME=/tmp -e TRITON_CACHE_DIR=/tmp/.triton -e XDG_CACHE_HOME=/tmp/.cache \
      --entrypoint python3 qwen25vl-guide:latest /workspace/tools/<script>.py
    ```
- GPU: NVIDIA H200 (139.8 GB), Triton 3.7.0
- 모델: Qwen2.5-VL-3B-Instruct (`models/`, gitignore됨 → download 스크립트)
- 데이터: DocVQA 등 (`data/`, 36GB, gitignore됨)

## 아키텍처 핵심 수치 (ViT)
- 32 blocks = 28 window-attn + 4 full-attn
- window=64 tokens (4×4 merged cells × SMU=4), head_dim=**80**, NH=16, C=1280
- spatial_merge=2 → SMU=4, RWS=8(=112/14), WMS=4, WIN_TOK=64
- **head_dim=80 → PAD_DIM=128 → 37.5% BW 낭비** ← 핵심 최적화 포인트

## GUIDE vs FA2 핵심
- GUIDE: descriptor 기반 window attention, **scatter/reverse 불필요** (partition 제거)
- FA2: get_window_index + scatter + reverse 오버헤드 (1.52ms@448 ~ 7.95ms@1792)
- 커널 자체는 메모리 바운드 (산술강도 32 FLOP/byte << ridge 296)
- BW 활용: FA2=78.6%, GUIDE-pad128=52.8% (NSys 측정)

## 이번 세션에서 한 실험들 (시간순)

### 1. BLOCK_M/N 분리 (`tools/bench_block_mn_sweep.py`)
- Q tile(BLOCK_M)과 KV tile(BLOCK_N) 독립 튜닝
- n_kv_blocks = cdiv(WIN_TOK, BLOCK_N), BN=32 → n_kv=2 (sw-pipeline 가능)
- 결과: kernel 단독 최대 16% 개선(1344/1792px), 하지만 E2E는 0.1% (Amdahl)
- 최적: BM=64, BN=32, warps=2 (high-res), BM=16/BN=32 (448px)
- 로그: `results/bench_block_mn_sweep.log`

### 2. range variant 비교 (`tools/bench_range_variants.py`)
- `range` vs `tl.static_range` vs `tl.range` (KV 루프)
- 결론: **`tl.range` ≈ `range`** (Triton 3.7이 내부적으로 동일 처리)
- `tl.static_range`: N_KV=1일 때만 이득(1.29×), N_KV≥2에선 레지스터 압박으로 역효과
- **결정: 현재 `range()` 유지**
- 로그: `results/bench_range_variants.log`

### 3. ★ Head-split 64+16 (`tools/bench_head_split.py`) — 가장 중요
- head_dim 80을 [0:64]+[64:80] 두 청크로 분리 → PAD 낭비 제거
- QK: dot(q_main,k_main) + dot(q_tail,k_tail) 누적
- V: acc_main += p@v_main, acc_tail += p@v_tail
- **결과: kernel BW 이론 1.6× 중 80~88% 실현**
  | res | split/FA2 | split/pad128 |
  |-----|-----------|--------------|
  | 448 | 0.964× (FA2 추월!) | 1.118× |
  | 896 | 0.864× (추월!) | 1.411× |
  | 1344| 0.945× (추월!) | 1.354× |
  | 1792| 1.035× | 1.277× |
- **448/896/1344px에서 GUIDE-split이 FA2를 처음으로 추월**
- 최적: BM=64, BN=32, warps=2
- 로그: `results/bench_head_split.log`
- **다음 단계(미완료): guide_model_patch.py에 head-split 통합 → 실제 E2E 측정**

### 4. Analytic 전체 파이프라인 영향 (`tools/analytic_head_split_impact.py`)
- head-split이 VE/TTFT/E2E/tok-s에 미치는 영향 계산
- 결과: **kernel은 VE의 1.5~1.9%, TTFT의 0.8~1.1%**
  → head-split 적용해도 TTFT 개선 0.1~0.36%, output tok/s 불변
- output tok/s는 LLM decode가 병목 → vision kernel 최적화 무관
- 출력: `results/analytic_head_split_impact.txt`

### Window block 전용 speedup (vs SDPA) — analytic
| | 448 | 896 | 1344 | 1792 |
|--|-----|-----|------|------|
| FA2 | 5.05× | 3.35× | 3.12× | 3.03× |
| GUIDE-split | 6.17× | 3.83× | 3.47× | 3.26× |
- 고해상도로 갈수록 speedup 수렴: GUIDE는 non-kernel(QKV/MLP 97.7%)이 bottleneck

### 5. 메모리 측정 (`tools/measure_peak_memory.py`)
- 1792px 실측 peak: **~8.0 GB** (H200의 5.6%)
  - 모델 가중치 7.16GB, VIT 활성화 756MB, KV cache ~570MB
- prefill→decode 전환 시 오히려 감소 (활성화 202MB 해제, KV 144KB/token)
- 30GB 도달하려면: batch ~27장(1792px) 또는 학습 시나리오
- 로그: `results/peak_memory_1792.log`

## 핵심 교훈 (Amdahl)
```
window attn kernel = VE의 ~2%, TTFT의 ~1%
→ 커널 아무리 최적화해도 E2E 영향 미미
진짜 레버리지: QKV/MLP GEMM (VE의 75%), partition 제거(GUIDE가 이미 함)
head-split은 "커널 단독으로는 FA2 추월"이라는 학술적 의의가 큼 (E2E 실익은 작음)
```

## 미완료 / 다음 작업 후보
- [ ] head-split을 `guide_model_patch.py`에 통합 → 실제 E2E/accuracy 측정
- [ ] head-split + BLOCK_M/N(64,32,w2) 조합 최적 config를 패치 기본값으로
- [ ] full-attn ×4 블록(1792px에서 VE의 22%)은 compute-bound → head-split 무효, 별도 접근 필요
