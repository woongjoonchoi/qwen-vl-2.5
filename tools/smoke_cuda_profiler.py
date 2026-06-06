#!/usr/bin/env python3
"""CUDA NVTX smoke test for Docker/host profiler validation."""
import torch
import time

assert torch.cuda.is_available(), "CUDA not available"

torch.cuda.nvtx.range_push("smoke_total")
torch.cuda.nvtx.range_push("smoke_matmul")

a = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
b = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)

for _ in range(10):
    c = a @ b

torch.cuda.synchronize()

torch.cuda.nvtx.range_pop()
torch.cuda.nvtx.range_pop()

print("SMOKE_CUDA_PROFILER_OK")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"CUDA: {torch.version.cuda}")
