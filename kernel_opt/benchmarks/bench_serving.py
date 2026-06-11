"""
Single-process before/after throughput benchmark.

Loads the model ONCE, runs warmup then measurement back-to-back
on the same LLM instance — no GPU memory issues between runs.

Usage:
  # Baseline (gemv.py dispatch in utils.py)
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python kernel_opt/benchmarks/bench_serving.py

  # Optimised (shape-aware dispatch in utils.py)
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python kernel_opt/benchmarks/bench_serving.py
"""

from __future__ import annotations

import os
import time

import subprocess
import sys
import torch

# ── GPU selection ──────────────────────────────────────────────────────────────
MIN_FREE_GB = 15.0

def _gpu_memory():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.free,memory.total",
         "--format=csv,noheader,nounits"], text=True)
    rows = []
    for line in out.strip().splitlines():
        idx, free, total = [x.strip() for x in line.split(",")]
        rows.append((int(idx), int(free) / 1024, int(total) / 1024))
    return rows

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    gpus = _gpu_memory()
    best = max(gpus, key=lambda g: g[1])
    if best[1] < MIN_FREE_GB:
        print(f"ERROR: No GPU with >= {MIN_FREE_GB} GiB free.")
        for g in gpus:
            print(f"  GPU {g[0]}: {g[1]:.1f}/{g[2]:.1f} GiB free")
        sys.exit(1)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(best[0])
    print(f"Auto-selected GPU {best[0]} ({best[1]:.1f} GiB free)")

from vllm import LLM, SamplingParams

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL         = "Qwen/Qwen2.5-7B-Instruct"
INPUT_LEN     = 32
OUTPUT_LEN    = 512
NUM_PROMPTS   = 190
MAX_NUM_SEQS  = 16
GPU_MEM_UTIL  = 0.90     # GPUs are clean, use full memory for larger KV cache
PROMPT        = "Hello, how are you? " * (INPUT_LEN // 5)

gpus = _gpu_memory()
chosen = int(os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0])
g = next(r for r in gpus if r[0] == chosen)
print(f"GPU {chosen}: {g[1]:.1f}/{g[2]:.1f} GiB free  util={GPU_MEM_UTIL}")

# ── Load model once ────────────────────────────────────────────────────────────
print(f"\nLoading {MODEL} ...")
llm = LLM(
    model=MODEL,
    dtype="bfloat16",
    enforce_eager=True,
    gpu_memory_utilization=GPU_MEM_UTIL,
    max_num_seqs=MAX_NUM_SEQS,
    trust_remote_code=True,
)
print("Model loaded.\n")

params   = SamplingParams(temperature=0.0, max_tokens=OUTPUT_LEN, ignore_eos=True)
prompts  = [PROMPT] * NUM_PROMPTS


def run_and_time(label: str) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    llm.generate(prompts, params)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    total_output = NUM_PROMPTS * OUTPUT_LEN
    tps = total_output / elapsed
    print(f"  {label:12s}  {elapsed:.1f}s  {tps:.1f} output tokens/s")
    return tps


# ── Warmup (triggers Triton autotune / JIT — discard) ─────────────────────────
print("Warmup run (triggers autotune, discard result)...")
run_and_time("warmup")

# ── Measurement (warm cache) ───────────────────────────────────────────────────
print("\nMeasurement run (warm cache — real number)...")
tps = run_and_time("measurement")

print(f"\nFinal: {tps:.2f} output tokens/s")
