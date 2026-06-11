"""
Benchmark: triton_skinny_gemm vs cuBLAS across batch sizes 1-32.

Covers all three Qwen2.5-7B shapes in bfloat16 on RTX 3090.
Writes speedup metrics to artemis_results.json for Artemis tracking.

Usage:
  ORIG=$(pwd) && /home/vishal/Desktop/vllm/.venv/bin/python \
    kernel_opt/benchmarks/bench_skinny_gemm.py "$ORIG"
"""

import json
import os
import pathlib
import statistics
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from kernels.skinny_gemm import triton_skinny_gemm

DEVICE  = f"cuda:{os.environ.get('CUDA_DEVICE', '0')}"
torch.cuda.set_device(DEVICE)

SHAPES = [
    (3584,  3584,  "attn_proj   "),
    (37888, 3584,  "gate_up_proj"),
    (3584,  18944, "ffn_down    "),
]
BATCH_SIZES = [1, 4, 8, 16, 32]
DTYPE       = torch.bfloat16
N_RUNS      = 5
N_WARMUP    = 100
N_MEASURE   = 300


def bench(fn):
    times = []
    for _ in range(N_RUNS):
        for _ in range(N_WARMUP):
            fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(N_MEASURE):
            fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / N_MEASURE * 1000)
    return statistics.median(times)


print(f"device: {DEVICE}  |  dtype: {DTYPE}  |  runs: {N_RUNS}  |  warmup: {N_WARMUP}  |  measure: {N_MEASURE}")
print(f"{'Shape':>38}  {'cuBLAS (us)':>12}  {'Triton (us)':>12}  {'Speedup':>8}")
print("-" * 80)

results    = {}
all_faster = True

for B in BATCH_SIZES:
    for M, K, label in SHAPES:
        W = torch.randn(M, K, dtype=DTYPE, device=DEVICE)
        X = torch.randn(B, K, dtype=DTYPE, device=DEVICE)

        t_cub = bench(lambda: torch.mm(X, W.t()))
        try:
            t_tri = bench(lambda: triton_skinny_gemm(W, X))
        except Exception as e:
            print(f"  ERROR {label} B={B}: {e}")
            t_tri = float("inf")

        sp   = t_cub / t_tri if t_tri != float("inf") else 0.0
        flag = "FASTER" if sp >= 1.0 else "SLOWER"
        if sp < 1.0:
            all_faster = False

        key = f"{label.strip()}_B{B}"
        print(f"  B={B:2d} {label} [{M:6d},{K:6d}]  {t_cub:>12.2f}  {t_tri:>12.2f}  {sp:>7.2f}x  {flag}")
        results[key] = {
            "batch":     B,
            "cublas_us": round(t_cub, 2),
            "triton_us": round(t_tri, 2) if t_tri != float("inf") else -1.0,
            "speedup":   round(sp, 4),
        }
    print()

print("Verdict:", "ALL SHAPES FASTER" if all_faster else "SOME SHAPES SLOWER")

# Flat metrics for Artemis
flat = {}
for key, v in results.items():
    flat[f"speedup_{key}"]    = v["speedup"]
    flat[f"triton_us_{key}"]  = v["triton_us"]

for B in BATCH_SIZES:
    batch_speedups = [
        v["speedup"] for k, v in results.items()
        if v["batch"] == B and v["speedup"] > 0
    ]
    flat[f"avg_speedup_B{B}"] = (
        round(sum(batch_speedups) / len(batch_speedups), 4)
        if batch_speedups else 0.0
    )

flat["avg_speedup_all"] = round(
    sum(v["speedup"] for v in results.values() if v["speedup"] > 0)
    / max(sum(1 for v in results.values() if v["speedup"] > 0), 1),
    4,
)
flat["all_faster"] = 1.0 if all_faster else 0.0

out_path = (
    pathlib.Path(sys.argv[1]) / "artemis_results.json"
    if len(sys.argv) > 1
    else pathlib.Path("artemis_results.json")
)
out_path.write_text(json.dumps([flat], indent=2) + "\n")
print(f"Wrote metrics to: {out_path.resolve()}")
