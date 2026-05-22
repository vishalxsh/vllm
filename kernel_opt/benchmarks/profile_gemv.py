"""
Kernel profiler — shows ALL CUDA kernels and their % of GPU time.
Run manually to confirm which kernel is the bottleneck.

Usage:
  ~/vllm-env/bin/python ~/vllm-kernel-opt/benchmarks/profile.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.profiler

DEVICE = "cuda"
SHAPES = [
    (3584,  3584,  "attn_proj"),
    (18944, 3584,  "ffn_gate "),
    (3584,  18944, "ffn_down "),
]


def main():
    torch.manual_seed(0)
    inputs = [
        (torch.randn(M, N, dtype=torch.float16, device=DEVICE),
         torch.randn(N,    dtype=torch.float16, device=DEVICE), lbl)
        for M, N, lbl in SHAPES
    ]

    # Warm up
    for W, x, _ in inputs:
        torch.mv(W, x)
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for _ in range(20):
            for W, x, _ in inputs:
                torch.mv(W, x)
        torch.cuda.synchronize()

    events = sorted(prof.key_averages(), key=lambda e: e.device_time_total, reverse=True)
    total  = sum(e.device_time_total for e in events)

    print(f"Device : {torch.cuda.get_device_name(0)}")
    print()
    print(f"  {'Kernel':<58} {'Time (us)':>10}  {'%':>6}")
    print(f"  {'-'*58} {'-'*10}  {'-'*6}")
    for e in events[:10]:
        pct  = 100 * e.device_time_total / total if total > 0 else 0
        name = e.key[:58]
        print(f"  {name:<58} {e.device_time_total/20:>10.1f}  {pct:>5.1f}%")

    print()
    print(f"  Total per decode step: {total/20/1000:.3f} ms")


if __name__ == "__main__":
    main()
