"""
Benchmark: triton_skinny_gemm vs cuBLAS across batch sizes 1-32.

Covers all three Qwen2.5-7B shapes in bfloat16 on RTX 3090.
Writes speedup metrics to artemis_results.json for Artemis tracking.

Statistical design (2026-06-12):
  * Emits ONE ROW PER REPEAT (N_REPEAT independent samples), not a single
    collapsed median — so Artemis sees real variance and computes confidence
    intervals. With n=1 you cannot tell a true gain from measurement noise.
  * Aggregates speedups with the GEOMETRIC mean (the correct way to average
    ratios; the arithmetic mean is biased and over-weights large wins). Also
    emits min_speedup_all (worst-shape regression guard) and all_faster.

Usage:
  ORIG=$(pwd) && /home/vishal/Desktop/vllm/.venv/bin/python \
    kernel_opt/benchmarks/bench_skinny_gemm.py "$ORIG"
"""

import json
import math
import os
import pathlib
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
N_REPEAT    = 7      # independent samples written as separate rows -> CIs
N_WARMUP    = 100
N_MEASURE   = 300


def time_once(fn):
    """One timed sample: mean us per call over N_MEASURE iterations."""
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(N_MEASURE):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / N_MEASURE * 1000


def geomean(xs):
    """Geometric mean — the correct aggregate for ratios. 0 if any sample <= 0
    (a failed/slower-than-undefined shape makes the whole repeat invalid)."""
    if not xs or any(x <= 0 for x in xs):
        return 0.0
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


print(f"device: {DEVICE} | dtype: {DTYPE} | repeats: {N_REPEAT} | "
      f"warmup: {N_WARMUP} | measure: {N_MEASURE}")

# Pre-build tensors once and warm up each shape once; then collect N_REPEAT
# independent paired (cuBLAS, Triton) samples per shape.
tensors = {}
for B in BATCH_SIZES:
    for M, K, label in SHAPES:
        W = torch.randn(M, K, dtype=DTYPE, device=DEVICE)
        X = torch.randn(B, K, dtype=DTYPE, device=DEVICE)
        for _ in range(N_WARMUP):
            torch.mm(X, W.t())
            triton_skinny_gemm(W, X)
        tensors[(B, M, K)] = (W, X, label)

rows = []        # one dict per repeat
for r in range(N_REPEAT):
    flat = {}
    speedups = []          # all 15 shape speedups this repeat (for geomean/min)
    all_faster = True
    for B in BATCH_SIZES:
        for M, K, label in SHAPES:
            W, X, _ = tensors[(B, M, K)]
            t_cub = time_once(lambda: torch.mm(X, W.t()))
            try:
                t_tri = time_once(lambda: triton_skinny_gemm(W, X))
            except Exception as e:
                print(f"  repeat {r} ERROR {label} B={B}: {e}")
                t_tri = float("inf")
            sp = t_cub / t_tri if t_tri != float("inf") else 0.0
            if sp < 1.0:
                all_faster = False
            key = f"{label.strip()}_B{B}"
            flat[f"speedup_{key}"]   = round(sp, 4)
            flat[f"triton_us_{key}"] = round(t_tri, 2) if t_tri != float("inf") else -1.0
            flat[f"cublas_us_{key}"] = round(t_cub, 2)
            speedups.append(sp)

    # per-batch arithmetic means kept for readability; aggregate objective is geomean
    for B in BATCH_SIZES:
        bs = [flat[f"speedup_{lab.strip()}_B{B}"] for _, _, lab in SHAPES]
        flat[f"avg_speedup_B{B}"] = round(sum(bs) / len(bs), 4)

    flat["avg_speedup_all"]    = round(sum(speedups) / len(speedups), 4)   # legacy
    flat["geomean_speedup_all"] = round(geomean(speedups), 4)              # objective
    flat["min_speedup_all"]     = round(min(speedups), 4)                  # regression guard
    flat["all_faster"]          = 1.0 if all_faster else 0.0
    flat["repeat"]              = r
    rows.append(flat)
    print(f"  repeat {r}: geomean={flat['geomean_speedup_all']:.4f} "
          f"min={flat['min_speedup_all']:.4f} all_faster={int(flat['all_faster'])}")

# Summary across repeats (informational)
g = [row["geomean_speedup_all"] for row in rows]
print(f"\ngeomean_speedup_all over {N_REPEAT} repeats: "
      f"mean={sum(g)/len(g):.4f}  min={min(g):.4f}  max={max(g):.4f}")

out_path = (
    pathlib.Path(sys.argv[1]) / "artemis_results.json"
    if len(sys.argv) > 1
    else pathlib.Path("artemis_results.json")
)
out_path.write_text(json.dumps(rows, indent=2) + "\n")
print(f"Wrote {len(rows)} sample rows to: {out_path.resolve()}")
