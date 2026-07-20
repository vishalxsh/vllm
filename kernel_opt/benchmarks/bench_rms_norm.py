"""
Benchmark: triton_fused_add_rms_norm vs vLLM's CUDA op
(_C.fused_add_rms_norm) at decode rows across Qwen/Mistral/Llama hidden
sizes. Both variants are IN-PLACE and fused with the residual add — this
is vLLM's actual hot serving-path op (what ensure_tuned races against),
not the plain out-of-place kernel.

This is the one framework kernel that currently TIES its incumbent rather
than beating it (see 2026-07-16 finding) — the Artemis campaign target for
this case is a real kernel restructuring, not config search.

Timing note: uses triton.testing.do_bench, NOT the hand-rolled CUDA-event
loop bench_skinny_gemm.py uses. At RMSNorm's scale (a few us of actual GPU
work per call vs GEMM's tens-hundreds of us), a naive back-to-back Python
loop measures Triton's heavier per-call launch overhead rather than GPU
time, understating it relative to a lean torch.ops._C call — do_bench is
the same tool ensure_tuned() itself uses to decide ENABLED, so this
benchmark and the production tuning decision agree by construction.

Repeated in-place calls without re-cloning inputs each iteration let x/r
values drift, but the kernel does fixed-shape, data-independent work, so
drift does not affect timing.

Same statistical design as bench_skinny_gemm.py: N_REPEAT independent rows
for real CIs, geometric mean as the aggregate objective.

Usage:
  .venv/bin/python kernel_opt/benchmarks/bench_rms_norm.py "$ORIG"
"""

import json
import math
import os
import pathlib
import sys

import torch
import triton.testing

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))
import vllm._custom_ops  # noqa: F401 — registers torch.ops._C
from vllm.kernels.triton.rms_norm import triton_fused_add_rms_norm

DEVICE = f"cuda:{os.environ.get('CUDA_DEVICE', '0')}"
torch.cuda.set_device(DEVICE)

HIDDEN_SIZES = [(3584, "qwen"), (4096, "mistral_llama")]
ROWS         = [1, 4, 8, 16, 32]
DTYPE        = torch.bfloat16
EPS          = 1e-6
N_REPEAT     = 7


def time_once(fn):
    return triton.testing.do_bench(fn, warmup=10, rep=50, return_mode="median") * 1000


def geomean(xs):
    if not xs or any(x <= 0 for x in xs):
        return 0.0
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


print(f"device: {DEVICE} | dtype: {DTYPE} | repeats: {N_REPEAT}")

tensors = {}
for N, label in HIDDEN_SIZES:
    w = torch.rand(N, dtype=DTYPE, device=DEVICE) * 0.5 + 0.75
    for rows in ROWS:
        # Separate buffer pairs per variant: both mutate in place, and a
        # shared pair would let one variant's drift feed the other's inputs.
        x_cuda = torch.randn(rows, N, dtype=DTYPE, device=DEVICE)
        r_cuda = torch.randn(rows, N, dtype=DTYPE, device=DEVICE)
        x_tri = x_cuda.clone()
        r_tri = r_cuda.clone()
        tensors[(N, rows)] = (w, x_cuda, r_cuda, x_tri, r_tri, label)

rows_out = []
for rep in range(N_REPEAT):
    flat = {}
    speedups = []
    all_faster = True
    for N, label in HIDDEN_SIZES:
        for rows in ROWS:
            w, x_cuda, r_cuda, x_tri, r_tri, _ = tensors[(N, rows)]
            t_cuda = time_once(lambda: torch.ops._C.fused_add_rms_norm(x_cuda, r_cuda, w, EPS))
            t_tri = time_once(lambda: triton_fused_add_rms_norm(x_tri, r_tri, w, EPS))
            sp = t_cuda / t_tri
            if sp < 1.0:
                all_faster = False
            key = f"{label}_N{N}_rows{rows}"
            flat[f"speedup_{key}"] = round(sp, 4)
            flat[f"triton_us_{key}"] = round(t_tri, 3)
            flat[f"cuda_us_{key}"] = round(t_cuda, 3)
            speedups.append(sp)

    flat["geomean_speedup_all"] = round(geomean(speedups), 4)  # objective
    flat["min_speedup_all"] = round(min(speedups), 4)
    flat["all_faster"] = 1.0 if all_faster else 0.0
    flat["repeat"] = rep
    rows_out.append(flat)
    print(f"  repeat {rep}: geomean={flat['geomean_speedup_all']:.4f} "
          f"min={flat['min_speedup_all']:.4f} all_faster={int(flat['all_faster'])}")

g = [row["geomean_speedup_all"] for row in rows_out]
print(f"\ngeomean_speedup_all over {N_REPEAT} repeats: "
      f"mean={sum(g)/len(g):.4f}  min={min(g):.4f}  max={max(g):.4f}")

out_path = (
    pathlib.Path(sys.argv[1]) / "artemis_results.json"
    if len(sys.argv) > 1
    else pathlib.Path("artemis_results.json")
)
out_path.write_text(json.dumps(rows_out, indent=2) + "\n")
print(f"Wrote {len(rows_out)} sample rows to: {out_path.resolve()}")
