"""
Artemis benchmark: our bucket=4 (B=1) Triton kernels vs torch.compile
max-autotune, across the production GEMM/fused shapes.

Target: the one remaining gap found by bench_torch_compile.py (2026-07-14,
re-checked after the 2026-07-20 B<=4 bucket fix) — Inductor's own autotuned
Triton GEMM still beats ours at B=1 on most shapes (0.89-0.99x), because it
autotunes per EXACT batch size while our tuner only searches a pruned
12-candidate space per bucket. This is the Artemis campaign target: widen
or restructure the bucket=4 search, not cuBLAS-race tuning (already won).

torch.compile is compiled ONCE per shape (outside the repeat loop) — only
the timing measurement is repeated for real confidence intervals, per the
project's statistical-rigor convention (see bench_skinny_gemm.py).

Usage:
  .venv/bin/python kernel_opt/benchmarks/bench_vs_torch_compile_b1.py "$ORIG"
"""

import json
import math
import os
import pathlib
import sys

import torch
import torch.nn.functional as F
import triton.testing

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))
from vllm.kernels.triton.skinny_gemm import ensure_tuned as sg_ensure_tuned
from vllm.kernels.triton.skinny_gemm import triton_skinny_gemm
from vllm.kernels.triton.fused_gate_up_silu import ensure_tuned as fu_ensure_tuned
from vllm.kernels.triton.fused_gate_up_silu import triton_fused_gate_up_silu

DEVICE = f"cuda:{os.environ.get('CUDA_DEVICE', '0')}"
torch.cuda.set_device(DEVICE)
torch.manual_seed(0)

B = 1  # the bucket=4 target batch size
N_REPEAT = 7

GEMM_SHAPES = [
    ("qwen7b_qkv", 4608, 3584),
    ("qwen7b_o", 3584, 3584),
    ("qwen7b_gate_up", 37888, 3584),
    ("qwen7b_down", 3584, 18944),
    ("mistral7b_qkv", 6144, 4096),
    ("mistral7b_lm_head", 32768, 4096),
]
FUSED_SHAPES = [
    ("qwen7b_fused_gate_up_silu", 37888, 3584),   # M_half = 18944
    ("mistral7b_fused_gate_up_silu", 28672, 4096),  # M_half = 14336
]


def gemm_eager(x, w):
    return F.linear(x, w)


def fused_eager(x, w):
    y = F.linear(x, w)
    gate, up = y.chunk(2, dim=-1)
    return F.silu(gate) * up


def compile_tc_best(fn):
    import torch._dynamo
    import torch._inductor.config as icfg
    torch._dynamo.reset()
    icfg.max_autotune_gemm_backends = "ATEN,TRITON"
    return torch.compile(fn, mode="max-autotune-no-cudagraphs", dynamic=False)


def time_once(fn):
    return triton.testing.do_bench(fn, warmup=15, rep=80, return_mode="median") * 1000


def geomean(xs):
    if not xs or any(x <= 0 for x in xs):
        return 0.0
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


print(f"device: {DEVICE} | B={B} | repeats: {N_REPEAT}")

# Compile once per shape; cache the compiled callables and inputs.
cases = []
for name, M, K in GEMM_SHAPES:
    W = torch.randn(M, K, dtype=torch.bfloat16, device=DEVICE) * 0.02
    sg_ensure_tuned(W)
    X = torch.randn(B, K, dtype=torch.bfloat16, device=DEVICE)
    tc_fn = compile_tc_best(lambda x, w=W: gemm_eager(x, w))
    tc_fn(X)  # trigger compile + autotune
    ours_fn = lambda x=X, w=W: triton_skinny_gemm(w, x)
    tc_call = lambda x=X, fn=tc_fn: fn(x)
    cases.append((name, ours_fn, tc_call))
    print(f"  compiled {name} [{M},{K}]")

for name, M, K in FUSED_SHAPES:
    W = torch.randn(M, K, dtype=torch.bfloat16, device=DEVICE) * 0.02
    fu_ensure_tuned(W)
    X = torch.randn(B, K, dtype=torch.bfloat16, device=DEVICE)
    tc_fn = compile_tc_best(lambda x, w=W: fused_eager(x, w))
    tc_fn(X)
    ours_fn = lambda x=X, w=W: triton_fused_gate_up_silu(w, x)
    tc_call = lambda x=X, fn=tc_fn: fn(x)
    cases.append((name, ours_fn, tc_call))
    print(f"  compiled {name} [{M},{K}]")

rows = []
for rep in range(N_REPEAT):
    flat = {}
    speedups = []
    all_faster = True
    for name, ours_fn, tc_call in cases:
        t_ours = time_once(ours_fn)
        t_tc = time_once(tc_call)
        sp = t_tc / t_ours
        if sp < 1.0:
            all_faster = False
        flat[f"speedup_vs_tc_{name}"] = round(sp, 4)
        flat[f"ours_us_{name}"] = round(t_ours, 3)
        flat[f"tc_us_{name}"] = round(t_tc, 3)
        speedups.append(sp)

    flat["geomean_speedup_vs_tc_all"] = round(geomean(speedups), 4)  # objective
    flat["min_speedup_vs_tc_all"] = round(min(speedups), 4)
    flat["all_faster_than_tc"] = 1.0 if all_faster else 0.0
    flat["repeat"] = rep
    rows.append(flat)
    print(f"  repeat {rep}: geomean={flat['geomean_speedup_vs_tc_all']:.4f} "
          f"min={flat['min_speedup_vs_tc_all']:.4f} "
          f"all_faster={int(flat['all_faster_than_tc'])}")

g = [row["geomean_speedup_vs_tc_all"] for row in rows]
print(f"\ngeomean_speedup_vs_tc_all over {N_REPEAT} repeats: "
      f"mean={sum(g)/len(g):.4f}  min={min(g):.4f}  max={max(g):.4f}")

out_path = (
    pathlib.Path(sys.argv[1]) / "artemis_results.json"
    if len(sys.argv) > 1
    else pathlib.Path("artemis_results.json")
)
out_path.write_text(json.dumps(rows, indent=2) + "\n")
print(f"Wrote {len(rows)} sample rows to: {out_path.resolve()}")
