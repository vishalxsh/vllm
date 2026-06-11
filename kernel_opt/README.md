# kernel_opt

Profiling and kernel optimisation workspace for vLLM decode on Qwen2.5-7B (RTX 3090, beast3).

---

## What this does

During LLM text generation (decode phase), the GPU spends ~80% of its time on matrix multiplications
(GEMMs). This workspace:

1. **Profiles** the real vLLM decode pipeline to show which kernels are taking how much time
2. **Benchmarks** custom Triton kernels against cuBLAS to measure speedup
3. **Compares** before/after kernel swaps so Artemis can track improvements

---

## Folder structure

```
kernel_opt/
  kernels/
    gemv.py            Triton GEMV for B=1 — currently live in vLLM
    skinny_gemm.py     Triton skinny GEMM for B=1–32 — Artemis optimisation target

  profiling/           Python package — the profiling framework
    config.py          ProfileConfig dataclass (model, batch sizes, tokens, etc.)
    results.py         Data classes: KernelStat, BatchProfile, ComparisonResult
    variants.py        Kernel variants: stock_cublas, triton_gemv, triton_skinny_gemm
    runner.py          ProfileRunner — owns the vLLM engine, runs sweeps and comparisons
    report.py          ConsoleReport (terminal) + JSONReport (Artemis-compatible JSON)

  benchmarks/
    profile_vllm.py    Main CLI — sweep and before/after comparison
    profile_decode.py  Simple single-batch profiler (legacy, wraps the framework)
    bench_skinny_gemm.py  Microbenchmark: Triton vs cuBLAS across shapes and batch sizes

  artemis_results.json  Latest benchmark output (read by Artemis)
```

---

## Setup

```bash
cd /home/vishal/Desktop/vllm
source .venv/bin/activate
```

---

## 1 — Profile vLLM decode (shows % GPU time per kernel)

```bash
# Profile the current vLLM kernel at a single batch size
.venv/bin/python kernel_opt/benchmarks/profile_decode.py --batch-size 1

# Sweep all batch sizes (B=1,4,8,16,32) for the current kernel
.venv/bin/python kernel_opt/benchmarks/profile_vllm.py sweep \
    --variant triton_gemv --output kernel_opt/results/sweep_triton_gemv.json
```

---

## 2 — Before/after comparison (what Artemis uses to measure improvement)

```bash
# Compare stock cuBLAS vs current Triton GEMV at B=1
.venv/bin/python kernel_opt/benchmarks/profile_vllm.py compare \
    --baseline stock_cublas --candidate triton_gemv \
    --batch-sizes 1 \
    --output kernel_opt/results/compare_b1.json

# Compare current Triton GEMV vs new skinny GEMM across all batch sizes
.venv/bin/python kernel_opt/benchmarks/profile_vllm.py compare \
    --baseline triton_gemv --candidate triton_skinny_gemm \
    --output kernel_opt/results/compare_skinny_gemm.json
```

Available variants:

| Variant | What it runs |
|---|---|
| `stock_cublas` | Pure cuBLAS — the unmodified baseline |
| `triton_gemv` | Current vLLM — Triton GEMV at B=1, cuBLAS at B>1 |
| `triton_skinny_gemm` | Candidate — Triton tl.dot skinny GEMM at B=1–32 |

---

## 3 — Microbenchmark individual kernels (Artemis benchmark step)

```bash
ORIG=$(pwd) && .venv/bin/python kernel_opt/benchmarks/bench_skinny_gemm.py "$ORIG"
```

Writes speedup metrics to `artemis_results.json`.

---

## 4 — Correctness check (Artemis test step)

```bash
.venv/bin/python kernel_opt/kernels/skinny_gemm.py
```

Exits 0 if all shapes and batch sizes pass, non-zero otherwise.

---

## Current results (2026-05-22, beast3 GPU 0)

| Shape | B=1 | B=4 | B=8 | B=16 | B=32 |
|---|---|---|---|---|---|
| attn_proj [3584×3584] | 0.48x | 0.47x | 0.48x | 0.50x | 0.43x |
| ffn_gate [18944×3584] | 1.07x | 1.05x | 1.05x | 1.03x | 1.27x |
| ffn_down [3584×18944] | 1.10x | 1.08x | 1.08x | 1.08x | 1.12x |

`attn_proj` is slower because the square shape is a bad fit for the `tl.dot` tile strategy.
`ffn_gate` and `ffn_down` are already faster. Fixing `attn_proj` is the current blocker.

---

## Artemis config

| Step | Command |
|---|---|
| Build | `VLLM_USE_PRECOMPILED=1 uv pip install --python .venv/bin/python -e . --torch-backend=auto` |
| Test | `.venv/bin/python kernel_opt/kernels/skinny_gemm.py` |
| Benchmark | `ORIG=$(pwd) && .venv/bin/python kernel_opt/benchmarks/bench_skinny_gemm.py "$ORIG"` |
