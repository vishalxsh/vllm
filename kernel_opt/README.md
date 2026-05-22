# kernel_opt

Triton kernel optimisation workspace for vLLM decode (Qwen2.5-7B, RTX 3090).

## Setup

```bash
cd /home/vishal/Desktop/vllm
source .venv/bin/activate
```

## Kernels

| File | Scope |
|---|---|
| `kernels/gemv.py` | batch=1 GEMV — integrated into vLLM |
| `kernels/triton.py` | skinny GEMM batch=1–32 — Artemis target |

## Run correctness check

```bash
# GEMV (batch=1)
.venv/bin/python kernel_opt/kernels/gemv.py

# Skinny GEMM (batch=1-32)
.venv/bin/python kernel_opt/kernels/triton.py
```

## Run benchmarks

```bash
# GEMV benchmark
ORIG=$(pwd) && .venv/bin/python kernel_opt/benchmarks/bench_7b.py "$ORIG"

# Skinny GEMM benchmark
ORIG=$(pwd) && .venv/bin/python kernel_opt/benchmarks/bench_skinny_gemm.py "$ORIG"
```

Results written to `artemis_results.json` at the vLLM root.

## Promote a finished kernel to vLLM

```bash
cp kernel_opt/kernels/triton.py vllm/kernels/triton/triton.py
# then update vllm/model_executor/layers/utils.py dispatch
```

## Artemis config

| Step | Command |
|---|---|
| Build | `VLLM_USE_PRECOMPILED=1 uv pip install --python /home/vishal/Desktop/vllm/.venv/bin/python -e /home/vishal/Desktop/vllm --torch-backend=auto` |
| Test | `/home/vishal/Desktop/vllm/.venv/bin/python kernel_opt/kernels/triton.py` |
| Benchmark | `ORIG=$(pwd) && /home/vishal/Desktop/vllm/.venv/bin/python kernel_opt/benchmarks/bench_skinny_gemm.py "$ORIG"` |
