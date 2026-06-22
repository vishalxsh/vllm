#!/usr/bin/env bash
# Benchmark the skinny-GEMM kernel candidate.
# Writes artemis_results.json at the repo root with all metric values.
set -euo pipefail

CUDA_VISIBLE_DEVICES=${CUDA_DEVICE:-3} \
  .venv/bin/python kernel_opt/benchmarks/bench_skinny_gemm.py
