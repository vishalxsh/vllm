#!/usr/bin/env bash
# Correctness test for the skinny-GEMM kernel candidate.
# Exits 0 if correct, non-zero if the candidate broke anything.
set -euo pipefail

CUDA_VISIBLE_DEVICES=${CUDA_DEVICE:-3} \
  .venv/bin/python kernel_opt/kernels/skinny_gemm.py
