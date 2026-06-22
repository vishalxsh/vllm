#!/usr/bin/env bash
# Build the vLLM candidate on the Artemis runner.
# Called by Artemis after each generated candidate is applied.
set -euo pipefail

uv venv --python 3.12
VLLM_VERSION_OVERRIDE=0.1.0 VLLM_USE_PRECOMPILED=1 \
  uv pip install -p .venv -e . --torch-backend=auto
