"""
Single-batch decode profiler — thin wrapper around the profiling framework.

For full multi-batch profiling and before/after comparison use profile_vllm.py.

Usage:
  .venv/bin/python kernel_opt/benchmarks/profile_decode.py
  .venv/bin/python kernel_opt/benchmarks/profile_decode.py --batch-size 8
"""

from __future__ import annotations

import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from kernel_opt.profiling import ProfileConfig, ProfileRunner, ConsoleReport
from kernel_opt.profiling.variants import TritonGemvVariant


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-batch vLLM decode profiler")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--profile-tokens", type=int, default=10)
    parser.add_argument("--warmup-tokens", type=int, default=20)
    args = parser.parse_args()

    config = ProfileConfig(
        batch_sizes=[args.batch_size],
        profile_tokens=args.profile_tokens,
        warmup_tokens=args.warmup_tokens,
    )
    runner = ProfileRunner(config)
    profile = runner.profile_batch(args.batch_size, TritonGemvVariant())
    ConsoleReport.print_profile(profile)


if __name__ == "__main__":
    main()
