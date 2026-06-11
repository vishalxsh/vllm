"""
vLLM kernel profiling framework — CLI entrypoint.

Modes
-----
  sweep    Profile all batch sizes for one kernel variant.
  compare  Before/after comparison between two variants at all batch sizes.

Variants
--------
  stock_cublas       Baseline — pure cuBLAS for all shapes and batch sizes
  triton_gemv        Current vLLM — Triton scalar GEMV at B=1, cuBLAS at B>1
  triton_skinny_gemm Candidate — Triton tl.dot skinny GEMM at B=1–32

Usage
-----
  # Sweep the current vLLM kernel across B=1-32
  .venv/bin/python kernel_opt/benchmarks/profile_vllm.py sweep \\
      --variant triton_gemv --output results/sweep_triton_gemv.json

  # Compare stock cuBLAS vs current Triton GEMV at B=1 only
  .venv/bin/python kernel_opt/benchmarks/profile_vllm.py compare \\
      --baseline stock_cublas --candidate triton_gemv \\
      --batch-sizes 1 --output results/compare_b1.json

  # Full before/after across all batch sizes
  .venv/bin/python kernel_opt/benchmarks/profile_vllm.py compare \\
      --baseline triton_gemv --candidate triton_skinny_gemm \\
      --output results/compare_skinny_gemm.json
"""

from __future__ import annotations

# MUST be set before any vLLM import — forces the engine into the same
# process so torch.profiler can capture GPU events.
import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import argparse
import pathlib
import sys

# Make kernel_opt a top-level package when run from the vLLM root
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from kernel_opt.profiling import (
    ProfileConfig,
    ProfileRunner,
    ConsoleReport,
    JSONReport,
)
from kernel_opt.profiling.variants import REGISTRY


def _build_config(args: argparse.Namespace) -> ProfileConfig:
    return ProfileConfig(
        model_id=args.model,
        batch_sizes=args.batch_sizes,
        warmup_tokens=args.warmup_tokens,
        profile_tokens=args.profile_tokens,
        top_n=args.top_n,
    )


def cmd_sweep(args: argparse.Namespace) -> None:
    variant = REGISTRY[args.variant]
    config = _build_config(args)
    runner = ProfileRunner(config)

    print(f"\nMode   : sweep")
    print(f"Variant: {variant.name} — {variant.description}")
    print(f"Batches: {config.batch_sizes}\n")

    profiles = runner.sweep(variant)
    ConsoleReport.print_sweep(profiles)

    if args.output:
        data = JSONReport.sweep_dict(profiles)
        JSONReport.save(data, pathlib.Path(args.output))


def cmd_compare(args: argparse.Namespace) -> None:
    baseline = REGISTRY[args.baseline]
    candidate = REGISTRY[args.candidate]
    config = _build_config(args)
    runner = ProfileRunner(config)

    print(f"\nMode     : compare")
    print(f"Baseline : {baseline.name} — {baseline.description}")
    print(f"Candidate: {candidate.name} — {candidate.description}")
    print(f"Batches  : {config.batch_sizes}\n")

    results = runner.compare(baseline, candidate)

    for r in results:
        ConsoleReport.print_profile(r.baseline)
        ConsoleReport.print_profile(r.candidate)

    ConsoleReport.print_comparison(results)

    if args.output:
        data = JSONReport.comparison_dict(results)
        JSONReport.save(data, pathlib.Path(args.output))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="vLLM kernel profiling framework",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", default="Qwen/Qwen2.5-7B-Instruct", help="HuggingFace model ID"
    )
    parser.add_argument(
        "--batch-sizes", nargs="+", type=int, default=[1, 4, 8, 16, 32],
        metavar="B", help="Batch sizes to profile"
    )
    parser.add_argument("--warmup-tokens", type=int, default=20)
    parser.add_argument("--profile-tokens", type=int, default=10)
    parser.add_argument("--top-n", type=int, default=20, help="Top N kernels to report")

    sub = parser.add_subparsers(dest="cmd", required=True)

    sweep_p = sub.add_parser("sweep", help="Profile one variant across all batch sizes")
    sweep_p.add_argument(
        "--variant", choices=list(REGISTRY), default="triton_gemv",
        help="Kernel variant to profile"
    )
    sweep_p.add_argument(
        "--output", type=str, default=None, metavar="PATH",
        help="Write JSON results to this path"
    )

    compare_p = sub.add_parser("compare", help="Before/after comparison between two variants")
    compare_p.add_argument(
        "--baseline", choices=list(REGISTRY), default="stock_cublas",
        help="Baseline variant"
    )
    compare_p.add_argument(
        "--candidate", choices=list(REGISTRY), default="triton_gemv",
        help="Candidate variant"
    )
    compare_p.add_argument(
        "--output", type=str, default=None, metavar="PATH",
        help="Write JSON results to this path"
    )

    args = parser.parse_args()

    if args.cmd == "sweep":
        cmd_sweep(args)
    elif args.cmd == "compare":
        cmd_compare(args)


if __name__ == "__main__":
    main()
