from __future__ import annotations

import json
from pathlib import Path
from typing import List

from .results import BatchProfile, ComparisonResult


class ConsoleReport:
    """Formats profiling results for terminal output."""

    @staticmethod
    def print_profile(p: BatchProfile) -> None:
        sep = "─" * 100
        print(f"\n{sep}")
        print(f"  variant={p.variant_name}  B={p.batch_size}")
        print(f"  total GPU time : {p.total_us / 1000:.2f} ms  "
              f"({p.profile_tokens} tokens profiled)")
        print(f"  per decode step: {p.per_step_ms:.3f} ms")
        print(sep)
        print(f"  {'Rank':<5} {'Kernel':<65} {'Total (us)':>12} {'%GPU':>7} {'Calls':>7}")
        print(f"  {'-'*5} {'-'*65} {'-'*12} {'-'*7} {'-'*7}")
        for rank, k in enumerate(p.kernels, 1):
            print(
                f"  {rank:<5} {k.name[:65]:<65} "
                f"{k.total_us:>12.1f} {k.pct:>6.1f}% {k.calls:>7}"
            )

    @staticmethod
    def print_sweep(profiles: List[BatchProfile]) -> None:
        for p in profiles:
            ConsoleReport.print_profile(p)

    @staticmethod
    def print_comparison(results: List[ComparisonResult], top_diffs: int = 8) -> None:
        sep = "=" * 100

        print(f"\n{sep}")
        print("  BEFORE / AFTER — PER-KERNEL BREAKDOWN")
        print(sep)

        for r in results:
            tag = "▲ FASTER" if r.is_faster else "▼ SLOWER"
            print(
                f"\n  B={r.batch_size:2d}  "
                f"{r.baseline.variant_name} {r.baseline.per_step_ms:.3f} ms  →  "
                f"{r.candidate.variant_name} {r.candidate.per_step_ms:.3f} ms  "
                f"speedup={r.speedup:.3f}x  {tag}"
            )
            print(f"  {'Kernel':<65} {'Δ us':>10}  {'Δ %':>8}  {'direction':>10}")
            print(f"  {'-'*65} {'-'*10}  {'-'*8}  {'-'*10}")
            for diff in r.kernel_diffs[:top_diffs]:
                arrow = "slower ▲" if diff.delta_us > 0 else "faster ▼"
                print(
                    f"  {diff.name[:65]:<65} "
                    f"{diff.delta_us:>+10.1f}  {diff.pct_change:>+7.1f}%  {arrow:>10}"
                )

        print(f"\n{sep}")
        print("  SPEEDUP SUMMARY")
        print(sep)
        print(
            f"  {'B':>4}  {'baseline (ms)':>14}  {'candidate (ms)':>15}  "
            f"{'speedup':>9}  {'result':>10}"
        )
        print(f"  {'-'*4}  {'-'*14}  {'-'*15}  {'-'*9}  {'-'*10}")
        for r in results:
            tag = "FASTER ▲" if r.is_faster else "SLOWER ▼"
            print(
                f"  {r.batch_size:>4}  {r.baseline.per_step_ms:>14.3f}  "
                f"{r.candidate.per_step_ms:>15.3f}  {r.speedup:>9.3f}x  {tag:>10}"
            )
        print()


class JSONReport:
    """Serialises profiling results to JSON for Artemis ingestion."""

    @staticmethod
    def _profile_dict(p: BatchProfile) -> dict:
        return {
            "variant": p.variant_name,
            "batch_size": p.batch_size,
            "total_us": round(p.total_us, 2),
            "per_step_us": round(p.per_step_us, 2),
            "per_step_ms": round(p.per_step_ms, 3),
            "profile_tokens": p.profile_tokens,
            "top_kernels": [
                {
                    "name": k.name,
                    "total_us": round(k.total_us, 2),
                    "pct": round(k.pct, 2),
                    "calls": k.calls,
                }
                for k in p.kernels
            ],
        }

    @staticmethod
    def sweep_dict(profiles: List[BatchProfile]) -> dict:
        return {
            "mode": "sweep",
            "variant": profiles[0].variant_name if profiles else "",
            "batch_sizes": [p.batch_size for p in profiles],
            "profiles": [JSONReport._profile_dict(p) for p in profiles],
            # Flat metrics consumed directly by Artemis
            "metrics": {
                f"per_step_ms_B{p.batch_size}": round(p.per_step_ms, 3)
                for p in profiles
            },
        }

    @staticmethod
    def comparison_dict(results: List[ComparisonResult]) -> dict:
        summary = []
        details = []

        for r in results:
            summary.append(
                {
                    "batch_size": r.batch_size,
                    "baseline_variant": r.baseline.variant_name,
                    "candidate_variant": r.candidate.variant_name,
                    "baseline_per_step_ms": round(r.baseline.per_step_ms, 3),
                    "candidate_per_step_ms": round(r.candidate.per_step_ms, 3),
                    "speedup": round(r.speedup, 4),
                    "faster": r.is_faster,
                }
            )
            details.append(
                {
                    "batch_size": r.batch_size,
                    "baseline": JSONReport._profile_dict(r.baseline),
                    "candidate": JSONReport._profile_dict(r.candidate),
                    "kernel_diffs": [
                        {
                            "name": d.name,
                            "baseline_us": round(d.baseline_us, 2),
                            "candidate_us": round(d.candidate_us, 2),
                            "delta_us": round(d.delta_us, 2),
                            "pct_change": round(d.pct_change, 2),
                        }
                        for d in r.kernel_diffs[:20]
                    ],
                }
            )

        return {
            "mode": "compare",
            "baseline_variant": results[0].baseline.variant_name if results else "",
            "candidate_variant": results[0].candidate.variant_name if results else "",
            "summary": summary,
            "details": details,
            # Flat speedup metrics consumed directly by Artemis
            "metrics": {
                f"speedup_B{r.batch_size}": round(r.speedup, 4) for r in results
            },
        }

    @staticmethod
    def save(data: dict, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2) + "\n")
        print(f"Wrote results → {path.resolve()}")
