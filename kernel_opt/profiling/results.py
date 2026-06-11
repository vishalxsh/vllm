from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class KernelStat:
    name: str
    total_us: float
    pct: float
    calls: int


@dataclass
class BatchProfile:
    variant_name: str
    batch_size: int
    total_us: float
    profile_tokens: int
    kernels: List[KernelStat]

    @property
    def per_step_us(self) -> float:
        return self.total_us / self.profile_tokens

    @property
    def per_step_ms(self) -> float:
        return self.per_step_us / 1000


@dataclass
class KernelDiff:
    name: str
    baseline_us: float
    candidate_us: float
    baseline_pct: float
    candidate_pct: float

    @property
    def delta_us(self) -> float:
        return self.candidate_us - self.baseline_us

    @property
    def pct_change(self) -> float:
        if self.baseline_us == 0:
            return 0.0
        return self.delta_us / self.baseline_us * 100


@dataclass
class ComparisonResult:
    batch_size: int
    baseline: BatchProfile
    candidate: BatchProfile

    @property
    def speedup(self) -> float:
        if self.candidate.per_step_us == 0:
            return 0.0
        return self.baseline.per_step_us / self.candidate.per_step_us

    @property
    def is_faster(self) -> bool:
        return self.speedup >= 1.0

    @property
    def kernel_diffs(self) -> List[KernelDiff]:
        baseline_map = {k.name: k for k in self.baseline.kernels}
        candidate_map = {k.name: k for k in self.candidate.kernels}
        all_names = set(baseline_map) | set(candidate_map)

        diffs = [
            KernelDiff(
                name=name,
                baseline_us=baseline_map[name].total_us if name in baseline_map else 0.0,
                candidate_us=candidate_map[name].total_us if name in candidate_map else 0.0,
                baseline_pct=baseline_map[name].pct if name in baseline_map else 0.0,
                candidate_pct=candidate_map[name].pct if name in candidate_map else 0.0,
            )
            for name in all_names
        ]
        # Sort by absolute delta so the biggest movers appear first
        return sorted(diffs, key=lambda d: abs(d.delta_us), reverse=True)
