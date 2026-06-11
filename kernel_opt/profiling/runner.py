"""
ProfileRunner — owns a single vLLM engine and orchestrates profiling sweeps.

Multi-batch simulation
----------------------
To simulate B=N decode, we submit N identical prompts simultaneously.
vLLM prefills all N in one chunked pass, then decodes all N together in each
subsequent step, giving an effective decode batch size of N.
"""

from __future__ import annotations

import collections
import os
import subprocess
from typing import List

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from .config import ProfileConfig
from .results import BatchProfile, ComparisonResult, KernelStat
from .variants import KernelVariant


class ProfileRunner:
    """
    Manages a single vLLM LLM instance across all profiling calls.

    The engine is initialised lazily on the first call to `load()` or any
    profiling method.  Reusing one engine across variants avoids the
    multi-minute reload cost and ensures a fair comparison.
    """

    MIN_FREE_GB = 14.0

    def __init__(self, config: ProfileConfig) -> None:
        self.config = config
        self._llm = None
        self._gpu_idx = self._select_gpu()

        # Must be set before any vLLM or CUDA import
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self._gpu_idx)
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    # ── GPU selection ──────────────────────────────────────────────────────────

    def _select_gpu(self) -> int:
        rows = self._nvidia_smi_memory()
        print("GPU memory at startup:")
        for idx, free, total in rows:
            print(f"  GPU {idx}: {free:.1f} / {total:.1f} GiB free")

        if "CUDA_VISIBLE_DEVICES" in os.environ:
            chosen = int(os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0])
            match = next((r for r in rows if r[0] == chosen), None)
            if match is None or match[1] < self.MIN_FREE_GB:
                raise RuntimeError(
                    f"GPU {chosen} has {match[1] if match else '?':.1f} GiB free; "
                    f"need >= {self.MIN_FREE_GB:.0f} GiB."
                )
            return chosen

        best = max(rows, key=lambda r: r[1])
        if best[1] < self.MIN_FREE_GB:
            raise RuntimeError(
                f"No GPU has >= {self.MIN_FREE_GB:.0f} GiB free for {self.config.model_id}."
            )
        return best[0]

    @staticmethod
    def _nvidia_smi_memory() -> list[tuple[int, float, float]]:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        rows = []
        for line in out.strip().splitlines():
            idx, free_mb, total_mb = [x.strip() for x in line.split(",")]
            rows.append((int(idx), int(free_mb) / 1024, int(total_mb) / 1024))
        return rows

    # ── Engine lifecycle ───────────────────────────────────────────────────────

    def load(self) -> None:
        """Initialise the vLLM engine if not already loaded."""
        if self._llm is not None:
            return

        from vllm import LLM

        rows = self._nvidia_smi_memory()
        match = next(r for r in rows if r[0] == self._gpu_idx)
        free_gb, total_gb = match[1], match[2]
        util = min(0.92, max(0.50, (free_gb - 1.0) / total_gb))

        print(
            f"\nLoading {self.config.model_id} on GPU {self._gpu_idx} "
            f"({free_gb:.1f}/{total_gb:.1f} GiB free, util={util:.2f}) ..."
        )
        self._llm = LLM(
            model=self.config.model_id,
            dtype=self.config.dtype,
            enforce_eager=True,
            gpu_memory_utilization=util,
            tensor_parallel_size=self.config.tensor_parallel_size,
            trust_remote_code=self.config.trust_remote_code,
        )
        print("Model loaded.\n")

    # ── Core profiling ─────────────────────────────────────────────────────────

    def _warmup(self, batch_size: int) -> None:
        from vllm import SamplingParams

        prompts = [self.config.prompt] * batch_size
        params = SamplingParams(temperature=0.0, max_tokens=self.config.warmup_tokens)
        self._llm.generate(prompts, params)
        torch.cuda.synchronize()

    def profile_batch(
        self,
        batch_size: int,
        variant: KernelVariant,
    ) -> BatchProfile:
        """Profile one (batch_size, variant) pair and return a BatchProfile."""
        from vllm import SamplingParams

        self.load()

        prompts = [self.config.prompt] * batch_size
        params = SamplingParams(temperature=0.0, max_tokens=self.config.profile_tokens)

        print(f"  warmup  B={batch_size:2d}  variant={variant.name}")
        self._warmup(batch_size)

        print(f"  profile B={batch_size:2d}  variant={variant.name}")

        with variant.active():
            with profile(
                activities=[ProfilerActivity.CUDA],
                record_shapes=False,
                with_stack=False,
                acc_events=True,
            ) as prof:
                with record_function("vllm_generate"):
                    self._llm.generate(prompts, params)
                torch.cuda.synchronize()

        return self._aggregate(prof, variant.name, batch_size)

    def _aggregate(
        self,
        prof,
        variant_name: str,
        batch_size: int,
    ) -> BatchProfile:
        kernel_times: dict[str, dict] = collections.defaultdict(
            lambda: {"count": 0, "us": 0.0}
        )
        for evt in prof.events():
            if evt.device_type == torch.autograd.DeviceType.CUDA:
                kernel_times[evt.name]["count"] += 1
                kernel_times[evt.name]["us"] += evt.device_time

        total_us = sum(v["us"] for v in kernel_times.values())
        sorted_kernels = sorted(
            kernel_times.items(), key=lambda x: x[1]["us"], reverse=True
        )

        kernels = [
            KernelStat(
                name=name,
                total_us=info["us"],
                pct=100.0 * info["us"] / total_us if total_us else 0.0,
                calls=info["count"],
            )
            for name, info in sorted_kernels[: self.config.top_n]
        ]

        return BatchProfile(
            variant_name=variant_name,
            batch_size=batch_size,
            total_us=total_us,
            profile_tokens=self.config.profile_tokens,
            kernels=kernels,
        )

    # ── Sweep and compare ──────────────────────────────────────────────────────

    def sweep(self, variant: KernelVariant) -> List[BatchProfile]:
        """Profile all configured batch sizes for one variant."""
        results = []
        for bs in self.config.batch_sizes:
            results.append(self.profile_batch(bs, variant))
        return results

    def compare(
        self,
        baseline: KernelVariant,
        candidate: KernelVariant,
    ) -> List[ComparisonResult]:
        """Profile baseline then candidate at every batch size and return diffs."""
        comparisons = []
        for bs in self.config.batch_sizes:
            b = self.profile_batch(bs, baseline)
            c = self.profile_batch(bs, candidate)
            comparisons.append(ComparisonResult(batch_size=bs, baseline=b, candidate=c))
        return comparisons
