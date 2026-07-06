"""
Kernel variants for before/after profiling.

Swapping strategy
-----------------
`vllm/model_executor/layers/utils._cuda_gemm_dispatch_impl` contains a lazy
import at call time:

    from vllm.kernels.triton.gemv import triton_gemv

Because the import happens inside the function body (not at module load), it
resolves `vllm.kernels.triton.gemv.triton_gemv` on every call.  Patching the
module attribute is therefore a clean, fully reversible hook — no torch op
re-registration required.

Limitations
-----------
The `x.shape[0] == 1` guard in the dispatch means the Triton path is only
reached at B=1.  At B>1, all variants fall through to cuBLAS via
`torch.nn.functional.linear`.  TritonSkinnyGemmVariant at B>1 therefore shows
the same behaviour as the other variants today; it will diverge once
skinny_gemm is fully integrated into the dispatch.
"""

from __future__ import annotations

import abc
from contextlib import contextmanager
from typing import Iterator

import torch


class KernelVariant(abc.ABC):
    name: str
    description: str

    @contextmanager
    def active(self) -> Iterator[None]:
        """Context manager that activates this variant for the duration of the block."""
        yield

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


# ---------------------------------------------------------------------------
# Concrete variants
# ---------------------------------------------------------------------------

class StockCuBLASVariant(KernelVariant):
    """Baseline: pure cuBLAS for all shapes and batch sizes.

    Patches both dispatch paths so no Triton kernel is called:
    - vllm.kernels.triton.skinny_gemm.triton_skinny_gemm → torch.mm (cuBLAS)
      This covers the cuda_gemm_dispatch custom op (B=1–32, all 3 shapes).
    - vllm.kernels.triton.gemv.triton_gemv → torch.mv (cuBLAS GEMV)
      This covers any remaining B=1 GEMV fallthrough.
    """

    name = "stock_cublas"
    description = "Baseline — cuBLAS GEMV/GEMM for all shapes and batch sizes"

    @contextmanager
    def active(self) -> Iterator[None]:
        import vllm.kernels.triton.gemv as gemv_mod
        import vllm.kernels.triton.skinny_gemm as skinny_mod
        from vllm.model_executor.models.llama import LlamaMLP
        from vllm.model_executor.models.qwen2 import Qwen2MLP

        original_gemv   = gemv_mod.triton_gemv
        original_skinny = skinny_mod.triton_skinny_gemm
        original_qwen_forward  = Qwen2MLP.forward
        original_llama_forward = LlamaMLP.forward

        def _cublas_gemv(weight: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            return torch.mv(weight, x)

        def _cublas_gemm(W: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.linear(X, W)

        def _unfused_forward(self_mlp, x):
            gate_up, _ = self_mlp.gate_up_proj(x)
            x = self_mlp.act_fn(gate_up)
            x, _ = self_mlp.down_proj(x)
            return x

        gemv_mod.triton_gemv         = _cublas_gemv
        skinny_mod.triton_skinny_gemm = _cublas_gemm
        Qwen2MLP.forward             = _unfused_forward
        LlamaMLP.forward             = _unfused_forward
        try:
            yield
        finally:
            gemv_mod.triton_gemv         = original_gemv
            skinny_mod.triton_skinny_gemm = original_skinny
            Qwen2MLP.forward             = original_qwen_forward
            LlamaMLP.forward             = original_llama_forward


class TritonGemvVariant(KernelVariant):
    """Current vLLM state: Triton scalar GEMV for B=1, cuBLAS for B>1.

    No patching required — this is the default dispatch in utils.py.
    """

    name = "triton_gemv"
    description = "Current vLLM — Triton GEMV (gemv.py) at B=1, cuBLAS at B>1"

    # active() inherits the no-op from KernelVariant


class TritonSkinnyGemmVariant(KernelVariant):
    """Triton skinny GEMM only — fused gate+SiLU disabled.

    Patches Qwen2MLP.forward() back to the unfused two-step path so we can
    isolate the incremental effect of the fused kernel.
    """

    name = "triton_skinny_gemm"
    description = "Triton skinny GEMM at B=1–32, unfused SiLU (baseline for fusion)"

    @contextmanager
    def active(self) -> Iterator[None]:
        from vllm.model_executor.models.qwen2 import Qwen2MLP

        original_forward = Qwen2MLP.forward

        def _unfused_forward(self_mlp, x):
            gate_up, _ = self_mlp.gate_up_proj(x)
            x = self_mlp.act_fn(gate_up)
            x, _ = self_mlp.down_proj(x)
            return x

        Qwen2MLP.forward = _unfused_forward
        try:
            yield
        finally:
            Qwen2MLP.forward = original_forward


class FusedGateUpSiluVariant(KernelVariant):
    """Triton skinny GEMM + fused gate_up_proj/SiluAndMul in one kernel pass.

    No patching needed — Qwen2MLP.forward() already calls
    triton_fused_gate_up_silu for B<=32 bfloat16 unquantized weights.
    """

    name = "fused_gate_up_silu"
    description = "Triton skinny GEMM + fused gate+SiLU kernel at B=1–32"

    # active() inherits the no-op from KernelVariant


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

REGISTRY: dict[str, KernelVariant] = {
    v.name: v
    for v in [
        StockCuBLASVariant(),
        TritonGemvVariant(),
        TritonSkinnyGemmVariant(),
        FusedGateUpSiluVariant(),
    ]
}
