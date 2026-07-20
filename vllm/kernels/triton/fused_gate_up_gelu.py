"""Fused gate_up_proj GEMM + GeGLU (GELU-and-mul) Triton kernel.

GeGLU sibling of fused_gate_up_silu — same two-accumulator structure, the
activation applied in registers is tanh-approximated GELU (matching HF
Gemma / Liger's geglu):

  gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
  out     = gelu(gate) * up

  W : [2*M_half, K]  fused gate+up weight (gate rows first)
  X : [B, K]
  Y : [B, M_half]

Config cache and tuning are shared with the SiLU variant (identical memory
and tensor-core behaviour — only the epilogue differs), so ensure_tuned from
fused_gate_up_silu covers this kernel too.
"""

import sys
import torch
import triton
import triton.language as tl

from vllm.kernels.triton.fused_gate_up_silu import (
    _SUPPORTED_DTYPES,
    _pick_config,
    ensure_tuned,  # noqa: F401 — re-exported: same configs tune both variants
)


@triton.jit
def _fused_gate_up_gelu_kernel(
    W_ptr, X_ptr, Y_ptr,
    M_half, K, B,
    stride_wm, stride_wk,
    stride_xb, stride_xk,
    stride_ym, stride_yb,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)

    acc_gate = tl.zeros((BLOCK_M, BLOCK_B), dtype=tl.float32)
    acc_up   = tl.zeros((BLOCK_M, BLOCK_B), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        gate_mask = (offs_m[:, None] < M_half) & (offs_k[None, :] < K)
        w_gate = tl.load(
            W_ptr + offs_m[:, None] * stride_wm + offs_k[None, :] * stride_wk,
            mask=gate_mask, other=0.0,
        )
        w_up = tl.load(
            W_ptr + (offs_m + M_half)[:, None] * stride_wm + offs_k[None, :] * stride_wk,
            mask=gate_mask, other=0.0,
        )
        x_mask = (offs_b[:, None] < B) & (offs_k[None, :] < K)
        x_tile = tl.load(
            X_ptr + offs_b[:, None] * stride_xb + offs_k[None, :] * stride_xk,
            mask=x_mask, other=0.0,
        )
        acc_gate = tl.dot(w_gate, tl.trans(x_tile), acc_gate)
        acc_up   = tl.dot(w_up,   tl.trans(x_tile), acc_up)

    # tanh-approx GELU in fp32 registers; tanh(z) = 2*sigmoid(2z) - 1
    z = 0.7978845608028654 * (acc_gate + 0.044715 * acc_gate * acc_gate * acc_gate)
    gelu_gate = 0.5 * acc_gate * (2.0 * tl.sigmoid(2.0 * z))
    output = gelu_gate * acc_up

    y_mask = (offs_m[:, None] < M_half) & (offs_b[None, :] < B)
    tl.store(
        Y_ptr + offs_m[:, None] * stride_ym + offs_b[None, :] * stride_yb,
        output.to(Y_ptr.dtype.element_ty),
        mask=y_mask,
    )


def triton_fused_gate_up_gelu(W: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """
    Args:
        W : [2*M_half, K] bfloat16 or float16 — fused gate+up weight
        X : [B, K] same dtype as W,  B = 1..32

    Returns:
        Y : [B, M_half] — gelu_tanh(gate) * up
    """
    assert W.ndim == 2 and X.ndim == 2
    assert W.dtype in _SUPPORTED_DTYPES and X.dtype == W.dtype
    assert W.is_cuda and X.is_cuda
    assert W.shape[0] % 2 == 0, "W rows must be even (gate+up stacked)"

    M2, K  = W.shape
    M_half = M2 // 2
    B      = X.shape[0]

    BLOCK_M, BLOCK_K, BLOCK_B, num_warps, num_stages = _pick_config(W, M_half, K, B)
    Y = torch.empty((B, M_half), dtype=X.dtype, device=W.device)
    grid = (triton.cdiv(M_half, BLOCK_M), triton.cdiv(B, BLOCK_B))
    _fused_gate_up_gelu_kernel[grid](
        W, X, Y,
        M_half, K, B,
        W.stride(0), W.stride(1),
        X.stride(0), X.stride(1),
        1, M_half,
        BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K, BLOCK_B=BLOCK_B,
        num_warps=num_warps, num_stages=num_stages,
    )
    return Y


if __name__ == "__main__":
    torch.manual_seed(0)
    all_passed = True

    # Gemma-2-9B-ish and Qwen shapes
    for M_half, K in ((14336, 3584), (18944, 3584)):
        for dtype in _SUPPORTED_DTYPES:
            for B in (1, 4, 16, 32):
                W = torch.randn(2 * M_half, K, dtype=dtype, device="cuda")
                X = torch.randn(B, K, dtype=dtype, device="cuda")

                gate_up = X.float() @ W.float().t()
                gate, up = gate_up[:, :M_half], gate_up[:, M_half:]
                ref = torch.nn.functional.gelu(gate, approximate="tanh") * up

                out = triton_fused_gate_up_gelu(W, X)
                max_diff  = (out.float() - ref).abs().max().item()
                out_scale = max(ref.abs().max().item(), 1.0)
                ok = max_diff < out_scale * 0.05
                all_passed &= ok
                print(f"  [{M_half:5d},{K}] {str(dtype):15s} B={B:2d}  "
                      f"max_abs={max_diff:.4f}  {'PASSED' if ok else 'FAILED'}")

    print("\nOverall:", "PASSED" if all_passed else "FAILED")
    sys.exit(0 if all_passed else 1)
