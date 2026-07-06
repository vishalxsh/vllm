"""
Fused gate_up_proj GEMM + SiluAndMul Triton kernel for vLLM decode.

Standard two-kernel path:
  gate_up = X @ W.T                 -> [B, 2*M_half]  (write to HBM)
  out     = silu(gate_up[:, :M_half]) * gate_up[:, M_half:]   (read from HBM)

Fused single-kernel path:
  Compute gate and up tiles simultaneously in registers, apply SiLU,
  write only the final [B, M_half] result. The intermediate [B, 2*M_half]
  tensor never touches HBM.

Target: Qwen2.5-7B bfloat16 on RTX 3090, batch sizes 1-32.
  W : [2*M_half, K]  = [37888, 3584]  fused gate+up weight
  X : [B, K]         = [B, 3584]      input vectors
  Y : [B, M_half]    = [B, 18944]     silu(gate) * up output
"""

import sys
import torch
import triton
import triton.language as tl


@triton.jit
def _fused_gate_up_silu_kernel(
    W_ptr, X_ptr, Y_ptr,
    M_half, K, B,
    stride_wm, stride_wk,
    stride_xb, stride_xk,
    stride_ym, stride_yb,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    pid_m = tl.program_id(0)   # indexes over M_half output rows
    pid_b = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)

    acc_gate = tl.zeros((BLOCK_M, BLOCK_B), dtype=tl.float32)
    acc_up   = tl.zeros((BLOCK_M, BLOCK_B), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)

        # Gate weight tile: W[offs_m, offs_k]
        gate_mask = (offs_m[:, None] < M_half) & (offs_k[None, :] < K)
        w_gate = tl.load(
            W_ptr + offs_m[:, None] * stride_wm + offs_k[None, :] * stride_wk,
            mask=gate_mask, other=0.0,
        ).to(tl.bfloat16)

        # Up weight tile: W[offs_m + M_half, offs_k]
        w_up = tl.load(
            W_ptr + (offs_m + M_half)[:, None] * stride_wm + offs_k[None, :] * stride_wk,
            mask=gate_mask,  # same shape bounds as gate
            other=0.0,
        ).to(tl.bfloat16)

        # Input tile: X[offs_b, offs_k]
        x_mask = (offs_b[:, None] < B) & (offs_k[None, :] < K)
        x_tile = tl.load(
            X_ptr + offs_b[:, None] * stride_xb + offs_k[None, :] * stride_xk,
            mask=x_mask, other=0.0,
        ).to(tl.bfloat16)

        # Two GEMMs sharing the same X tile
        acc_gate = tl.dot(w_gate, tl.trans(x_tile), acc_gate)
        acc_up   = tl.dot(w_up,   tl.trans(x_tile), acc_up)

    # SiLU(gate) * up  — all in float32 registers, no HBM round-trip
    output = acc_gate * tl.sigmoid(acc_gate) * acc_up

    y_mask = (offs_m[:, None] < M_half) & (offs_b[None, :] < B)
    tl.store(
        Y_ptr + offs_m[:, None] * stride_ym + offs_b[None, :] * stride_yb,
        output.to(tl.bfloat16),
        mask=y_mask,
    )


# ---------------------------------------------------------------------------
# Configs  (M_half, K, B_bucket) -> (BLOCK_M, BLOCK_K, BLOCK_B, nw, ns)
# Autotuned offline on RTX 3090 (sweep over BLOCK_M/K/B x warps x stages,
# correctness-gated, interleaved refinement): BLOCK_K=128 wins despite the
# double-accumulator register pressure, matching the plain skinny GEMM.
# ---------------------------------------------------------------------------

def _bucket_b(B: int) -> int:
    return 16 if B <= 16 else 32


_CONFIGS = {
    (18944, 3584, 16): (64, 128, 16, 4, 3),
    (18944, 3584, 32): (64, 128, 32, 4, 3),
}
_DEFAULT_CONFIG = (32, 64, 16, 4, 2)


def _pick_config(M_half: int, K: int, B: int):
    cfg = _CONFIGS.get((M_half, K, _bucket_b(B)))
    if cfg is not None:
        return cfg
    bm, bk, _, nw, ns = _DEFAULT_CONFIG
    return (bm, bk, _bucket_b(B), nw, ns)


def triton_fused_gate_up_silu(W: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """
    Args:
        W : [2*M_half, K] bfloat16  — fused gate+up projection weight
        X : [B, K] bfloat16,  B = 1..32

    Returns:
        Y : [B, M_half] bfloat16  — silu(gate) * up
    """
    assert W.ndim == 2 and X.ndim == 2
    assert W.dtype == torch.bfloat16 and X.dtype == torch.bfloat16
    assert W.is_cuda and X.is_cuda
    assert W.shape[0] % 2 == 0, "W rows must be even (gate+up stacked)"

    M2, K  = W.shape
    M_half = M2 // 2
    B      = X.shape[0]

    BLOCK_M, BLOCK_K, BLOCK_B, num_warps, num_stages = _pick_config(M_half, K, B)

    Y    = torch.empty((B, M_half), dtype=torch.bfloat16, device=W.device)
    grid = (triton.cdiv(M_half, BLOCK_M), triton.cdiv(B, BLOCK_B))
    _fused_gate_up_silu_kernel[grid](
        W, X, Y,
        M_half, K, B,
        W.stride(0), W.stride(1),
        X.stride(0), X.stride(1),
        1, M_half,  # stride_ym=1, stride_yb=M_half : Y is [B, M_half] contiguous
        BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K, BLOCK_B=BLOCK_B,
        num_warps=num_warps, num_stages=num_stages,
    )
    return Y


# ---------------------------------------------------------------------------
# Correctness + speed comparison vs unfused path
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import math

    torch.manual_seed(0)

    # Qwen2.5-7B gate_up shape
    M_half, K = 18944, 3584
    batch_sizes = [1, 4, 8, 16, 32]
    all_passed  = True

    print("=== Correctness ===")
    for B in batch_sizes:
        W = torch.randn(2 * M_half, K, dtype=torch.bfloat16, device="cuda")
        X = torch.randn(B, K, dtype=torch.bfloat16, device="cuda")

        # Reference: separate GEMM + silu_and_mul
        gate_up = (X.float() @ W.float().t()).bfloat16()         # [B, 2*M_half]
        gate, up = gate_up[:, :M_half], gate_up[:, M_half:]
        ref = (torch.nn.functional.silu(gate.float()) * up.float()).bfloat16()

        # Fused kernel
        out = triton_fused_gate_up_silu(W, X)

        max_diff = (ref.float() - out.float()).abs().max().item()
        passed   = max_diff < 2.0
        if not passed:
            all_passed = False
        print(f"  B={B:2d}  max_abs={max_diff:.4f}  {'PASSED' if passed else 'FAILED'}")

    print()
    print("Overall:", "PASSED" if all_passed else "FAILED")

    print("\n=== Speed: fused vs unfused (us) ===")
    from vllm.kernels.triton.skinny_gemm import triton_skinny_gemm

    for B in batch_sizes:
        W = torch.randn(2 * M_half, K, dtype=torch.bfloat16, device="cuda")
        X = torch.randn(B, K, dtype=torch.bfloat16, device="cuda")

        # Unfused: skinny_gemm → silu_and_mul
        t_unfused = triton.testing.do_bench(
            lambda: torch.nn.functional.silu(
                triton_skinny_gemm(W, X)[:, :M_half]
            ) * triton_skinny_gemm(W, X)[:, M_half:],
            warmup=100, rep=300,
        )

        # Fused
        t_fused = triton.testing.do_bench(
            lambda: triton_fused_gate_up_silu(W, X),
            warmup=100, rep=300,
        )

        speedup = t_unfused / t_fused
        print(f"  B={B:2d}  unfused={t_unfused*1e3:.1f}us  fused={t_fused*1e3:.1f}us  speedup={speedup:.3f}x")

    sys.exit(0 if all_passed else 1)
