"""
Triton skinny GEMM kernel for vLLM decode.

Target: Qwen2.5-7B bfloat16 on RTX 3090, batch sizes 1-32.
  W : [M, K]  weight matrix
  X : [B, K]  input vectors (B = 1..32)
  Y : [B, M]  output

Shapes:
  attn_proj  [3584,  3584]
  ffn_gate   [18944, 3584]
  ffn_down   [3584,  18944]
"""

import sys
import torch
import triton
import triton.language as tl


@triton.jit
def _skinny_gemm_kernel(
    W_ptr, X_ptr, Y_ptr,
    M, K, B,
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

    acc = tl.zeros((BLOCK_M, BLOCK_B), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        w_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        w_tile = tl.load(
            W_ptr + offs_m[:, None] * stride_wm + offs_k[None, :] * stride_wk,
            mask=w_mask, other=0.0,
        ).to(tl.bfloat16)
        x_mask = (offs_b[:, None] < B) & (offs_k[None, :] < K)
        x_tile = tl.load(
            X_ptr + offs_b[:, None] * stride_xb + offs_k[None, :] * stride_xk,
            mask=x_mask, other=0.0,
        ).to(tl.bfloat16)
        acc = tl.dot(w_tile, tl.trans(x_tile), acc)

    y_mask = (offs_m[:, None] < M) & (offs_b[None, :] < B)
    tl.store(
        Y_ptr + offs_m[:, None] * stride_ym + offs_b[None, :] * stride_yb,
        acc.to(tl.bfloat16),
        mask=y_mask,
    )


# ---------------------------------------------------------------------------
# Fixed configs (no autotune). v3: write Y directly as [B,M] — no post-copy.
# ---------------------------------------------------------------------------

def _bucket_b(B: int) -> int:
    return 16 if B <= 16 else 32


_CONFIGS = {
    (3584,  3584,  16): (32, 64,  16, 4, 4),
    (3584,  3584,  32): (32, 64,  32, 4, 4),
    (18944, 3584,  16): (64, 64,  16, 4, 3),
    (18944, 3584,  32): (64, 64,  32, 4, 3),
    (3584,  18944, 16): (64, 128, 16, 4, 3),
    (3584,  18944, 32): (64, 128, 32, 4, 3),
}
_DEFAULT_CONFIG = (32, 64, 16, 4, 3)


def _pick_config(M: int, K: int, B: int):
    bb  = _bucket_b(B)
    cfg = _CONFIGS.get((M, K, bb))
    if cfg is not None:
        return cfg
    bm, bk, _, nw, ns = _DEFAULT_CONFIG
    return (bm, bk, bb, nw, ns)


def triton_skinny_gemm(W: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """
    Args:
        W : [M, K] bfloat16
        X : [B, K] bfloat16,  B = 1..32

    Returns:
        Y : [B, M] bfloat16
    """
    assert W.ndim == 2 and X.ndim == 2, "W must be 2D, X must be 2D [B, K]"
    assert W.dtype == torch.bfloat16 and X.dtype == torch.bfloat16
    assert W.is_cuda and X.is_cuda

    M, K = W.shape
    B    = X.shape[0]

    BLOCK_M, BLOCK_K, BLOCK_B, num_warps, num_stages = _pick_config(M, K, B)

    # Allocate Y directly in [B, M] layout; kernel writes Y[m,b] via
    # (stride_ym=1, stride_yb=M) so no .t().contiguous() copy is needed.
    Y    = torch.empty((B, M), dtype=torch.bfloat16, device=W.device)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(B, BLOCK_B))
    _skinny_gemm_kernel[grid](
        W, X, Y,
        M, K, B,
        W.stride(0), W.stride(1),
        X.stride(0), X.stride(1),
        1, M,  # stride_ym, stride_yb : Y is [B, M] contiguous
        BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K, BLOCK_B=BLOCK_B,
        num_warps=num_warps, num_stages=num_stages,
    )
    return Y


# ---------------------------------------------------------------------------
# Correctness check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)

    shapes = [
        (3584,  3584,  "attn_proj"),
        (18944, 3584,  "ffn_gate "),
        (3584,  18944, "ffn_down "),
    ]
    batch_sizes = [1, 4, 8, 16, 32]
    all_passed  = True

    for M, K, label in shapes:
        for B in batch_sizes:
            W     = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
            X     = torch.randn(B, K, dtype=torch.bfloat16, device="cuda")
            y_ref = (X.float() @ W.float().t()).bfloat16()
            y_tri = triton_skinny_gemm(W, X)

            max_diff = (y_ref.float() - y_tri.float()).abs().max().item()
            threshold = 5.0 if K >= 18944 else 2.0
            passed   = max_diff < threshold
            if not passed:
                all_passed = False
            status = "PASSED" if passed else "FAILED"
            print(f"  {label} [{M:6d},{K:6d}]  B={B:2d}  max_abs={max_diff:.4f}  {status}")

    print()
    print("Overall:", "PASSED" if all_passed else "FAILED")
    sys.exit(0 if all_passed else 1)
