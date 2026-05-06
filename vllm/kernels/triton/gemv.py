"""
Triton GEMV kernel for batch=1 decode in vLLM.
Replaces cuBLAS in default_unquantized_gemm when x.shape[-2] == 1.
Target shapes (Qwen2.5-1.5B): [8960,1536], [1536,8960], [1536,1536]

Two compute paths selected per shape by autotune:
  USE_DOT=False  scalar FMA   — good for large M (many blocks)
  USE_DOT=True   tl.dot TC    — activates Tensor Cores on Ampere
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # ── Scalar configs (USE_DOT=False) ───────────────────────────────────
        triton.Config({'BLOCK_M':  4, 'BLOCK_K':  512, 'USE_DOT': False}, num_warps=2,  num_stages=2),
        triton.Config({'BLOCK_M':  4, 'BLOCK_K': 1024, 'USE_DOT': False}, num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_M':  8, 'BLOCK_K':  512, 'USE_DOT': False}, num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_M':  8, 'BLOCK_K': 1024, 'USE_DOT': False}, num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_K':  512, 'USE_DOT': False}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_K': 1024, 'USE_DOT': False}, num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_K':  512, 'USE_DOT': False}, num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K':  256, 'USE_DOT': False}, num_warps=8,  num_stages=3),
        # ── Tensor Core configs (USE_DOT=True) ───────────────────────────────
        # BLOCK_K >= 16 required; large BLOCK_K reduces loop iterations
        triton.Config({'BLOCK_M': 16, 'BLOCK_K': 256, 'USE_DOT': True},  num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_K': 512, 'USE_DOT': True},  num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_K': 256, 'USE_DOT': True},  num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_K': 512, 'USE_DOT': True},  num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 256, 'USE_DOT': True},  num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 512, 'USE_DOT': True},  num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_M': 128,'BLOCK_K': 256, 'USE_DOT': True},  num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_M': 128,'BLOCK_K': 512, 'USE_DOT': True},  num_warps=8,  num_stages=3),
    ],
    key=['M', 'N'],
)
@triton.jit
def _gemv_kernel(
    W_ptr, x_ptr, y_ptr,
    M, N,
    stride_wm, stride_wn,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    USE_DOT: tl.constexpr,
):
    pid_m    = tl.program_id(0)
    row_ids  = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = row_ids < M

    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for k in range(0, N, BLOCK_K):
        k_ids  = k + tl.arange(0, BLOCK_K)
        k_mask = k_ids < N

        x_chunk = tl.load(x_ptr + k_ids, mask=k_mask, other=0.0)
        w_ptrs  = W_ptr + row_ids[:, None] * stride_wm + k_ids[None, :] * stride_wn
        w_tile  = tl.load(w_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0)

        if USE_DOT:
            # Broadcast x to [BLOCK_K, 16] → tl.dot uses mma.sync (Tensor Cores)
            # All 16 output cols are identical; sum/16 recovers the correct value
            x_bc    = tl.broadcast_to(x_chunk[:, None], (BLOCK_K, 16))
            partial = tl.dot(w_tile, x_bc, out_dtype=tl.float32)
            acc    += tl.sum(partial, axis=1) / 16.0
        else:
            acc += tl.sum(w_tile.to(tl.float32) * x_chunk.to(tl.float32)[None, :], axis=1)

    tl.store(y_ptr + row_ids, acc.to(tl.float16), mask=row_mask)


def triton_gemv(W: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    assert W.ndim == 2 and x.ndim == 1
    assert W.shape[1] == x.shape[0]
    assert W.is_cuda and x.is_cuda

    M, N = W.shape
    y    = torch.empty(M, dtype=torch.float16, device=W.device)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']),)

    _gemv_kernel[grid](
        W, x, y,
        M, N,
        W.stride(0), W.stride(1),
    )
    return y


if __name__ == "__main__":
    torch.manual_seed(0)
    shapes     = [(1536, 1536), (8960, 1536), (1536, 8960)]
    all_passed = True

    for M, N in shapes:
        W = torch.randn(M, N, dtype=torch.float16, device="cuda")
        x = torch.randn(N,    dtype=torch.float16, device="cuda")

        y_ref = (W.float() @ x.float()).half()
        y_tri = triton_gemv(W, x)

        max_diff = (y_ref - y_tri).abs().max().item()
        rel_diff = ((y_ref - y_tri).abs() / (y_ref.abs() + 1e-6)).max().item()
        status   = "PASSED" if max_diff < 0.5 else "FAILED"
        if status == "FAILED":
            all_passed = False
        print(f"  [{M:5d},{N:5d}]  max_abs={max_diff:.4f}  max_rel={rel_diff:.4f}  {status}")

    print()
    print("Overall:", "PASSED" if all_passed else "FAILED")
