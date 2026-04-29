"""
Custom Triton GEMV kernel for batch=1 decode in vLLM.
Target shapes (Qwen2.5-1.5B):
  [1536, 1536], [8960, 1536], [1536, 8960]

Usage (correctness check):
  ~/vllm-env/bin/python ~/vllm-kernel-opt/kernels/gemv.py
"""

import torch
import triton
import triton.language as tl


@triton.jit
def gemv_kernel(
    W_ptr, x_ptr, y_ptr,
    M, N,
    stride_wm, stride_wn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row_start = tl.program_id(0) * BLOCK_M
    row_ids   = row_start + tl.arange(0, BLOCK_M)
    row_mask  = row_ids < M

    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for col_start in range(0, N, BLOCK_N):
        col_ids  = col_start + tl.arange(0, BLOCK_N)
        col_mask = col_ids < N

        x_chunk = tl.load(x_ptr + col_ids, mask=col_mask, other=0.0).to(tl.float32)

        w_ptrs  = W_ptr + row_ids[:, None] * stride_wm + col_ids[None, :] * stride_wn
        w_block = tl.load(w_ptrs, mask=row_mask[:, None] & col_mask[None, :], other=0.0).to(tl.float32)

        acc += tl.sum(w_block * x_chunk[None, :], axis=1)

    tl.store(y_ptr + row_ids, acc.to(tl.float16), mask=row_mask)


def triton_gemv(W: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    assert W.ndim == 2 and x.ndim == 1
    assert W.shape[1] == x.shape[0]
    assert W.is_cuda and x.is_cuda

    M, N    = W.shape
    y       = torch.empty(M, dtype=torch.float16, device=W.device)
    BLOCK_M = 16
    BLOCK_N = 256
    grid    = (triton.cdiv(M, BLOCK_M),)

    gemv_kernel[grid](
        W, x, y,
        M, N,
        W.stride(0), W.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
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
