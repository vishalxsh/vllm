"""
Custom Triton GEMV kernel for batch=1 decode in vLLM.

Target shapes (Qwen2.5-7B, bfloat16):
  [3584, 3584], [18944, 3584], [3584, 18944]

Usage (correctness check):
  ~/vllm-env/bin/python ~/vllm-kernel-opt/kernels/gemv.py
"""

import sys

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
    SPLIT_K: tl.constexpr,
    USE_ATOMIC: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    row_start = pid_m * BLOCK_M
    row_ids   = row_start + tl.arange(0, BLOCK_M)
    row_mask  = row_ids < M

    n_per_split = (N + SPLIT_K - 1) // SPLIT_K
    n_start     = pid_k * n_per_split
    n_end       = tl.minimum(n_start + n_per_split, N)

    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for col_start in range(n_start, n_end, BLOCK_N):
        col_ids  = col_start + tl.arange(0, BLOCK_N)
        col_mask = col_ids < n_end
        x_chunk  = tl.load(x_ptr + col_ids, mask=col_mask, other=0.0).to(tl.float32)
        w_ptrs   = W_ptr + row_ids[:, None] * stride_wm + col_ids[None, :] * stride_wn
        w_block  = tl.load(
            w_ptrs, mask=row_mask[:, None] & col_mask[None, :], other=0.0
        ).to(tl.float32)
        acc += tl.sum(w_block * x_chunk[None, :], axis=1)

    if USE_ATOMIC:
        tl.atomic_add(y_ptr + row_ids, acc.to(y_ptr.dtype.element_ty), mask=row_mask)
    else:
        tl.store(y_ptr + row_ids, acc.to(y_ptr.dtype.element_ty), mask=row_mask)


# Tuned for Qwen2.5-7B on RTX 3090 (idle GPU). Artemis heuristic + BLOCK_N sweep.
_SHAPE_TUNING: dict[tuple[int, int], dict] = {
    (3584, 3584):    dict(BLOCK_M=32, BLOCK_N=512, SPLIT_K=4,  USE_ATOMIC=True),
    (18944, 3584):   dict(BLOCK_M=32, BLOCK_N=128, SPLIT_K=1,  USE_ATOMIC=False),
    (3584, 18944):   dict(BLOCK_M=32, BLOCK_N=512, SPLIT_K=4,  USE_ATOMIC=True),
}


def _default_tuning(M: int, N: int) -> dict:
    if M >= 8192:
        split_k, atomic = 1, False
    elif N >= 16384:
        split_k, atomic = 4, True
    elif N >= 8192:
        split_k, atomic = 4, True
    else:
        split_k, atomic = 4, True
    return dict(BLOCK_M=32, BLOCK_N=128, SPLIT_K=split_k, USE_ATOMIC=atomic)


def triton_gemv(W: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 2:
        assert x.shape[0] == 1
        x = x.squeeze(0)
    assert W.ndim == 2 and x.ndim == 1
    assert W.shape[1] == x.shape[0]
    assert W.is_cuda and x.is_cuda

    M, N = W.shape
    cfg = _SHAPE_TUNING.get((M, N), _default_tuning(M, N))
    BLOCK_M = cfg["BLOCK_M"]
    BLOCK_N = cfg["BLOCK_N"]
    SPLIT_K = cfg["SPLIT_K"]
    USE_ATOMIC = cfg["USE_ATOMIC"]

    if SPLIT_K == 1 and not USE_ATOMIC:
        y = torch.empty(M, dtype=W.dtype, device=W.device)
        grid = (triton.cdiv(M, BLOCK_M), 1)
        gemv_kernel[grid](
            W, x, y,
            M, N,
            W.stride(0), W.stride(1),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            SPLIT_K=1,
            USE_ATOMIC=False,
            num_warps=4,
            num_stages=4,
        )
        return y

    y = torch.zeros(M, dtype=W.dtype, device=W.device)
    grid = (triton.cdiv(M, BLOCK_M), SPLIT_K)
    gemv_kernel[grid](
        W, x, y,
        M, N,
        W.stride(0), W.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        SPLIT_K=SPLIT_K,
        USE_ATOMIC=USE_ATOMIC,
        num_warps=4,
        num_stages=4,
    )
    return y


if __name__ == "__main__":
    torch.manual_seed(0)
    shapes = [(3584, 3584), (18944, 3584), (3584, 18944)]
    all_passed = True

    for M, N in shapes:
        W = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
        x = torch.randn(N, dtype=torch.bfloat16, device="cuda")
        y_ref = (W.float() @ x.float()).bfloat16()
        y_tri = triton_gemv(W, x)

        max_diff = (y_ref.float() - y_tri.float()).abs().max().item()
        status = "PASSED" if max_diff < 5.0 else "FAILED"
        if status == "FAILED":
            all_passed = False
        print(f"  [{M:6d},{N:6d}]  max_abs={max_diff:.4f}  {status}")

    print()
    print("Overall:", "PASSED" if all_passed else "FAILED")
    sys.exit(0 if all_passed else 1)
