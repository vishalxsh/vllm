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
# Shape-keyed configs + runtime tuning.
#
# _CONFIGS is a cache keyed by (M, K, B_bucket), pre-seeded from offline
# tuning (Artemis + Triton autotune) for Qwen2.5-7B. ensure_tuned() tunes
# unseen shapes at WEIGHT-LOAD time and races the winner against cuBLAS —
# a shape is only registered in _ENABLED (and thus dispatched to Triton)
# if Triton wins at BOTH batch buckets, so no model can get slower.
#
# @triton.autotune is deliberately not used: it adds Python dispatch
# overhead (~1-2 us) that regresses fast kernels like attn_proj at small B
# where kernel time is only 3-5 us, it re-tunes mid-serving on new keys
# (latency spikes), and its timing syncs crash CUDA graph capture. Under
# vLLM's fullgraph torch.compile the config is frozen at trace time anyway
# — tuning later than weight loading is a hard error or a silent no-op.
# ---------------------------------------------------------------------------

def _bucket_b(B: int) -> int:
    return 16 if B <= 16 else 32


# (M, K, B_bucket) -> (BLOCK_M, BLOCK_K, BLOCK_B, num_warps, num_stages)
# attn_proj:    Artemis-tuned  BLOCK_K=64,  ns=4 wins (56 K-loops, deep pipeline)
# gate_up_proj: autotune confirmed BLOCK_K=128, BLOCK_M=128 for large-B
# ffn_down:     autotune confirmed BLOCK_K=128 beats BLOCK_K=64
_CONFIGS = {
    (3584,  3584,  16): (32, 64,  16, 4, 4),
    (3584,  3584,  32): (32, 64,  32, 4, 4),
    (37888, 3584,  16): (64,  128, 16, 4, 3),
    (37888, 3584,  32): (128, 128, 32, 8, 3),
    (3584,  18944, 16): (32,  128, 16, 4, 3),
    (3584,  18944, 32): (64,  128, 32, 4, 3),
}
_DEFAULT_CONFIG = (32, 64, 16, 4, 3)

# Shapes the dispatch may route to Triton. Pre-seeded with the Qwen2.5-7B
# shapes (validated vs cuBLAS offline: 1.04-1.22x per shape).
_ENABLED: set = {
    (3584, 3584),
    (37888, 3584),
    (3584, 18944),
}

# Pruned candidate space for runtime tuning, spanning the offline winners:
# (BLOCK_M, BLOCK_K, num_warps, num_stages).
_TUNE_SPACE = [
    (32, 64, 4, 4), (32, 64, 4, 3), (32, 128, 4, 3), (32, 128, 4, 4),
    (64, 64, 4, 3), (64, 128, 4, 3), (64, 128, 4, 2), (64, 128, 8, 3),
    (128, 128, 8, 3), (128, 128, 4, 2), (128, 64, 4, 3), (128, 128, 8, 2),
]

# Triton must beat cuBLAS by this factor at both buckets to be enabled.
_ENABLE_MARGIN = 1.02


def is_enabled(M: int, K: int) -> bool:
    return (M, K) in _ENABLED


def _pick_config(M: int, K: int, B: int):
    bb  = _bucket_b(B)
    cfg = _CONFIGS.get((M, K, bb))
    if cfg is not None:
        return cfg
    bm, bk, _, nw, ns = _DEFAULT_CONFIG
    return (bm, bk, bb, nw, ns)


def _launch(W, X, Y, M, K, B, cfg):
    BLOCK_M, BLOCK_K, BLOCK_B, num_warps, num_stages = cfg
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


def _tune(W: torch.Tensor, M: int, K: int, bucket: int):
    """Return (best_cfg, best_ms, cublas_ms) for this shape/bucket."""
    import triton.testing

    X = torch.randn(bucket, K, dtype=torch.bfloat16, device=W.device)
    Y = torch.empty((bucket, M), dtype=torch.bfloat16, device=W.device)
    best, best_t = None, float("inf")
    for bm, bk, nw, ns in _TUNE_SPACE:
        cfg = (bm, bk, bucket, nw, ns)
        try:
            t = triton.testing.do_bench(
                lambda: _launch(W, X, Y, M, K, bucket, cfg),
                warmup=10, rep=50, return_mode="median",
            )
        except Exception:
            continue  # OutOfResources etc.
        if t < best_t:
            best, best_t = cfg, t
    t_cublas = triton.testing.do_bench(
        lambda: torch.nn.functional.linear(X, W),
        warmup=10, rep=50, return_mode="median",
    )
    if best is None:
        bm, bk, _, nw, ns = _DEFAULT_CONFIG
        best = (bm, bk, bucket, nw, ns)
    return best, best_t, t_cublas


def ensure_tuned(W: torch.Tensor) -> None:
    """Tune W's shape and enable Triton dispatch for it iff it beats cuBLAS.

    Call at WEIGHT-LOAD time (eager, before torch.compile tracing and CUDA
    graph capture — see module comment). Idempotent per shape.
    """
    if torch.compiler.is_compiling():
        return
    if W.ndim != 2 or W.dtype != torch.bfloat16 or not W.is_cuda:
        return
    if torch.cuda.is_current_stream_capturing():
        return
    M, K = W.shape
    if (M, K) in _ENABLED:
        return
    if all((M, K, b) in _CONFIGS for b in (16, 32)):
        return  # already tuned and judged not-faster-than-cuBLAS
    wins = 0
    for bucket in (16, 32):
        cfg, t_tri, t_cu = _tune(W, M, K, bucket)
        _CONFIGS[(M, K, bucket)] = cfg
        if t_tri * _ENABLE_MARGIN < t_cu:
            wins += 1
        print(f"[skinny_gemm] tuned (M={M}, K={K}, bucket={bucket}) -> {cfg} "
              f"triton={t_tri*1e3:.0f}us cublas={t_cu*1e3:.0f}us")
    if wins == 2:
        _ENABLED.add((M, K))
        print(f"[skinny_gemm] ENABLED (M={M}, K={K}) — beats cuBLAS at both buckets")
    else:
        print(f"[skinny_gemm] NOT enabled (M={M}, K={K}) — cuBLAS kept")


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
        (3584,  3584,  "attn_proj   "),
        (37888, 3584,  "gate_up_proj"),
        (3584,  18944, "ffn_down    "),
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
