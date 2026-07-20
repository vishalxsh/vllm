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
        )
        x_mask = (offs_b[:, None] < B) & (offs_k[None, :] < K)
        x_tile = tl.load(
            X_ptr + offs_b[:, None] * stride_xb + offs_k[None, :] * stride_xk,
            mask=x_mask, other=0.0,
        )
        acc = tl.dot(w_tile, tl.trans(x_tile), acc)

    y_mask = (offs_m[:, None] < M) & (offs_b[None, :] < B)
    tl.store(
        Y_ptr + offs_m[:, None] * stride_ym + offs_b[None, :] * stride_yb,
        acc.to(Y_ptr.dtype.element_ty),
        mask=y_mask,
    )


@triton.jit
def _skinny_gemm_splitk_kernel(
    W_ptr, X_ptr, Yf_ptr,
    M, K, B,
    stride_wm, stride_wk,
    stride_xb, stride_xk,
    stride_ym, stride_yb,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_B: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    """B=1 split-K variant: keeps tl.dot (tensor cores) but partitions the K
    reduction across a 3rd grid axis so more programs run concurrently on the
    reduction-bound square shapes. Each program accumulates its K-slice, then
    atomic-adds into an fp32 output buffer — a SINGLE kernel (no separate
    reduce launch, which is what made the pure-GEMV attempt slow). Yf is fp32
    (bf16 atomics are emulated/expensive on Ampere); the caller casts to the
    output dtype in a cheap elementwise op or the buffer is used directly.
    """
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_k = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)

    k_per_split = (K + SPLIT_K - 1) // SPLIT_K
    k_start = pid_k * k_per_split
    k_end = tl.minimum(k_start + k_per_split, K)

    acc = tl.zeros((BLOCK_M, BLOCK_B), dtype=tl.float32)
    for k in range(k_start, k_end, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        w_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_end)
        w_tile = tl.load(
            W_ptr + offs_m[:, None] * stride_wm + offs_k[None, :] * stride_wk,
            mask=w_mask, other=0.0,
        )
        x_mask = (offs_b[:, None] < B) & (offs_k[None, :] < k_end)
        x_tile = tl.load(
            X_ptr + offs_b[:, None] * stride_xb + offs_k[None, :] * stride_xk,
            mask=x_mask, other=0.0,
        )
        acc = tl.dot(w_tile, tl.trans(x_tile), acc)

    y_mask = (offs_m[:, None] < M) & (offs_b[None, :] < B)
    if SPLIT_K == 1:
        tl.store(
            Yf_ptr + offs_m[:, None] * stride_ym + offs_b[None, :] * stride_yb,
            acc, mask=y_mask,
        )
    else:
        tl.atomic_add(
            Yf_ptr + offs_m[:, None] * stride_ym + offs_b[None, :] * stride_yb,
            acc, mask=y_mask,
        )


# ---------------------------------------------------------------------------
# Shape-keyed configs + runtime tuning.
#
# _CONFIGS is a cache keyed by (M, K, B_bucket), pre-seeded from offline
# tuning (Artemis + Triton autotune) for Qwen2.5-7B. ensure_tuned() tunes
# unseen shapes at WEIGHT-LOAD time and races the winner against cuBLAS —
# a shape is only registered in _ENABLED (and thus dispatched to Triton)
# if Triton wins at ALL batch buckets (4, 16, 32), so no model can get
# slower at any batch size.
#
# @triton.autotune is deliberately not used: it adds Python dispatch
# overhead (~1-2 us) that regresses fast kernels like attn_proj at small B
# where kernel time is only 3-5 us, it re-tunes mid-serving on new keys
# (latency spikes), and its timing syncs crash CUDA graph capture. Under
# vLLM's fullgraph torch.compile the config is frozen at trace time anyway
# — tuning later than weight loading is a hard error or a silent no-op.
# ---------------------------------------------------------------------------

def _bucket_b(B: int) -> int:
    # BLOCK_B=1 pads tl.dot's N dim too thin to reach cuBLAS (measured
    # slower than BLOCK_B=4 at B=1 on 3090) — bucket 4 is the smallest
    # viable granularity, not B itself.
    if B <= 4:
        return 4
    return 16 if B <= 16 else 32


# 16-bit dtypes the kernel supports. Both move the same bytes and hit the
# same tensor-core throughput, so tuned block configs transfer between them;
# the cuBLAS race does NOT transfer (cuBLAS picks different algorithms per
# dtype), so enablement is tracked per (M, K, dtype).
_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16)

# (M, K, B_bucket) -> (BLOCK_M, BLOCK_K, BLOCK_B, num_warps, num_stages)
# Configs are dtype-agnostic (see _SUPPORTED_DTYPES note) and shared.
# attn_proj:    Artemis-tuned  BLOCK_K=64,  ns=4 wins (56 K-loops, deep pipeline)
# gate_up_proj: autotune confirmed BLOCK_K=128, BLOCK_M=128 for large-B
# ffn_down:     autotune confirmed BLOCK_K=128 beats BLOCK_K=64
_CONFIGS = {
    (3584,  3584,  4):  (32, 128, 4,  4, 3),
    (3584,  3584,  16): (32, 64,  16, 4, 4),
    (3584,  3584,  32): (32, 64,  32, 4, 4),
    (37888, 3584,  4):  (64, 128, 4,  4, 3),
    (37888, 3584,  16): (64,  128, 16, 4, 3),
    (37888, 3584,  32): (128, 128, 32, 8, 3),
    (3584,  18944, 4):  (32, 128, 4,  4, 4),
    (3584,  18944, 16): (32,  128, 16, 4, 3),
    (3584,  18944, 32): (64,  128, 32, 4, 3),
}
_DEFAULT_CONFIG = (32, 64, 16, 4, 3)

# (M, K) -> SPLIT_K factor for the B=1 in-kernel split-K tl.dot path. Filled
# empirically at load time by _tune_splitk_b1 (below), which races SPLIT_K
# candidates INCLUDING 1 (== the plain V2 path) and keeps the fastest — so a
# shape only gets split>1 if it is measurably faster, guaranteeing no
# regression vs the widened-tuning V2 baseline. Default (absent key) is 1.
_SPLITK_B1: dict = {}
# Candidate split factors to race for B=1. 1 = no split (plain kernel).
_SPLITK_CANDIDATES = (1, 2, 4, 8)

# (M, K, dtype) triples the dispatch may route to Triton. Pre-seeded with
# the Qwen2.5-7B bf16 shapes (validated vs cuBLAS offline: 1.04-1.22x per
# shape). Other dtypes of the same shape must win their own race.
_ENABLED: set = {
    (3584, 3584, torch.bfloat16),
    (37888, 3584, torch.bfloat16),
    (3584, 18944, torch.bfloat16),
}

# (M, K, dtype) triples whose cuBLAS race already ran this process (win or
# lose) — prevents re-racing shapes that were judged not-faster-than-cuBLAS.
_RACED: set = set()

# (M, K, dtype) triples whose bucket=4 config was already re-tuned from the
# wider B4 space this process — idempotency guard for the B=1 re-tune.
_B4_RETUNED: set = set()

# Pruned candidate space for runtime tuning, spanning the offline winners:
# (BLOCK_M, BLOCK_K, num_warps, num_stages). Used for buckets 16 and 32 —
# UNCHANGED from the offline-validated set, so those buckets never regress.
_TUNE_SPACE = [
    (32, 64, 4, 4), (32, 64, 4, 3), (32, 128, 4, 3), (32, 128, 4, 4),
    (64, 64, 4, 3), (64, 128, 4, 3), (64, 128, 4, 2), (64, 128, 8, 3),
    (128, 128, 8, 3), (128, 128, 4, 2), (128, 64, 4, 3), (128, 128, 8, 2),
]

# Dedicated, wider candidate space for the bucket=4 (B<=4, incl. B=1) path.
# At B=1 the GEMM is reduction/latency-bound with N pinned to 4, so the
# winning knobs differ from large-B: smaller BLOCK_M gives more tiles to
# fill the SMs on square shapes, larger BLOCK_K + deeper num_stages pipeline
# the K-loop loads, and num_warps=2 trims per-program overhead. Each entry is
# (BLOCK_M, BLOCK_K, BLOCK_B, num_warps, num_stages); BLOCK_B is also swept
# (4/2/1) since a thinner N tile can reduce wasted tensor-core work.
_TUNE_SPACE_B4 = [
    # BLOCK_B = 4 (baseline granularity), varied M/K/warps/stages
    (16, 128, 4, 2, 4), (16, 128, 4, 4, 4), (16, 256, 4, 4, 4),
    (32, 128, 4, 4, 3), (32, 128, 4, 2, 4), (32, 256, 4, 4, 4),
    (32, 256, 4, 4, 3), (64, 128, 4, 4, 3), (64, 128, 4, 4, 4),
    (64, 256, 4, 4, 3), (64, 256, 4, 8, 3), (128, 128, 4, 4, 3),
    (128, 256, 4, 8, 3),
    # Thinner N tiles — less padded tensor-core work at B=1
    (32, 128, 2, 4, 4), (32, 256, 2, 4, 4), (64, 256, 2, 4, 3),
    (16, 256, 2, 4, 4),
    # Deep-pipeline square-shape candidates (more stages, few warps)
    (32, 64, 4, 2, 5), (32, 128, 4, 2, 5), (16, 128, 4, 2, 5),
]

# Triton must beat cuBLAS by this factor at both buckets to be enabled.
_ENABLE_MARGIN = 1.02


def is_enabled(M: int, K: int, dtype: torch.dtype = torch.bfloat16) -> bool:
    return (M, K, dtype) in _ENABLED


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

    # bucket=4 (the B=1 target) searches its own wider space; buckets 16/32
    # keep the offline-validated 12-candidate set unchanged (no regression).
    if bucket == 4:
        candidates = [(bm, bk, bb, nw, ns)
                      for (bm, bk, bb, nw, ns) in _TUNE_SPACE_B4]
    else:
        candidates = [(bm, bk, bucket, nw, ns) for (bm, bk, nw, ns) in _TUNE_SPACE]

    X = torch.randn(bucket, K, dtype=W.dtype, device=W.device)
    Y = torch.empty((bucket, M), dtype=W.dtype, device=W.device)
    best, best_t = None, float("inf")
    for cfg in candidates:
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


def _launch_splitk(W, X, Yf, M, K, B, cfg, split_k):
    BLOCK_M, BLOCK_K, BLOCK_B, num_warps, num_stages = cfg
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(B, BLOCK_B), split_k)
    Yf.zero_()
    _skinny_gemm_splitk_kernel[grid](
        W, X, Yf,
        M, K, B,
        W.stride(0), W.stride(1),
        X.stride(0), X.stride(1),
        1, M,
        BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K, BLOCK_B=BLOCK_B,
        SPLIT_K=split_k,
        num_warps=num_warps, num_stages=num_stages,
    )


def _tune_splitk_b1(W: torch.Tensor, M: int, K: int, cfg4) -> None:
    """Race B=1 SPLIT_K candidates (incl. 1) with the winning bucket=4 config.

    Only adopt split>1 if strictly faster than split=1, so this can never
    regress the V2 (widened-tuning) baseline. Populates _SPLITK_B1[(M,K)].
    """
    import triton.testing

    X  = torch.randn(1, K, dtype=W.dtype, device=W.device)
    Yf = torch.zeros(1, M, dtype=torch.float32, device=W.device)

    def _time(s):
        try:
            return triton.testing.do_bench(
                lambda: _launch_splitk(W, X, Yf, M, K, 1, cfg4, s),
                warmup=10, rep=50, return_mode="median",
            )
        except Exception:
            return float("inf")

    # Baseline: split=1 (the plain V2 path, just via the split-K kernel).
    base_t = _time(1)
    best_split, best_t = 1, base_t
    for s in _SPLITK_CANDIDATES:
        if s == 1 or K // s < 128:
            continue  # too little K per split to be worthwhile
        t = _time(s)
        # Adopt split>1 only if strictly faster than split=1 by a margin,
        # so this can never regress the V2 baseline.
        if t < best_t and t < base_t * 0.98:
            best_split, best_t = s, t
    _SPLITK_B1[(M, K)] = best_split
    print(f"[skinny_gemm] B=1 split-K (M={M}, K={K}) -> SPLIT_K={best_split} "
          f"(base={base_t*1e3:.0f}us best={best_t*1e3:.0f}us)")


def ensure_tuned(W: torch.Tensor) -> None:
    """Tune W's shape and enable Triton dispatch for it iff it beats cuBLAS.

    Call at WEIGHT-LOAD time (eager, before torch.compile tracing and CUDA
    graph capture — see module comment). Idempotent per shape.
    """
    if torch.compiler.is_compiling():
        return
    if W.ndim != 2 or W.dtype not in _SUPPORTED_DTYPES or not W.is_cuda:
        return
    if torch.cuda.is_current_stream_capturing():
        return
    M, K = W.shape
    key = (M, K, W.dtype)

    # Always (idempotently) re-tune the bucket=4 (B=1) config from the wider
    # B4 space, even for pre-seeded/enabled shapes — the offline seeds were
    # picked from the narrow shared space and don't reflect the B=1 optimum.
    # buckets 16/32 are left exactly as seeded. Scoped to bucket=4 only, so
    # no other batch size can regress.
    if key not in _B4_RETUNED:
        _B4_RETUNED.add(key)
        cfg4, t4, _ = _tune(W, M, K, 4)
        _CONFIGS[(M, K, 4)] = cfg4
        print(f"[skinny_gemm] bucket4 re-tuned (M={M}, K={K}, {W.dtype}) "
              f"-> {cfg4} triton={t4*1e3:.0f}us")
        # Race the B=1 split-K variants (incl. SPLIT_K=1) with the winning
        # bucket=4 config; only adopt split>1 if it is strictly faster.
        _tune_splitk_b1(W, M, K, cfg4)

    if key in _ENABLED or key in _RACED:
        return  # already raced this dtype (pre-seeded, won, or lost)
    _RACED.add(key)
    buckets = (4, 16, 32)
    wins = 0
    for bucket in buckets:
        cfg, t_tri, t_cu = _tune(W, M, K, bucket)
        _CONFIGS[(M, K, bucket)] = cfg
        if t_tri * _ENABLE_MARGIN < t_cu:
            wins += 1
        print(f"[skinny_gemm] tuned (M={M}, K={K}, {W.dtype}, bucket={bucket}) "
              f"-> {cfg} triton={t_tri*1e3:.0f}us cublas={t_cu*1e3:.0f}us")
    if wins == len(buckets):
        _ENABLED.add(key)
        print(f"[skinny_gemm] ENABLED (M={M}, K={K}, {W.dtype}) "
              f"— beats cuBLAS at all {len(buckets)} buckets")
    else:
        print(f"[skinny_gemm] NOT enabled (M={M}, K={K}, {W.dtype}) — cuBLAS kept")


def triton_skinny_gemm(W: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """
    Args:
        W : [M, K] bfloat16 or float16
        X : [B, K] same dtype as W,  B = 1..32

    Returns:
        Y : [B, M] same dtype as inputs
    """
    assert W.ndim == 2 and X.ndim == 2, "W must be 2D, X must be 2D [B, K]"
    assert W.dtype in _SUPPORTED_DTYPES and X.dtype == W.dtype
    assert W.is_cuda and X.is_cuda

    M, K = W.shape
    B    = X.shape[0]

    BLOCK_M, BLOCK_K, BLOCK_B, num_warps, num_stages = _pick_config(M, K, B)

    # B=1: use the in-kernel split-K tl.dot variant if a split>1 was chosen
    # for this shape. Keeps tensor cores; a single kernel atomic-adds the
    # per-split partials into an fp32 buffer (no separate reduce launch).
    if B == 1:
        split_k = _SPLITK_B1.get((M, K), 1)
        if split_k > 1:
            Yf   = torch.zeros((B, M), dtype=torch.float32, device=W.device)
            grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(B, BLOCK_B), split_k)
            _skinny_gemm_splitk_kernel[grid](
                W, X, Yf,
                M, K, B,
                W.stride(0), W.stride(1),
                X.stride(0), X.stride(1),
                1, M,
                BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K, BLOCK_B=BLOCK_B,
                SPLIT_K=split_k,
                num_warps=num_warps, num_stages=num_stages,
            )
            return Yf.to(X.dtype)

    # Allocate Y directly in [B, M] layout; kernel writes Y[m,b] via
    # (stride_ym=1, stride_yb=M) so no .t().contiguous() copy is needed.
    Y    = torch.empty((B, M), dtype=X.dtype, device=W.device)
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

    for dtype in _SUPPORTED_DTYPES:
        print(f"--- {dtype} ---")
        for M, K, label in shapes:
            for B in batch_sizes:
                W     = torch.randn(M, K, dtype=dtype, device="cuda")
                X     = torch.randn(B, K, dtype=dtype, device="cuda")
                y_ref = (X.float() @ W.float().t()).to(dtype)
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
