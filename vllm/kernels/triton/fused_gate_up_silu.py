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
        )

        # Up weight tile: W[offs_m + M_half, offs_k]
        w_up = tl.load(
            W_ptr + (offs_m + M_half)[:, None] * stride_wm + offs_k[None, :] * stride_wk,
            mask=gate_mask,  # same shape bounds as gate
            other=0.0,
        )

        # Input tile: X[offs_b, offs_k]
        x_mask = (offs_b[:, None] < B) & (offs_k[None, :] < K)
        x_tile = tl.load(
            X_ptr + offs_b[:, None] * stride_xb + offs_k[None, :] * stride_xk,
            mask=x_mask, other=0.0,
        )

        # Two GEMMs sharing the same X tile
        acc_gate = tl.dot(w_gate, tl.trans(x_tile), acc_gate)
        acc_up   = tl.dot(w_up,   tl.trans(x_tile), acc_up)

    # SiLU(gate) * up  — all in float32 registers, no HBM round-trip
    output = acc_gate * tl.sigmoid(acc_gate) * acc_up

    y_mask = (offs_m[:, None] < M_half) & (offs_b[None, :] < B)
    tl.store(
        Y_ptr + offs_m[:, None] * stride_ym + offs_b[None, :] * stride_yb,
        output.to(Y_ptr.dtype.element_ty),
        mask=y_mask,
    )


# ---------------------------------------------------------------------------
# Shape-keyed runtime tuning.
#
# _CONFIGS is a cache: (M_half, K, B_bucket) -> (BLOCK_M, BLOCK_K, BLOCK_B,
# num_warps, num_stages). Known shapes are pre-seeded from offline sweeps;
# an unseen shape is tuned once on its first eager call (~1-2s, cached for
# the process lifetime), so the same kernel code serves any SiLU-MLP model
# (Qwen2, Llama, Mistral, ...). Tuning never runs during CUDA graph capture
# (do_bench syncs are illegal mid-capture) — the default config is used
# there instead; vLLM's eager warmup runs before capture, so in practice
# new shapes are tuned at model-load time.
# ---------------------------------------------------------------------------

def _bucket_b(B: int) -> int:
    # BLOCK_B=1 pads tl.dot's N dim too thin to beat cuBLAS at B=1 on 3090
    # (measured) — bucket 4 is the smallest viable granularity.
    if B <= 4:
        return 4
    return 16 if B <= 16 else 32


# 16-bit dtypes the kernel supports. Bytes moved and tensor-core throughput
# are identical for both, so tuned configs are shared across dtypes.
_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16)


_CONFIGS = {
    # Qwen2.5-7B, autotuned offline on RTX 3090 (2026-07-06 256-config
    # sweep): BLOCK_K=128 wins despite double-accumulator pressure.
    (18944, 3584, 4):  (64, 128, 4,  4, 3),
    (18944, 3584, 16): (64, 128, 16, 4, 3),
    (18944, 3584, 32): (64, 128, 32, 4, 3),
}
_DEFAULT_CONFIG = (32, 64, 16, 4, 2)

# (M_half, K, 4) keys whose bucket=4 config was already re-tuned from the
# wider B4 space this process — idempotency guard for the B=1 re-tune.
_B4_RETUNED: set = set()

# Pruned candidate space for runtime tuning: winners and near-winners of
# the offline sweep, as (BLOCK_M, BLOCK_K, num_warps, num_stages). Used for
# buckets 16/32 — UNCHANGED from the offline-validated set.
_TUNE_SPACE = [
    (64, 128, 4, 3), (64, 128, 4, 2), (64, 128, 8, 3), (64, 128, 8, 2),
    (128, 128, 4, 2), (128, 128, 8, 2),
    (64, 64, 4, 3), (64, 64, 4, 4),
    (32, 128, 4, 3), (32, 64, 4, 3), (32, 64, 4, 2), (128, 64, 4, 2),
]

# Dedicated wider space for the bucket=4 (B=1) fused path. The fused kernel
# runs TWO padded tl.dots per K-step at B=1 (gate + up, N pinned to 4), so
# it is doubly reduction/latency-bound. Sweep smaller BLOCK_M (more tiles to
# fill SMs), larger BLOCK_K + deeper stages (pipeline the K-loop that now
# feeds two dots), num_warps=2, and thinner BLOCK_B∈{2,4}. Each entry is
# (BLOCK_M, BLOCK_K, BLOCK_B, num_warps, num_stages).
_TUNE_SPACE_B4 = [
    (32, 128, 4, 4, 3), (32, 128, 4, 4, 4), (32, 256, 4, 4, 3),
    (32, 256, 4, 4, 4), (64, 128, 4, 4, 3), (64, 128, 4, 4, 4),
    (64, 128, 4, 8, 3), (64, 256, 4, 4, 3), (64, 256, 4, 8, 3),
    (128, 128, 4, 4, 3), (128, 128, 4, 8, 2), (16, 128, 4, 4, 4),
    (16, 256, 4, 4, 4),
    # thinner N tiles — less doubled padded tensor-core work at B=1
    (32, 128, 2, 4, 4), (32, 256, 2, 4, 4), (64, 256, 2, 4, 3),
    # deep-pipeline, few warps
    (32, 128, 4, 2, 5), (64, 128, 4, 2, 4), (32, 64, 4, 2, 5),
]


def _launch(W, X, Y, M_half, K, B, cfg):
    BLOCK_M, BLOCK_K, BLOCK_B, num_warps, num_stages = cfg
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


@torch.compiler.disable
def _tune(W: torch.Tensor, M_half: int, K: int, bucket: int):
    import triton.testing

    # bucket=4 (B=1 target) searches its own wider space; 16/32 unchanged.
    if bucket == 4:
        candidates = list(_TUNE_SPACE_B4)
    else:
        candidates = [(bm, bk, bucket, nw, ns) for (bm, bk, nw, ns) in _TUNE_SPACE]

    X = torch.randn(bucket, K, dtype=W.dtype, device=W.device)
    Y = torch.empty((bucket, M_half), dtype=W.dtype, device=W.device)
    best, best_t = None, float("inf")
    for cfg in candidates:
        try:
            t = triton.testing.do_bench(
                lambda: _launch(W, X, Y, M_half, K, bucket, cfg),
                warmup=10, rep=50, return_mode="median",
            )
        except Exception:
            continue  # OutOfResources etc. — config invalid on this GPU
        if t < best_t:
            best, best_t = cfg, t
    if best is None:
        bm, bk, _, nw, ns = _DEFAULT_CONFIG
        best = (bm, bk, bucket, nw, ns)
    print(f"[fused_gate_up_silu] tuned (M_half={M_half}, K={K}, "
          f"bucket={bucket}) -> {best} ({best_t*1e3:.0f}us)")
    return best


@torch.compiler.disable
def _time_cfg(W: torch.Tensor, M_half: int, K: int, bucket: int, cfg) -> float:
    """Median runtime (ms) of one specific config, for safe re-tune compares."""
    import triton.testing
    X = torch.randn(bucket, K, dtype=W.dtype, device=W.device)
    Y = torch.empty((bucket, M_half), dtype=W.dtype, device=W.device)
    try:
        return triton.testing.do_bench(
            lambda: _launch(W, X, Y, M_half, K, bucket, cfg),
            warmup=10, rep=50, return_mode="median",
        )
    except Exception:
        return float("inf")


def _pick_config(W: torch.Tensor, M_half: int, K: int, B: int):
    bucket = _bucket_b(B)
    key = (M_half, K, bucket)
    cfg = _CONFIGS.get(key)
    if cfg is None:
        # Never tune during dynamo tracing (vLLM compiles fullgraph — a
        # graph break is a hard error) or CUDA graph capture (do_bench
        # syncs are illegal mid-capture). An unseeded shape here means
        # ensure_tuned() was not called at weight-load time; degrade to
        # the default config rather than crash.
        if torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing():
            bm, bk, _, nw, ns = _DEFAULT_CONFIG
            return (bm, bk, bucket, nw, ns)
        cfg = _tune(W, M_half, K, bucket)
        _CONFIGS[key] = cfg
    return cfg


def ensure_tuned(W: torch.Tensor) -> None:
    """Tune all B-buckets for W's shape if not yet cached.

    Call this at WEIGHT-LOAD time (e.g. from Model.load_weights), which
    runs eagerly. Under vLLM's fullgraph torch.compile the kernel config
    is frozen into the traced graph as constants, and the decode-sized
    calls first execute during CUDA graph capture — so tuning any later
    than load is either a hard compile error or a silent no-op, and an
    unseeded shape would bake the default config into the graphs.
    """
    if torch.compiler.is_compiling():
        return  # must not tune (or graph-break) during tracing
    if W.ndim != 2 or W.dtype not in _SUPPORTED_DTYPES or not W.is_cuda:
        return
    if W.shape[0] % 2 != 0:
        return
    M_half, K = W.shape[0] // 2, W.shape[1]
    if torch.cuda.is_current_stream_capturing():
        return
    for bucket in (16, 32):
        key = (M_half, K, bucket)
        if key not in _CONFIGS:
            _CONFIGS[key] = _tune(W, M_half, K, bucket)
    # Re-tune bucket=4 from the wider B4 space, but ADOPT the new config only
    # if it is strictly faster than the existing seed — otherwise keep the
    # seed. This lets unseeded shapes benefit while guaranteeing pre-seeded
    # fused shapes never regress vs their offline-validated bucket=4 config.
    # Scoped to bucket=4 only; 16/32 above tuned only when unseeded.
    key4 = (M_half, K, 4)
    if key4 not in _B4_RETUNED:
        _B4_RETUNED.add(key4)
        new4 = _tune(W, M_half, K, 4)
        seed4 = _CONFIGS.get(key4)
        if seed4 is None:
            _CONFIGS[key4] = new4
        else:
            t_new = _time_cfg(W, M_half, K, 4, new4)
            t_seed = _time_cfg(W, M_half, K, 4, seed4)
            if t_new < t_seed:
                _CONFIGS[key4] = new4
                print(f"[fused_gate_up_silu] bucket4 adopted new "
                      f"(M_half={M_half}, K={K}) {new4} "
                      f"({t_new*1e3:.0f}us < seed {t_seed*1e3:.0f}us)")
            else:
                print(f"[fused_gate_up_silu] bucket4 kept seed "
                      f"(M_half={M_half}, K={K}) {seed4} "
                      f"({t_seed*1e3:.0f}us <= new {t_new*1e3:.0f}us)")


def triton_fused_gate_up_silu(W: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """
    Args:
        W : [2*M_half, K] bfloat16 or float16 — fused gate+up projection weight
        X : [B, K] same dtype as W,  B = 1..32

    Returns:
        Y : [B, M_half] same dtype as inputs — silu(gate) * up
    """
    assert W.ndim == 2 and X.ndim == 2
    assert W.dtype in _SUPPORTED_DTYPES and X.dtype == W.dtype
    assert W.is_cuda and X.is_cuda
    assert W.shape[0] % 2 == 0, "W rows must be even (gate+up stacked)"

    M2, K  = W.shape
    M_half = M2 // 2
    B      = X.shape[0]

    cfg = _pick_config(W, M_half, K, B)
    Y   = torch.empty((B, M_half), dtype=X.dtype, device=W.device)
    _launch(W, X, Y, M_half, K, B, cfg)
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
    for dtype in _SUPPORTED_DTYPES:
        print(f"--- {dtype} ---")
        for B in batch_sizes:
            W = torch.randn(2 * M_half, K, dtype=dtype, device="cuda")
            X = torch.randn(B, K, dtype=dtype, device="cuda")

            # Reference: separate GEMM + silu_and_mul
            gate_up = (X.float() @ W.float().t()).to(dtype)      # [B, 2*M_half]
            gate, up = gate_up[:, :M_half], gate_up[:, M_half:]
            ref = (torch.nn.functional.silu(gate.float()) * up.float()).to(dtype)

            # Fused kernel
            out = triton_fused_gate_up_silu(W, X)

            max_diff  = (ref.float() - out.float()).abs().max().item()
            out_scale = max(ref.float().abs().max().item(), 1.0)
            passed    = max_diff < out_scale * 0.05  # 5% relative — output is silu*up ~O(K)
            if not passed:
                all_passed = False
            print(f"  B={B:2d}  max_abs={max_diff:.4f}  {'PASSED' if passed else 'FAILED'}")

    print()
    print("Overall:", "PASSED" if all_passed else "FAILED")

    print("\n=== Speed: fused vs unfused (us) ===")
    from skinny_gemm import triton_skinny_gemm  # sibling file, not the vllm package —
    # avoids vllm/kernels/__init__.py's eager import chain (aiter_ops -> vllm.platforms
    # -> vllm._C), which the Artemis runner's precompiled build doesn't currently ship

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
