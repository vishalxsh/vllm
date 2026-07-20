"""Triton RMSNorm kernels (plain + fused residual-add) for vLLM decode.

Same framework contract as skinny_gemm: shape-keyed configs, ensure_tuned()
at weight-load time races Triton against the incumbent (vLLM's CUDA ops
torch.ops._C.rms_norm / fused_add_rms_norm) and enables Triton per
(N, dtype) only if it wins — no model can get slower.

Numerics: full fp32 row math with a single rounding at the store. This is
slightly MORE accurate than the CUDA op, which rounds x*rstd to the model
dtype before multiplying by the weight.

  X : [rows, N]   rows = tokens (1..32 decode, thousands prefill)
  W : [N]
"""

import sys
import torch
import triton
import triton.language as tl


@triton.jit
def _rms_norm_kernel(
    X_ptr, W_ptr, Y_ptr,
    N,
    stride_x, stride_y,
    eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)
    rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / N + eps)
    w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(Y_ptr + row * stride_y + offs,
             (x * rstd * w).to(Y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _fused_add_rms_norm_kernel(
    X_ptr, R_ptr, W_ptr,
    N,
    stride_x, stride_r,
    eps,
    BLOCK_N: tl.constexpr,
):
    # In-place, mirroring _C.fused_add_rms_norm:
    #   residual <- x + residual ;  x <- rmsnorm(residual) * w
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(R_ptr + row * stride_r + offs, mask=mask, other=0.0).to(tl.float32)
    s = x + r
    tl.store(R_ptr + row * stride_r + offs,
             s.to(R_ptr.dtype.element_ty), mask=mask)
    rstd = 1.0 / tl.sqrt(tl.sum(s * s, axis=0) / N + eps)
    w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(X_ptr + row * stride_x + offs,
             (s * rstd * w).to(X_ptr.dtype.element_ty), mask=mask)


# ---------------------------------------------------------------------------
# Shape-keyed configs + race-gated enablement (house pattern).
# BLOCK_N must cover the full row (single-pass rstd), so the only free knob
# is num_warps; tuned per hidden size at load time.
# ---------------------------------------------------------------------------

_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16)

# (N, dtype) -> num_warps, shared by plain and fused variants.
_CONFIGS: dict = {}
_DEFAULT_WARPS = 8

# (N, dtype) whose race vs the CUDA op has run / been won this process.
_ENABLED: set = set()
_RACED: set = set()

_WARP_SPACE = (1, 2, 4, 8, 16)
_ENABLE_MARGIN = 1.02


def is_enabled(N: int, dtype: torch.dtype) -> bool:
    return (N, dtype) in _ENABLED


def _pick_warps(N: int, dtype: torch.dtype) -> int:
    return _CONFIGS.get((N, dtype), _DEFAULT_WARPS)


def triton_rms_norm(x: torch.Tensor, weight: torch.Tensor,
                    eps: float) -> torch.Tensor:
    """Out-of-place RMSNorm. x: [rows, N], weight: [N]."""
    rows, N = x.shape
    y = torch.empty_like(x)
    _rms_norm_kernel[(rows,)](
        x, weight, y, N, x.stride(0), y.stride(0), eps,
        BLOCK_N=triton.next_power_of_2(N),
        num_warps=_pick_warps(N, x.dtype),
    )
    return y


def triton_fused_add_rms_norm(x: torch.Tensor, residual: torch.Tensor,
                              weight: torch.Tensor, eps: float) -> None:
    """In-place: residual += x, then x = rmsnorm(residual)*w."""
    rows, N = x.shape
    _fused_add_rms_norm_kernel[(rows,)](
        x, residual, weight, N, x.stride(0), residual.stride(0), eps,
        BLOCK_N=triton.next_power_of_2(N),
        num_warps=_pick_warps(N, x.dtype),
    )


def _bench(fn):
    import triton.testing
    return triton.testing.do_bench(fn, warmup=10, rep=50, return_mode="median")


def ensure_tuned(weight: torch.Tensor, eps: float = 1e-6) -> None:
    """Tune num_warps for this hidden size and race vs _C.fused_add_rms_norm
    (the hot serving path); enable iff Triton wins at decode rows. Call at
    WEIGHT-LOAD time (same torch.compile / CUDA-graph window constraint as
    skinny_gemm.ensure_tuned)."""
    if torch.compiler.is_compiling():
        return
    if weight.ndim != 1 or weight.dtype not in _SUPPORTED_DTYPES or not weight.is_cuda:
        return
    if torch.cuda.is_current_stream_capturing():
        return
    import vllm._custom_ops  # noqa: F401 — registers torch.ops._C

    N = weight.numel()
    key = (N, weight.dtype)
    if key in _RACED:
        return
    _RACED.add(key)

    rows = 32
    x = torch.randn(rows, N, dtype=weight.dtype, device=weight.device)
    r = torch.randn(rows, N, dtype=weight.dtype, device=weight.device)

    best_nw, best_t = _DEFAULT_WARPS, float("inf")
    for nw in _WARP_SPACE:
        def run():
            _fused_add_rms_norm_kernel[(rows,)](
                x, r, weight, N, x.stride(0), r.stride(0), eps,
                BLOCK_N=triton.next_power_of_2(N), num_warps=nw,
            )
        try:
            t = _bench(run)
        except Exception:
            continue
        if t < best_t:
            best_nw, best_t = nw, t
    _CONFIGS[key] = best_nw

    t_cuda = _bench(lambda: torch.ops._C.fused_add_rms_norm(x, r, weight, eps))
    print(f"[rms_norm] tuned (N={N}, {weight.dtype}) -> warps={best_nw} "
          f"triton={best_t*1e3:.1f}us cuda={t_cuda*1e3:.1f}us")
    if best_t * _ENABLE_MARGIN < t_cuda:
        _ENABLED.add(key)
        print(f"[rms_norm] ENABLED (N={N}, {weight.dtype}) — beats CUDA op")
    else:
        print(f"[rms_norm] NOT enabled (N={N}, {weight.dtype}) — CUDA op kept")


# ---------------------------------------------------------------------------
# Correctness check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    eps = 1e-6
    all_passed = True

    for dtype in _SUPPORTED_DTYPES:
        print(f"--- {dtype} ---")
        for N in (3584, 4096):
            w = (torch.rand(N, dtype=dtype, device="cuda") * 0.5 + 0.75)
            for rows in (1, 16, 32, 2048):
                x = torch.randn(rows, N, dtype=dtype, device="cuda")
                r = torch.randn(rows, N, dtype=dtype, device="cuda")

                # plain
                ref = (x.float() * torch.rsqrt((x.float() ** 2).mean(-1, keepdim=True) + eps)
                       ) * w.float()
                out = triton_rms_norm(x, w, eps)
                e1 = (out.float() - ref).abs().max().item()

                # fused add (against fp32 reference of the same in-place contract)
                s = x.float() + r.float()
                ref_r = s.to(dtype)
                ref_x = ((s * torch.rsqrt((s ** 2).mean(-1, keepdim=True) + eps)) * w.float()).to(dtype)
                x2, r2 = x.clone(), r.clone()
                triton_fused_add_rms_norm(x2, r2, w, eps)
                e2 = max((x2.float() - ref_x.float()).abs().max().item(),
                         (r2.float() - ref_r.float()).abs().max().item())

                tol = 0.05
                ok = e1 < tol and e2 < tol
                all_passed &= ok
                print(f"  N={N} rows={rows:4d}  plain={e1:.4f} fused={e2:.4f}  "
                      f"{'PASSED' if ok else 'FAILED'}")

    print("\nOverall:", "PASSED" if all_passed else "FAILED")
    sys.exit(0 if all_passed else 1)
