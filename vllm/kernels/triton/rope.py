"""Triton rotary position embedding (RoPE) kernel, neox-style.

Drop-in for vLLM's CUDA op torch.ops._C.rotary_embedding with
is_neox=True and rot_dim == head_size (Llama / Qwen2 / Mistral layout):

  positions     : [T]            int64 token positions
  query         : [T, Hq * D]    rotated IN-PLACE
  key           : [T, Hk * D]    rotated IN-PLACE
  cos_sin_cache : [max_pos, D]   cos in [:, :D/2], sin in [:, D/2:]

One program per (token, head); rotation math in fp32, matching the CUDA
op's pairing x1' = x1*cos - x2*sin, x2' = x2*cos + x1*sin over pairs
(i, i + D/2).

Race-gated like the other framework kernels: ensure_tuned() times Triton
vs the CUDA op and enables per (head_size, dtype) only on a >=2% win.
"""

import sys
import torch
import triton
import triton.language as tl


@triton.jit
def _rope_kernel(
    POS_ptr, Q_ptr, K_ptr, CACHE_ptr,
    NQ: tl.constexpr, NK: tl.constexpr,
    stride_q, stride_k,
    HALF: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)

    pos = tl.load(POS_ptr + pid_t)
    offs = tl.arange(0, HALF)
    cos = tl.load(CACHE_ptr + pos * (2 * HALF) + offs).to(tl.float32)
    sin = tl.load(CACHE_ptr + pos * (2 * HALF) + HALF + offs).to(tl.float32)

    if pid_h < NQ:
        base = Q_ptr + pid_t * stride_q + pid_h * (2 * HALF)
    else:
        base = K_ptr + pid_t * stride_k + (pid_h - NQ) * (2 * HALF)

    x1 = tl.load(base + offs).to(tl.float32)
    x2 = tl.load(base + HALF + offs).to(tl.float32)
    tl.store(base + offs, (x1 * cos - x2 * sin).to(base.dtype.element_ty))
    tl.store(base + HALF + offs, (x2 * cos + x1 * sin).to(base.dtype.element_ty))


_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16)

# (head_size, dtype) -> num_warps
_CONFIGS: dict = {}
_DEFAULT_WARPS = 1
_ENABLED: set = set()
_RACED: set = set()
_WARP_SPACE = (1, 2, 4)
_ENABLE_MARGIN = 1.02


def is_enabled(head_size: int, dtype: torch.dtype) -> bool:
    return (head_size, dtype) in _ENABLED


def triton_rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
) -> None:
    """In-place neox RoPE on query and key. rot_dim must equal head_size."""
    assert cos_sin_cache.shape[-1] == head_size, "rot_dim != head_size unsupported"
    assert head_size % 2 == 0
    assert query.dtype in _SUPPORTED_DTYPES and key.dtype == query.dtype

    T = positions.numel()
    nq = query.shape[-1] // head_size
    nk = key.shape[-1] // head_size
    _rope_kernel[(T, nq + nk)](
        positions, query, key, cos_sin_cache,
        NQ=nq, NK=nk,
        stride_q=query.stride(0), stride_k=key.stride(0),
        HALF=head_size // 2,
        num_warps=_CONFIGS.get((head_size, query.dtype), _DEFAULT_WARPS),
    )


def ensure_tuned(head_size: int, num_q_heads: int, num_k_heads: int,
                 cos_sin_cache: torch.Tensor,
                 dtype: torch.dtype = torch.bfloat16) -> None:
    """Tune num_warps and race vs _C.rotary_embedding at decode T=32.
    Same weight-load-time window constraint as the other framework kernels."""
    if torch.compiler.is_compiling():
        return
    if dtype not in _SUPPORTED_DTYPES or cos_sin_cache.shape[-1] != head_size:
        return
    if torch.cuda.is_current_stream_capturing():
        return
    import triton.testing
    import vllm._custom_ops  # noqa: F401

    key = (head_size, dtype)
    if key in _RACED:
        return
    _RACED.add(key)

    dev = cos_sin_cache.device
    cache = cos_sin_cache.to(dtype)
    T = 32
    pos = torch.randint(0, cache.shape[0], (T,), device=dev, dtype=torch.int64)
    q = torch.randn(T, num_q_heads * head_size, dtype=dtype, device=dev)
    k = torch.randn(T, num_k_heads * head_size, dtype=dtype, device=dev)

    def bench(fn):
        return triton.testing.do_bench(fn, warmup=10, rep=50, return_mode="median")

    best_nw, best_t = _DEFAULT_WARPS, float("inf")
    for nw in _WARP_SPACE:
        def run():
            _rope_kernel[(T, num_q_heads + num_k_heads)](
                pos, q, k, cache,
                NQ=num_q_heads, NK=num_k_heads,
                stride_q=q.stride(0), stride_k=k.stride(0),
                HALF=head_size // 2, num_warps=nw,
            )
        try:
            t = bench(run)
        except Exception:
            continue
        if t < best_t:
            best_nw, best_t = nw, t
    _CONFIGS[key] = best_nw

    t_cuda = bench(lambda: torch.ops._C.rotary_embedding(
        pos, q, k, head_size, cache, True))
    print(f"[rope] tuned (D={head_size}, {dtype}) -> warps={best_nw} "
          f"triton={best_t*1e3:.1f}us cuda={t_cuda*1e3:.1f}us")
    if best_t * _ENABLE_MARGIN < t_cuda:
        _ENABLED.add(key)
        print(f"[rope] ENABLED (D={head_size}, {dtype}) — beats CUDA op")
    else:
        print(f"[rope] NOT enabled (D={head_size}, {dtype}) — CUDA op kept")


# ---------------------------------------------------------------------------
# Correctness check vs the CUDA op
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import vllm._custom_ops  # noqa: F401

    torch.manual_seed(0)
    all_passed = True
    max_pos = 4096

    # (head_size, nq, nk): Qwen2.5-7B / Mistral-7B / Llama-3.1-8B layouts
    for D, nq, nk in ((128, 28, 4), (128, 32, 8)):
        half = D // 2
        inv = 1.0 / (10000.0 ** (torch.arange(0, half, dtype=torch.float32) / half))
        t = torch.arange(max_pos, dtype=torch.float32)
        freqs = torch.outer(t, inv)
        for dtype in _SUPPORTED_DTYPES:
            cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(dtype).cuda()
            for T in (1, 16, 32, 256):
                pos = torch.randint(0, max_pos, (T,), device="cuda", dtype=torch.int64)
                q = torch.randn(T, nq * D, dtype=dtype, device="cuda")
                k = torch.randn(T, nk * D, dtype=dtype, device="cuda")

                q_ref, k_ref = q.clone(), k.clone()
                torch.ops._C.rotary_embedding(pos, q_ref, k_ref, D, cache, True)
                triton_rotary_embedding(pos, q, k, D, cache)

                diff = max((q - q_ref).abs().max().item(),
                           (k - k_ref).abs().max().item())
                ok = diff < 0.01  # same math, same precision — near bit-exact
                all_passed &= ok
                print(f"  D={D} q{nq}/k{nk} {str(dtype):15s} T={T:4d}  "
                      f"max_abs_vs_cuda={diff:.5f}  {'PASSED' if ok else 'FAILED'}")

    print("\nOverall:", "PASSED" if all_passed else "FAILED")
    sys.exit(0 if all_passed else 1)
