"""Head-to-head: our fused SwiGLU-MLP kernels vs Liger Kernel (v0.8.0).

Liger optimises a mostly-disjoint op set (elementwise/norm fusions, training
losses); the one genuinely overlapping unit is the SwiGLU MLP stage. Three
fair variants, each in its author's own intended integration:

  vllm_stock : merged cuBLAS gate_up GEMM  + vLLM CUDA silu_and_mul
  liger      : 2x cuBLAS GEMMs (gate, up)  + Liger SiLUMul Triton kernel
               (mirrors LigerSwiGLUMLP: separate gate/up projections)
  ours       : single fused Triton kernel (GEMM+GEMM+SiLU+mul, no HBM
               round-trip of the intermediate)

Plus isolation tables (elementwise silu-mul alone; RMSNorm as context) and a
Liger re-tune pass (num_warps sweep — BLOCK_SIZE is fixed by its row-per-
program design) so Liger is not handicapped by configs chosen for A100/H100.

Protocol: triton do_bench medians (L2-flushed), warmup=25/rep=100, bf16,
max relative error vs fp32 reference reported for every variant.

Usage: CUDA_VISIBLE_DEVICES=3 .venv/bin/python kernel_opt/benchmarks/bench_liger.py
"""

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
import triton.testing

from liger_kernel.ops.swiglu import _swiglu_forward_kernel, swiglu_forward
from liger_kernel.ops.rms_norm import rms_norm_forward
from liger_kernel.ops.utils import calculate_settings

import vllm._custom_ops  # noqa: F401 — registers torch.ops._C
from vllm.kernels.triton.fused_gate_up_silu import (
    ensure_tuned as fused_ensure_tuned,
    triton_fused_gate_up_silu,
)

DEV = "cuda"
DTYPE = torch.bfloat16
BATCHES = [1, 16, 32]
WARMUP, REP = 25, 100

# (label, M_half, K) — Qwen2.5-7B and Mistral-7B/Llama-3.1-8B MLP shapes
MLP_SHAPES = [
    ("qwen7b", 18944, 3584),
    ("mistral7b", 14336, 4096),
]

results = []


def bench(fn):
    return triton.testing.do_bench(fn, warmup=WARMUP, rep=REP, return_mode="median")


def rel_err(out, ref_f32):
    return ((out.float() - ref_f32).abs().max() / ref_f32.abs().max().clamp(min=1e-6)).item()


def liger_silu_mul_retuned(gate, up, n_cols):
    """Liger's kernel with the best num_warps for THIS GPU (its only free
    knob: BLOCK_SIZE must cover the whole row by design)."""
    a = gate.view(-1, n_cols)
    b = up.view(-1, n_cols)
    c = torch.empty_like(a)
    BLOCK_SIZE, _ = calculate_settings(n_cols)

    def run(nw):
        _swiglu_forward_kernel[(a.shape[0],)](
            a, b, c, c.stride(-2), 1.0,
            n_cols=n_cols, BLOCK_SIZE=BLOCK_SIZE, num_warps=nw,
        )

    best_nw, best_t = None, float("inf")
    for nw in (4, 8, 16, 32):
        try:
            t = bench(lambda: run(nw))
        except Exception:
            continue
        if t < best_t:
            best_nw, best_t = nw, t
    return best_nw, best_t, c


print("=" * 100)
print("A. SwiGLU MLP stage:  x[B,K] -> silu(gate) * up  [B,M_half]   (bf16, us, do_bench median)")
print("=" * 100)
def silu_and_mul(merged):
    """vLLM's CUDA silu_and_mul op on a merged [rows, 2*d] tensor."""
    out = torch.empty(merged.shape[0], merged.shape[1] // 2,
                      dtype=merged.dtype, device=merged.device)
    torch.ops._C.silu_and_mul(out, merged)
    return out


hdr = (f"{'case':<10} {'B':>3} | {'vllm_stock':>10} {'liger':>10} {'liger_rt':>10} "
       f"{'ours':>10} | {'ours/stock':>10} {'ours/liger':>10} | err(stock/liger/ours)")
print(hdr)
for label, M_half, K in MLP_SHAPES:
    torch.manual_seed(0)
    W2 = torch.randn(2 * M_half, K, dtype=DTYPE, device=DEV) / K**0.5
    Wg, Wu = W2[:M_half].contiguous(), W2[M_half:].contiguous()
    fused_ensure_tuned(W2)
    for B in BATCHES:
        x = torch.randn(B, K, dtype=DTYPE, device=DEV)

        # fp32 reference
        ref = F.silu(x.float() @ Wg.float().t()) * (x.float() @ Wu.float().t())

        # 1. vLLM stock: merged GEMM + CUDA silu_and_mul
        def vllm_stock():
            return silu_and_mul(F.linear(x, W2))
        t_stock = bench(vllm_stock)
        e_stock = rel_err(vllm_stock(), ref)

        # 2. Liger integration: two GEMMs + Liger Triton silu-mul
        def liger_path():
            return swiglu_forward(F.linear(x, Wg), F.linear(x, Wu))[2]
        t_liger = bench(liger_path)
        e_liger = rel_err(liger_path(), ref)

        # 2b. Liger re-tuned (num_warps swept on this GPU), GEMMs unchanged
        gate, up = F.linear(x, Wg), F.linear(x, Wu)
        best_nw, t_ew_rt, _ = liger_silu_mul_retuned(gate, up, M_half)
        t_gemms = bench(lambda: (F.linear(x, Wg), F.linear(x, Wu)))
        t_liger_rt = t_gemms + t_ew_rt

        # 3. ours: single fused kernel
        def ours():
            return triton_fused_gate_up_silu(W2, x)
        t_ours = bench(ours)
        e_ours = rel_err(ours(), ref)

        us = 1e3
        print(f"{label:<10} {B:>3} | {t_stock*us:>10.1f} {t_liger*us:>10.1f} "
              f"{t_liger_rt*us:>10.1f} {t_ours*us:>10.1f} | "
              f"{t_stock/t_ours:>9.3f}x {t_liger/t_ours:>9.3f}x | "
              f"{e_stock:.1e}/{e_liger:.1e}/{e_ours:.1e}  (liger nw={best_nw})")
        results.append(dict(
            section="swiglu_mlp", case=label, M_half=M_half, K=K, B=B,
            t_vllm_stock_us=round(t_stock * us, 2),
            t_liger_us=round(t_liger * us, 2),
            t_liger_retuned_us=round(t_liger_rt * us, 2),
            liger_best_num_warps=best_nw,
            t_ours_us=round(t_ours * us, 2),
            speedup_ours_vs_stock=round(t_stock / t_ours, 3),
            speedup_ours_vs_liger=round(t_liger / t_ours, 3),
            err_stock=e_stock, err_liger=e_liger, err_ours=e_ours,
        ))

print()
print("=" * 100)
print("B. Elementwise silu-mul ONLY (Liger's actual kernel vs vLLM CUDA op; GEMM excluded)")
print("=" * 100)
print(f"{'case':<10} {'rows':>5} | {'vllm _C op':>10} {'liger':>10} {'liger_rt':>10} | liger/vllm")
for label, M_half, K in MLP_SHAPES:
    for rows in [1, 16, 32, 2048]:  # 2048 = training/prefill-like regime
        gate = torch.randn(rows, M_half, dtype=DTYPE, device=DEV)
        up = torch.randn(rows, M_half, dtype=DTYPE, device=DEV)
        merged = torch.cat([gate, up], dim=1).contiguous()

        t_vllm = bench(lambda: silu_and_mul(merged))
        t_liger = bench(lambda: swiglu_forward(gate, up)[2])
        best_nw, t_rt, out_rt = liger_silu_mul_retuned(gate, up, M_half)

        ref = F.silu(gate.float()) * up.float()
        e_vllm = rel_err(silu_and_mul(merged), ref)
        e_liger = rel_err(swiglu_forward(gate, up)[2], ref)

        us = 1e3
        print(f"{label:<10} {rows:>5} | {t_vllm*us:>10.1f} {t_liger*us:>10.1f} "
              f"{t_rt*us:>10.1f} | {t_liger/t_vllm:>9.3f}x  "
              f"err {e_vllm:.1e}/{e_liger:.1e}  (nw={best_nw})")
        results.append(dict(
            section="silu_mul_only", case=label, M_half=M_half, rows=rows,
            t_vllm_us=round(t_vllm * us, 2), t_liger_us=round(t_liger * us, 2),
            t_liger_retuned_us=round(t_rt * us, 2), liger_best_num_warps=best_nw,
            ratio_liger_vs_vllm=round(t_liger / t_vllm, 3),
            err_vllm=e_vllm, err_liger=e_liger,
        ))

print()
print("=" * 100)
print("C. RMSNorm context (op we did NOT optimise: stock vLLM CUDA vs Liger Triton)")
print("=" * 100)
print(f"{'hidden':>6} {'rows':>5} | {'vllm _C op':>10} {'liger':>10} | liger/vllm")
for H in [3584, 4096]:
    w = (torch.rand(H, dtype=DTYPE, device=DEV) * 0.5 + 0.75)

    def vllm_rms(x):
        out = torch.empty_like(x)
        torch.ops._C.rms_norm(out, x, w, 1e-6)
        return out

    for rows in [1, 16, 32, 2048]:
        x = torch.randn(rows, H, dtype=DTYPE, device=DEV)

        t_vllm = bench(lambda: vllm_rms(x))
        t_liger = bench(lambda: rms_norm_forward(x, w, 1e-6, 0.0, "llama", None)[0])

        ref = (x.float() * torch.rsqrt((x.float() ** 2).mean(-1, keepdim=True) + 1e-6)) * w.float()
        e_vllm = rel_err(vllm_rms(x), ref)
        e_liger = rel_err(rms_norm_forward(x, w, 1e-6, 0.0, "llama", None)[0], ref)

        us = 1e3
        print(f"{H:>6} {rows:>5} | {t_vllm*us:>10.1f} {t_liger*us:>10.1f} | "
              f"{t_liger/t_vllm:>9.3f}x  err {e_vllm:.1e}/{e_liger:.1e}")
        results.append(dict(
            section="rmsnorm", hidden=H, rows=rows,
            t_vllm_us=round(t_vllm * us, 2), t_liger_us=round(t_liger * us, 2),
            ratio_liger_vs_vllm=round(t_liger / t_vllm, 3),
            err_vllm=e_vllm, err_liger=e_liger,
        ))

out = Path(__file__).resolve().parents[1] / "results" / "liger_comparison_20260716.json"
out.write_text(json.dumps(dict(
    liger_version="0.8.0",
    torch_version=torch.__version__,
    gpu=torch.cuda.get_device_name(),
    dtype="bfloat16",
    protocol=f"do_bench median warmup={WARMUP} rep={REP}",
    rows=results,
), indent=1))
print(f"\nwrote {out}")
sys.exit(0)
