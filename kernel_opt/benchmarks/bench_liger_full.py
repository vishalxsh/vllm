"""Full op-by-op matrix: our framework vs Liger Kernel vs stock vLLM.

Covers every inference-relevant op Liger implements — no missing rows:
  SwiGLU MLP stage, GeGLU MLP stage, RMSNorm (plain + fused residual-add),
  RoPE.
Liger's remaining ops (CrossEntropy, FusedLinearCE, KLDiv/JSD, DPO/ORPO
losses) are TRAINING-ONLY (need labels/gradients) and are out of scope for
an inference engine — declared, not benchmarked.

Each implementation runs in its author's intended integration and native
layout. Protocol: do_bench medians, dtype from argv (bf16 default, fp16
supported — races and CUDA/cuBLAS algorithm choices differ per dtype so
both matrices matter), warmup=25/rep=100, accuracy vs
fp32 reference where semantics are comparable. Our RMSNorm/RoPE kernels are
race-gated: the [rms_norm]/[rope] lines printed at the top are the load-time
enablement decisions on THIS GPU.

Usage: CUDA_VISIBLE_DEVICES=3 .venv/bin/python kernel_opt/benchmarks/bench_liger_full.py [bfloat16|float16]
"""

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
import triton.testing

from liger_kernel.ops.swiglu import swiglu_forward
from liger_kernel.ops.geglu import geglu_forward
from liger_kernel.ops.rms_norm import rms_norm_forward
from liger_kernel.ops.rope import rope_forward

import vllm._custom_ops  # noqa: F401 — registers torch.ops._C
from vllm.kernels.triton.fused_gate_up_silu import (
    ensure_tuned as fused_ensure_tuned,
    triton_fused_gate_up_silu,
)
from vllm.kernels.triton.fused_gate_up_gelu import triton_fused_gate_up_gelu
from vllm.kernels.triton import rms_norm as our_rms
from vllm.kernels.triton import rope as our_rope

DEV = "cuda"
DTYPE_NAME = sys.argv[1] if len(sys.argv) > 1 else "bfloat16"
DTYPE = {"bfloat16": torch.bfloat16, "float16": torch.float16}[DTYPE_NAME]
WARMUP, REP = 25, 100
results = []


def bench(fn):
    return triton.testing.do_bench(fn, warmup=WARMUP, rep=REP, return_mode="median")


def rel_err(out, ref_f32):
    return ((out.float() - ref_f32).abs().max() / ref_f32.abs().max().clamp(min=1e-6)).item()


def row(section, **kw):
    results.append(dict(section=section, **kw))


us = 1e3

# ---------------------------------------------------------------- race gates
print("== framework load-time tuning / races ==")
for H in (3584, 4096):
    w = (torch.rand(H, dtype=DTYPE, device=DEV) * 0.5 + 0.75)
    our_rms.ensure_tuned(w)
D, NQ, NK, MAXPOS = 128, 28, 4, 4096  # Qwen2.5-7B attention layout
half = D // 2
inv = 1.0 / (10000.0 ** (torch.arange(0, half, dtype=torch.float32) / half))
freqs = torch.outer(torch.arange(MAXPOS, dtype=torch.float32), inv)
vllm_cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(DTYPE).to(DEV)
our_rope.ensure_tuned(D, NQ, NK, vllm_cache, DTYPE)
print()

# ------------------------------------------------------- A/B: GLU MLP stages
def glu_stage(title, M_half, K, act_ref, vllm_op, liger_ew, ours_fused, case):
    print("=" * 98)
    print(f"{title}   x[B,{K}] -> [B,{M_half}]  ({DTYPE_NAME}, us)")
    print("=" * 98)
    print(f"{'B':>3} | {'vllm_stock':>10} {'liger':>10} {'ours':>10} | "
          f"{'ours/stock':>10} {'ours/liger':>10} | err(stock/liger/ours)")
    torch.manual_seed(0)
    W2 = torch.randn(2 * M_half, K, dtype=DTYPE, device=DEV) / K**0.5
    Wg, Wu = W2[:M_half].contiguous(), W2[M_half:].contiguous()
    fused_ensure_tuned(W2)
    for B in (1, 16, 32):
        x = torch.randn(B, K, dtype=DTYPE, device=DEV)
        ref = act_ref(x.float() @ Wg.float().t()) * (x.float() @ Wu.float().t())

        def stock():
            gu = F.linear(x, W2)
            out = torch.empty(B, M_half, dtype=DTYPE, device=DEV)
            vllm_op(out, gu)
            return out

        def liger():
            return liger_ew(F.linear(x, Wg), F.linear(x, Wu))[2]

        def ours():
            return ours_fused(W2, x)

        t_s, t_l, t_o = bench(stock), bench(liger), bench(ours)
        e_s, e_l, e_o = rel_err(stock(), ref), rel_err(liger(), ref), rel_err(ours(), ref)
        print(f"{B:>3} | {t_s*us:>10.1f} {t_l*us:>10.1f} {t_o*us:>10.1f} | "
              f"{t_s/t_o:>9.3f}x {t_l/t_o:>9.3f}x | {e_s:.1e}/{e_l:.1e}/{e_o:.1e}")
        row(case, M_half=M_half, K=K, B=B,
            t_vllm_us=round(t_s*us, 2), t_liger_us=round(t_l*us, 2),
            t_ours_us=round(t_o*us, 2),
            speedup_ours_vs_stock=round(t_s/t_o, 3),
            speedup_ours_vs_liger=round(t_l/t_o, 3),
            err_vllm=e_s, err_liger=e_l, err_ours=e_o)
    print()


glu_stage("A. SwiGLU MLP stage (Qwen2.5-7B)", 18944, 3584,
          F.silu, torch.ops._C.silu_and_mul, swiglu_forward,
          triton_fused_gate_up_silu, "swiglu_mlp")

glu_stage("B. GeGLU MLP stage (Gemma-class shape)", 14336, 3584,
          lambda t: F.gelu(t, approximate="tanh"),
          torch.ops._C.gelu_tanh_and_mul,
          lambda a, b: geglu_forward(a, b),
          triton_fused_gate_up_gelu, "geglu_mlp")

# ------------------------------------------------------------- C: RMSNorm
print("=" * 98)
print(f"C. RMSNorm plain (out-of-place)  [rows, H]  ({DTYPE_NAME}, us)")
print("=" * 98)
print(f"{'H':>5} {'rows':>5} | {'vllm CUDA':>10} {'liger':>10} {'ours':>10} | "
      f"{'ours/vllm':>10} {'ours/liger':>10} | err(vllm/liger/ours)")
for H in (3584, 4096):
    w = (torch.rand(H, dtype=DTYPE, device=DEV) * 0.5 + 0.75)
    for rows in (1, 16, 32, 2048):
        x = torch.randn(rows, H, dtype=DTYPE, device=DEV)
        ref = (x.float() * torch.rsqrt((x.float()**2).mean(-1, keepdim=True) + 1e-6)) * w.float()

        def vllm_impl():
            out = torch.empty_like(x)
            torch.ops._C.rms_norm(out, x, w, 1e-6)
            return out

        liger_impl = lambda: rms_norm_forward(x, w, 1e-6, 0.0, "llama", None)[0]
        ours_impl = lambda: our_rms.triton_rms_norm(x, w, 1e-6)

        t_v, t_l, t_o = bench(vllm_impl), bench(liger_impl), bench(ours_impl)
        e_v, e_l, e_o = rel_err(vllm_impl(), ref), rel_err(liger_impl(), ref), rel_err(ours_impl(), ref)
        print(f"{H:>5} {rows:>5} | {t_v*us:>10.1f} {t_l*us:>10.1f} {t_o*us:>10.1f} | "
              f"{t_v/t_o:>9.3f}x {t_l/t_o:>9.3f}x | {e_v:.1e}/{e_l:.1e}/{e_o:.1e}")
        row("rmsnorm_plain", H=H, rows=rows,
            t_vllm_us=round(t_v*us, 2), t_liger_us=round(t_l*us, 2),
            t_ours_us=round(t_o*us, 2),
            speedup_ours_vs_vllm=round(t_v/t_o, 3),
            speedup_ours_vs_liger=round(t_l/t_o, 3),
            err_vllm=e_v, err_liger=e_l, err_ours=e_o)

print()
print("=" * 98)
print("D. Fused residual-add + RMSNorm (in-place; Liger has NO fused-add variant)")
print("=" * 98)
print(f"{'H':>5} {'rows':>5} | {'vllm CUDA':>10} {'ours':>10} | ours/vllm")
for H in (3584, 4096):
    w = (torch.rand(H, dtype=DTYPE, device=DEV) * 0.5 + 0.75)
    for rows in (1, 16, 32, 2048):
        x0 = torch.randn(rows, H, dtype=DTYPE, device=DEV)
        r0 = torch.randn(rows, H, dtype=DTYPE, device=DEV)
        xv, rv = x0.clone(), r0.clone()
        xo, ro = x0.clone(), r0.clone()

        t_v = bench(lambda: torch.ops._C.fused_add_rms_norm(xv, rv, w, 1e-6))
        t_o = bench(lambda: our_rms.triton_fused_add_rms_norm(xo, ro, w, 1e-6))
        print(f"{H:>5} {rows:>5} | {t_v*us:>10.1f} {t_o*us:>10.1f} | {t_v/t_o:>9.3f}x")
        row("rmsnorm_fused_add", H=H, rows=rows,
            t_vllm_us=round(t_v*us, 2), t_ours_us=round(t_o*us, 2),
            t_liger_us=None, speedup_ours_vs_vllm=round(t_v/t_o, 3))

print()
print("=" * 98)
print("E. RoPE (neox, D=128, q28/k4 Qwen layout; each impl in its native layout)")
print("=" * 98)
print(f"{'T':>4} | {'vllm CUDA':>10} {'liger':>10} {'ours':>10} | "
      f"{'ours/vllm':>10} {'ours/liger':>10}")
cos_full = torch.cat((freqs.cos(), freqs.cos()), dim=-1).to(DTYPE).to(DEV)  # HF layout
sin_full = torch.cat((freqs.sin(), freqs.sin()), dim=-1).to(DTYPE).to(DEV)
for T in (1, 16, 32, 256):
    pos = torch.arange(T, device=DEV, dtype=torch.int64)
    q = torch.randn(T, NQ * D, dtype=DTYPE, device=DEV)
    k = torch.randn(T, NK * D, dtype=DTYPE, device=DEV)
    qh = torch.randn(1, NQ, T, D, dtype=DTYPE, device=DEV)  # Liger HF layout
    kh = torch.randn(1, NK, T, D, dtype=DTYPE, device=DEV)
    cos_t, sin_t = cos_full[:T].unsqueeze(0), sin_full[:T].unsqueeze(0)

    t_v = bench(lambda: torch.ops._C.rotary_embedding(pos, q, k, D, vllm_cache, True))
    t_l = bench(lambda: rope_forward(qh, kh, cos_t, sin_t))
    t_o = bench(lambda: our_rope.triton_rotary_embedding(pos, q, k, D, vllm_cache))
    print(f"{T:>4} | {t_v*us:>10.1f} {t_l*us:>10.1f} {t_o*us:>10.1f} | "
          f"{t_v/t_o:>9.3f}x {t_l/t_o:>9.3f}x")
    row("rope", T=T, D=D, nq=NQ, nk=NK,
        t_vllm_us=round(t_v*us, 2), t_liger_us=round(t_l*us, 2),
        t_ours_us=round(t_o*us, 2),
        speedup_ours_vs_vllm=round(t_v/t_o, 3),
        speedup_ours_vs_liger=round(t_l/t_o, 3))

print("\n(out of scope, training-only Liger ops: CrossEntropy, FusedLinearCE, "
      "KLDiv, JSD, DPO/ORPO/CPO losses, embedding backward)")

out = (Path(__file__).resolve().parents[1] / "results"
       / f"liger_full_matrix_{DTYPE_NAME}_20260716.json")
out.write_text(json.dumps(dict(
    liger_version="0.8.0",
    torch_version=torch.__version__,
    gpu=torch.cuda.get_device_name(),
    dtype=DTYPE_NAME,
    protocol=f"do_bench median warmup={WARMUP} rep={REP}",
    note="RoPE timed in each impl's native layout; accuracy col omitted for "
         "RoPE (ours verified bit-exact vs vLLM CUDA op in kernel self-test)",
    rows=results,
), indent=1))
print(f"\nwrote {out}")
sys.exit(0)
