"""Compare hand-tuned Triton kernels vs torch.compile(max-autotune) on decode shapes.

Contenders per case:
  cublas     : eager F.linear (the production baseline)
  ours       : triton_skinny_gemm / triton_fused_gate_up_silu (load-time tuned)
  tc_best    : torch.compile mode="max-autotune-no-cudagraphs" (Inductor picks
               best of ATen/cuBLAS vs its own Triton templates)
  tc_triton  : same but max_autotune_gemm_backends="TRITON" — Inductor MUST use
               its own generated Triton GEMM (pure compiler-codegen quality)

no-cudagraphs variant so every contender is measured as raw kernel launches
(vLLM wraps everything in CUDA graphs anyway, equally for all paths).

do_bench flushes L2 between reps; median of ~100ms of reps.
"""
import importlib.util
import json
import os
import sys
import time

import torch
import torch.nn.functional as F
import triton.testing

REPO = "/home/vishal/Desktop/vllm"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "torch_compile_comparison.json")

def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

skinny = _load("skinny_gemm", f"{REPO}/vllm/kernels/triton/skinny_gemm.py")
fused = _load("fused_gate_up_silu",
              f"{REPO}/vllm/kernels/triton/fused_gate_up_silu.py")

DEV = "cuda"
BATCHES = [1, 16, 32]

GEMM_SHAPES = [
    ("qwen7b_qkv",        4608, 3584),
    ("qwen7b_o",          3584, 3584),
    ("qwen7b_gate_up",   37888, 3584),
    ("qwen7b_down",       3584, 18944),
    ("mistral7b_qkv",     6144, 4096),
    ("mistral7b_lm_head", 32768, 4096),
]

FUSED_SHAPES = [
    ("qwen7b_fused_gate_up_silu",   37888, 3584),   # M_half = 18944
    ("mistral7b_fused_gate_up_silu", 28672, 4096),  # M_half = 14336
]


def bench(fn):
    return triton.testing.do_bench(fn, warmup=25, rep=100, return_mode="median")


def max_rel_err(y, ref):
    return ((y.float() - ref).abs() / ref.abs().clamp_min(1e-3)).max().item()


def compile_fresh(fn, triton_only):
    """Fresh dynamo state per case so cache-size limits never bite."""
    import torch._dynamo
    import torch._inductor.config as icfg
    torch._dynamo.reset()
    icfg.max_autotune_gemm_backends = "TRITON" if triton_only else "ATEN,TRITON"
    return torch.compile(fn, mode="max-autotune-no-cudagraphs", dynamic=False)


def run_case(op, name, W, B, eager_fn, ours_fn):
    K = W.shape[1]
    X = torch.randn(B, K, dtype=torch.bfloat16, device=DEV)
    ref = eager_fn(X.float(), W.float())

    row = {"op": op, "case": name, "M": W.shape[0], "K": K, "B": B}

    y = eager_fn(X, W)
    row["t_cublas_us"] = round(bench(lambda: eager_fn(X, W)) * 1e3, 2)
    row["err_cublas"] = round(max_rel_err(y, ref), 4)

    y = ours_fn(W, X)
    row["t_ours_us"] = round(bench(lambda: ours_fn(W, X)) * 1e3, 2)
    row["err_ours"] = round(max_rel_err(y, ref), 4)

    for tag, triton_only in (("tc_best", False), ("tc_triton", True)):
        t0 = time.time()
        try:
            cfn = compile_fresh(lambda x: eager_fn(x, W), triton_only)
            y = cfn(X)          # triggers compile + autotune
            y = cfn(X)
            row[f"compile_s_{tag}"] = round(time.time() - t0, 1)
            row[f"t_{tag}_us"] = round(bench(lambda: cfn(X)) * 1e3, 2)
            row[f"err_{tag}"] = round(max_rel_err(y, ref), 4)
        except Exception as e:
            row[f"t_{tag}_us"] = None
            row[f"error_{tag}"] = f"{type(e).__name__}: {e}"[:300]

    for tag in ("ours", "tc_best", "tc_triton"):
        t = row.get(f"t_{tag}_us")
        row[f"speedup_{tag}_vs_cublas"] = (
            round(row["t_cublas_us"] / t, 3) if t else None)
    print(json.dumps(row), flush=True)
    return row


def gemm_eager(X, W):
    return F.linear(X, W)


def fused_eager(X, W):
    y = F.linear(X, W)
    gate, up = y.chunk(2, dim=-1)
    return F.silu(gate) * up


def main():
    torch.manual_seed(0)
    results = []

    print(f"# torch {torch.__version__}, GPU {torch.cuda.get_device_name(0)}",
          flush=True)

    for name, M, K in GEMM_SHAPES:
        W = torch.randn(M, K, dtype=torch.bfloat16, device=DEV) * 0.02
        skinny.ensure_tuned(W)  # populate tuned configs (no-op if pre-seeded)
        for B in BATCHES:
            results.append(run_case("gemm", name, W, B,
                                    gemm_eager, skinny.triton_skinny_gemm))
        del W
        torch.cuda.empty_cache()

    for name, M, K in FUSED_SHAPES:
        W = torch.randn(M, K, dtype=torch.bfloat16, device=DEV) * 0.02
        fused.ensure_tuned(W)
        for B in BATCHES:
            results.append(run_case("fused_gate_up_silu", name, W, B,
                                    fused_eager, fused.triton_fused_gate_up_silu))
        del W
        torch.cuda.empty_cache()

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"# wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
