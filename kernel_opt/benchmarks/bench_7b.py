import os, sys, torch, statistics, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from kernels.gemv import triton_gemv

DEVICE = f"cuda:{os.environ.get('CUDA_DEVICE', '1')}"
torch.cuda.set_device(DEVICE)

WEIGHT_SHAPES = [
    (3584,  3584,  "attn_proj"),
    (18944, 3584,  "ffn_gate "),
    (3584,  18944, "ffn_down "),
]
BATCH_SIZES = [1]
DTYPE       = torch.bfloat16
N_RUNS      = 5
N_WARMUP    = 100
N_MEASURE   = 300


def bench(fn):
    times = []
    for _ in range(N_RUNS):
        for _ in range(N_WARMUP): fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(N_MEASURE): fn()
        end.record(); torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / N_MEASURE * 1000)
    return statistics.median(times)


print(f"dtype: {DTYPE}  |  runs: {N_RUNS}  |  warmup: {N_WARMUP}  |  measure: {N_MEASURE}")
print(f"{'Shape':>35}  {'cuBLAS (us)':>12}  {'Triton (us)':>12}  {'Speedup':>8}")
print("-" * 75)

results = {}
all_faster = True

for B in BATCH_SIZES:
    for M, N, label in WEIGHT_SHAPES:
        W = torch.randn(M, N, dtype=DTYPE, device=DEVICE)
        x = torch.randn(B, N, dtype=DTYPE, device=DEVICE)

        if B == 1:
            x1 = x.squeeze(0)
            t_cub = bench(lambda: torch.mv(W, x1))
            try:
                t_tri = bench(lambda: triton_gemv(W, x1))
            except Exception:
                t_tri = float("inf")
        else:
            t_cub = bench(lambda: torch.mm(x, W.t()))
            try:
                t_tri = bench(lambda: triton_gemv(W, x))
            except Exception:
                t_tri = float("inf")

        sp   = t_cub / t_tri if t_tri != float("inf") else 0.0
        flag = "FASTER" if sp >= 1.0 else "SLOWER"
        if sp < 1.0:
            all_faster = False

        key = f"{label.strip()}_B{B}"
        print(f"  B={B:2d} {label} [{M:6d},{N:6d}]  {t_cub:>12.2f}  {t_tri:>12.2f}  {sp:>7.2f}x {flag}")
        results[key] = {
            "batch":     B,
            "cublas_us": round(t_cub, 2),
            "triton_us": round(t_tri, 2),
            "speedup":   round(sp, 4),
        }
    print()

print("Verdict:", "ALL SHAPES FASTER" if all_faster else "SOME SHAPES SLOWER")

import json, pathlib

# Flat metrics for Artemis Metrics tab
flat = {}
for key, v in results.items():
    flat[f"speedup_{key}"] = v["speedup"]
    flat[f"triton_us_{key}"] = v["triton_us"] if v["triton_us"] != float("inf") else -1.0
for B in BATCH_SIZES:
    batch_speedups = [v["speedup"] for k, v in results.items() if v["batch"] == B and v["speedup"] > 0]
    flat[f"avg_speedup_B{B}"] = round(sum(batch_speedups) / len(batch_speedups), 4) if batch_speedups else 0.0
flat["all_faster"] = 1.0 if all_faster else 0.0

out_path = pathlib.Path(sys.argv[1]) / "artemis_results.json" if len(sys.argv) > 1 \
           else pathlib.Path("artemis_results.json")
out_path.write_text(json.dumps(flat, indent=2) + "\n")
print(f"Wrote metrics to: {out_path.resolve()}")
