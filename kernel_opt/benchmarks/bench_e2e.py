"""
E2E benchmark for one (variant, mode) combo — fresh process per run.

Matches the 2026-07-05 protocol:
  - variant patch applied BEFORE LLM() load (CUDAGraph bakes in the path)
  - throughput: 190 prompts x 512 output tokens, max_num_seqs=32
  - latency:    1 prompt  x 100 output tokens, max_num_seqs=1, 7 repeats
  - per-second telemetry (sm clock, temp, power) sampled during measurement

Usage:
  BENCH_GPU=3 CUDA_VISIBLE_DEVICES=3 .venv/bin/python bench_e2e.py \
      --variant stock_cublas|fused_gate_up_silu --mode throughput|latency \
      --out result.json
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import statistics
import subprocess
import sys
import threading
import time

p = argparse.ArgumentParser()
p.add_argument("--variant", required=True,
               choices=["stock_cublas", "fused_gate_up_silu"])
p.add_argument("--mode", required=True, choices=["throughput", "latency"])
p.add_argument("--out", required=True)
p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
args = p.parse_args()

GPU_INDEX = int(os.environ.get("BENCH_GPU", "3"))
MODEL = args.model
INPUT_LEN = 32
PROMPT = "Hello, how are you? " * (INPUT_LEN // 5)

sys.path.insert(0, "/home/vishal/Desktop/vllm")

# ── Telemetry sampler ─────────────────────────────────────────────────────────
class Telemetry:
    def __init__(self, gpu: int):
        self.gpu = gpu
        self.samples = []
        self._stop = threading.Event()
        self._thread = None

    def _loop(self):
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "-i", str(self.gpu),
                     "--query-gpu=clocks.sm,temperature.gpu,power.draw",
                     "--format=csv,noheader,nounits"], text=True).strip()
                clk, temp, pwr = [float(x) for x in out.split(",")]
                self.samples.append({"clk": clk, "temp": temp, "pwr": pwr})
            except Exception:
                pass
            self._stop.wait(1.0)

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)

    def summary(self):
        if not self.samples:
            return None
        def agg(key):
            vals = [s[key] for s in self.samples]
            return {"min": min(vals), "max": max(vals),
                    "mean": sum(vals) / len(vals)}
        return {"n": len(self.samples), "clk_mhz": agg("clk"),
                "temp_c": agg("temp"), "power_w": agg("pwr")}


# ── Variant patch (BEFORE LLM load) ───────────────────────────────────────────
stack = contextlib.ExitStack()
if args.variant == "stock_cublas":
    sys.path.insert(0, "/home/vishal/Desktop/vllm/kernel_opt")
    from profiling.variants import REGISTRY
    stack.enter_context(REGISTRY["stock_cublas"].active())
    print("Applied stock_cublas patch (pre-load)")
# fused_gate_up_silu is the default integrated path — no patch

from vllm import LLM, SamplingParams  # noqa: E402
import torch  # noqa: E402

max_num_seqs = 32 if args.mode == "throughput" else 1
print(f"Loading {MODEL} (max_num_seqs={max_num_seqs}) ...")
t_load = time.time()
llm = LLM(
    model=MODEL,
    dtype="bfloat16",
    gpu_memory_utilization=0.90,
    max_num_seqs=max_num_seqs,
    trust_remote_code=True,
)
print(f"Model loaded in {time.time()-t_load:.0f}s")

result = {
    "variant": args.variant,
    "mode": args.mode,
    "gpu": GPU_INDEX,
    "model": MODEL,
    "max_num_seqs": max_num_seqs,
}
tele = Telemetry(GPU_INDEX)

if args.mode == "throughput":
    NUM_PROMPTS, OUTPUT_LEN = 190, 512
    params = SamplingParams(temperature=0.0, max_tokens=OUTPUT_LEN,
                            ignore_eos=True)
    prompts = [PROMPT] * NUM_PROMPTS

    def run():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        llm.generate(prompts, params)
        torch.cuda.synchronize()
        return time.perf_counter() - t0

    print("Warmup run (autotune/JIT, discarded)...")
    warm = run()
    print(f"  warmup: {warm:.1f}s ({NUM_PROMPTS*OUTPUT_LEN/warm:.1f} tok/s)")

    print("Measurement run...")
    tele.start()
    elapsed = run()
    tele.stop()
    tps = NUM_PROMPTS * OUTPUT_LEN / elapsed
    print(f"  measurement: {elapsed:.1f}s  {tps:.1f} output tok/s")
    result.update({
        "num_prompts": NUM_PROMPTS, "output_len": OUTPUT_LEN,
        "warmup_s": warm, "elapsed_s": elapsed, "tokens_per_s": tps,
    })

else:  # latency
    OUTPUT_LEN, WARMUP_REPS, REPS = 100, 3, 7
    params = SamplingParams(temperature=0.0, max_tokens=OUTPUT_LEN,
                            ignore_eos=True)

    def run_once():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        llm.generate([PROMPT], params)
        torch.cuda.synchronize()
        return time.perf_counter() - t0

    print(f"Warmup ({WARMUP_REPS} reps, discarded)...")
    for _ in range(WARMUP_REPS):
        run_once()

    print(f"Measuring {REPS} repeats...")
    tele.start()
    times = [run_once() for _ in range(REPS)]
    tele.stop()
    ms_per_tok = [t / OUTPUT_LEN * 1e3 for t in times]
    med = statistics.median(ms_per_tok)
    print("  repeats (ms total): " +
          " ".join(f"{t*1e3:.1f}" for t in times))
    print(f"  median: {med:.3f} ms/tok  ({1e3/med:.1f} tok/s)")
    result.update({
        "output_len": OUTPUT_LEN, "repeats_s": times,
        "ms_per_token": ms_per_tok, "median_ms_per_token": med,
        "tokens_per_s": 1e3 / med,
    })

result["telemetry"] = tele.summary()
with open(args.out, "w") as f:
    json.dump(result, f, indent=2)
print(f"Saved {args.out}")
print(f"RESULT {args.variant} {args.mode} "
      f"{result.get('tokens_per_s', 0):.2f} tok/s")
