"""
Profile a single batch=1 decode step through Qwen2.5-7B-Instruct.
Outputs the top GPU kernels ranked by total CUDA time.

Usage:
  /home/vishal/Desktop/vllm/.venv/bin/python /home/vishal/vllm-kernel-opt/benchmarks/profile_decode.py

Requirements:
  - Model must be accessible (huggingface cache or local path)
  - Runs on CUDA device 0
  - Does NOT require the vLLM server to be running
"""

import os, sys, collections
import torch
from torch.profiler import profile, record_function, ProfilerActivity

MODEL_ID   = "Qwen/Qwen2.5-7B-Instruct"
DEVICE     = "cuda:0"
DTYPE      = torch.bfloat16
WARMUP     = 5    # decode steps before profiling
PROFILE    = 10   # decode steps to profile
TOP_N      = 20   # number of kernels to report

print(f"Loading {MODEL_ID} ...")
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model     = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=DTYPE,
    device_map=DEVICE,
    trust_remote_code=True,
).eval()

print(f"Model loaded on {DEVICE}  ({DTYPE})\n")

# ── Build a single-token decode context ──────────────────────────────────────
# Start with a short prompt, then cache the KV, then profile decode-only steps
prompt = "Hello, how are you?"
inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

with torch.no_grad():
    # Prefill — build KV cache
    out       = model(**inputs, use_cache=True)
    past_kv   = out.past_key_values
    next_tok  = out.logits[:, -1:, :].argmax(-1)   # [1, 1]

print("KV cache built. Running warmup decode steps ...")

# ── Warmup ───────────────────────────────────────────────────────────────────
with torch.no_grad():
    for _ in range(WARMUP):
        out2     = model(input_ids=next_tok,
                         past_key_values=past_kv,
                         use_cache=True)
        past_kv  = out2.past_key_values
        next_tok = out2.logits[:, -1:, :].argmax(-1)

torch.cuda.synchronize()
print(f"Warmup done. Profiling {PROFILE} decode steps ...\n")

# ── Profile ──────────────────────────────────────────────────────────────────
with torch.no_grad():
    with profile(
        activities=[ProfilerActivity.CUDA],
        record_shapes=False,
        with_stack=False,
    ) as prof:
        for _ in range(PROFILE):
            with record_function("decode_step"):
                out2     = model(input_ids=next_tok,
                                 past_key_values=past_kv,
                                 use_cache=True)
                past_kv  = out2.past_key_values
                next_tok = out2.logits[:, -1:, :].argmax(-1)
        torch.cuda.synchronize()

# ── Aggregate kernel times ────────────────────────────────────────────────────
kernel_times = collections.defaultdict(lambda: {"count": 0, "us": 0.0})
for evt in prof.events():
    if evt.device_type == torch.autograd.DeviceType.CUDA:
        name = evt.name
        kernel_times[name]["count"] += 1
        kernel_times[name]["us"]    += evt.cuda_time   # microseconds

total_us   = sum(v["us"] for v in kernel_times.values())
sorted_k   = sorted(kernel_times.items(), key=lambda x: x[1]["us"], reverse=True)

print(f"{'Rank':<5} {'Kernel':<65} {'Total (us)':>12} {'%GPU':>7} {'Calls':>7}")
print("-" * 100)
for rank, (name, info) in enumerate(sorted_k[:TOP_N], 1):
    pct = 100.0 * info["us"] / total_us if total_us else 0
    print(f"{rank:<5} {name[:65]:<65} {info['us']:>12.1f} {pct:>6.1f}% {info['count']:>7}")

print()
print(f"Total profiled GPU time: {total_us/1000:.2f} ms over {PROFILE} steps")
print(f"Average per decode step: {total_us/PROFILE/1000:.3f} ms")
