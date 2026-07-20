# Supervision update — auto-tuned decode kernels in vLLM

**Research question (as it has converged):**
When and why do vendor GEMM libraries (cuBLAS) underperform for LLM inference,
and how can replacement kernels be auto-tuned *safely inside a production
engine* whose compile/capture pipeline forbids runtime tuning?

**Scope (deliberate):** decode phase only — B ≤ 32 token batches.
Decode GEMMs are memory-bandwidth-bound (arithmetic intensity ≈ B); prefill is
compute-bound and cuBLAS is near-roofline there. The gap we exploit only
exists in the bandwidth-bound regime.

---

# Method: load-time bucketed tuning, race-gated enablement

**Constraint analysis** (the core engineering-research finding):
- vLLM runs models under fullgraph `torch.compile` → kernel config frozen at
  trace time; graph breaks forbidden (forward-time tuning hard-crashes init)
- First decode-shaped calls execute inside CUDA-graph capture → benchmark
  syncs illegal
- ⇒ the **only safe tuning window is weight-load time**

**Mechanism:**
- Per-shape tuning (12-candidate pruned space) at `load_weights`, ~12 s/model
- Batch dim bucketed {16, 32} (token count is dynamic at serve time)
- Kernel enabled per shape **only if it beats cuBLAS by ≥2% at both buckets**
  (live race on the target GPU) → no model can regress, by construction
- Same code path: Qwen2.5-7B, Mistral-7B, Llama-3.1-8B — zero per-model work

---

# Results: three model families, same code (RTX 3090, bf16)

| Model | Throughput (32 conc.) | B=1 latency |
|---|---|---|
| Qwen2.5-7B | +5.3% (1580 → 1664 tok/s) | **+11.4%** (19.13 → 17.17 ms/tok) |
| Mistral-7B-v0.3 | **+10.1%** (1444 → 1589) | +6.7% |
| Llama-3.1-8B | +9.4% (1384 → 1515) | +6.4% |

- Biggest single-kernel win: Mistral GQA qkv [6144×4096] — **1.50× vs cuBLAS**
  (found automatically; hand-curated shape list had missed the class)
- Fused gate_up+SiLU kernel at **~92% of HBM bandwidth roofline** — little
  headroom left at the kernel level
- Measurement protocol: fresh process per config, thermal cooldown gate,
  1 s clock/power/temp telemetry, 7-repeat latency (spread ≤ 0.1%),
  established e2e noise floor ≈ 0.3%

---

# New this week: baseline vs PyTorch's compiler (Inductor max-autotune)

Head-to-head on 24 cases (6 GEMM + 2 fused shapes × B ∈ {1,16,32}),
`mode="max-autotune"`, both free backend choice and Triton-codegen-only:

| Contender | Geomean vs cuBLAS |
|---|---|
| **Ours (load-time tuned)** | **1.140×** |
| torch.compile max-autotune (best) | 1.126× |
| torch.compile max-autotune (Triton-only) | 1.135× |

- We win or tie **every** shape at B = 16/32 (up to +14.9%)
- **Inductor wins B=1 on 3 shapes by 6–11%** — it specialises per exact batch;
  our tuner reuses the bucket-16 config at B=1. Mechanism understood →
  directly motivates next experiment (B=1 bucket)
- Deployability: max-autotune can't run under vLLM's dynamic-shape compile +
  CUDA-graph capture — the constraint our method is built around. Its Triton
  templates also beating cuBLAS independently confirms the skinny-shape gap
  is real, not an artifact of our benchmark

---

# Negative results & threats to validity (kept deliberately)

- **Qwen lm_head correctly rejected** by the race (cuBLAS already ~92% of
  roof; +1.7–3% < enablement margin) — the gate does its job in both directions
- **Kernel-level wins ≠ e2e wins**: +2.6% micro improvement from the 256-config
  sweep was invisible at e2e (< 0.3% noise floor); at high concurrency,
  attention/KV-bandwidth dominates and GEMM gains dilute
- **Power-limited card**: ~2% of the throughput gain co-occurs with higher
  boost clocks (more efficient kernel → higher clocks — legitimate but worth
  stating); clock locking blocked (no sudo on the box)
- **External validity limits**: one GPU model (RTX 3090 / sm86), bf16
  unquantized only, generation-heavy workload. Prefill-heavy workloads
  (long-context RAG, classification) would see gains → 0. Scoping statement,
  not a flaw — but it belongs in the write-up

---

# Next steps & publication angle

**Immediate experiments:**
1. Add B=1 (or ≤4) tuning bucket — closes the one regime where the compiler
   wins; re-run B=1 latency e2e (could push Qwen's +11.4% higher)
2. Possible: fused down_proj + residual-add kernel (expected small)

**Write-up (mapping to KTP deliverables / paper):**
- Contribution candidates: (a) constraint analysis — where tuning *can* live
  in a compiled+captured serving engine; (b) race-gated, never-regress
  enablement; (c) evidence that vendor-library gaps concentrate in
  decode-shape GEMMs, confirmed independently by Inductor's own codegen
- Evaluation already in hand: 3 families, compiler baseline, negative results,
  telemetry-validated protocol
- Question for discussion: target venue framing — systems (kernel + engine
  integration) vs. empirical (when do compilers/libraries underperform)?
