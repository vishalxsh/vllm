# KTP Update — Presentation Script

Simple, spoken-style talking points for each slide, plus answers to likely follow-up questions.

---

## Slide 1 — What this project is

This project is about making AI models run faster automatically, using a tool called Artemis. The idea is: build one method that works on any AI model, not just one. To do that, we ask four questions every time — is something slow? which part exactly, and why? how do we fix it? and can we actually build that fix? We proved this out on vLLM running Qwen — a 7-billion-parameter model — on one graphics card. The end goal is: the same method should work on any model, without us doing manual work each time.

## Slide 2 — Finding the problem

First, we built a tool that watches the model while it runs and tells us exactly which piece of code is eating up the most time. We ran it on Qwen while it was generating text, and found three specific calculations — big matrix multiplications — that were the slowest part. The reason they were slow: NVIDIA's standard library, cuBLAS, is built for big, evenly-shaped multiplications. Ours were oddly shaped — tiny on one side, huge on the other — and cuBLAS just wasn't tuned for that shape.

## Slide 3 — First fix

So we wrote our own version of that calculation, using a language called Triton, built specifically for these odd shapes. Then instead of manually trying different settings, we let Artemis's AI try about 31 different versions automatically and measure which was fastest — each one tested 7 times so we could trust the numbers, not just get lucky once. That got us about 8.8% faster on average at the calculation level, and translated to almost 5% more throughput overall.

## Slide 4 — What we found next (and what SiLU is)

At this point we asked: is that everything, or is there more? So we looked closer at what happens in one full step of generating a token.

**Quick explainer — what is SiLU:** after that big multiplication happens, the model runs a small extra step called an activation function — ours is called SiLU. Think of it like a filter or a gate — it looks at the numbers that just came out of the multiplication and adjusts them before passing them to the next part of the model. Every AI model does this, and it happens millions of times.

Here's what we found: the multiplication step and the SiLU step were being run separately — the result of the multiplication gets written down into memory, and then immediately picked back up and re-read for the SiLU step. That's like writing something on a piece of paper and immediately picking it back up to read it, instead of just keeping it in your head. That's wasted effort, and it happens on every layer, for every single word the model generates.

We also realized something bigger: on this hardware, generating each word is mostly limited by how fast we can pull the model's data out of memory — that's the real speed limit we're up against.

**Explaining the diagram:** this picture shows exactly that. On top, the old way — three separate boxes: the multiplication (GEMM), then a step where it writes the result to memory (that's the orange box — that's the wasted step, not useful work, just overhead), then SiLU reads it back. On the bottom, the new way — just one box. We combined the multiplication and the SiLU filter into a single operation, so the result never has to leave fast memory and get written down at all. One step instead of three.

## Slide 5 — How we fixed it

So we merged those two operations into one — the multiplication and the filter happen together, and the in-between result stays inside the chip the whole time, never touching slower memory. We also turned on a feature called CUDA Graphs, which cuts out some extra overhead every time the GPU launches a new task. Then we automatically tried 256 different tuning settings for this new fused version and kept the best one. That got us to about 92% of the theoretical speed limit for this hardware — there's very little room left to improve beyond that.

## Slide 6 — Results after fusion

Here are the real before-and-after numbers on Qwen: generating a single response got 8.7% faster, and serving 32 people at once got 3% more throughput. And this wasn't a lucky one-off measurement — we ran it fresh each time, made sure the GPU wasn't overheating, and tracked power and clock speed the whole time to make sure nothing was skewing the result.

## Slide 7 — Making it work on any model

Up to now, our kernel only knew about Qwen's specific shapes — it was hardcoded. So we rebuilt it to tune itself automatically: the first time it sees a new model, it takes about 12 seconds to test itself against the standard approach for every shape, and it only switches to our faster version where it actually wins by at least 2%. That means it can never make things worse — worst case, it just falls back to the normal approach. As a nice bonus, this automatic testing found one calculation our own hand-picked list had actually missed. Now the same code works across Qwen, Llama, and Mistral — three different model families — without us touching anything model-specific.

## Slide 8 — Results across three models

We ran the same test on three different models. Qwen improved by about 5% throughput and 11% latency. Mistral improved by 10% throughput and almost 7% latency. Llama improved by about 9% throughput and 6% latency. The important part: for Mistral and Llama, we didn't do any extra manual work — the self-tuning did everything on its own. And one specific calculation on Mistral ended up 1.5 times faster than the standard approach — our biggest single win.

## Slide 9 — The benchmark tool

Separately, we built a tool so all of this can be repeated easily, by anyone, on any codebase — not just this one. You describe a project in one simple file — the repo, three commands (build it, test it, benchmark it), and what number counts as "better" — and the tool handles everything else: uploading it, asking Artemis's AI to try to improve it, and showing you the results with proper statistics. Adding something new to test means editing that one file, never writing new code. It's fully working end-to-end on our vLLM case right now.

## Slide 10 — Summary and what's next

So, against our original plan: we built and proved the profiling tool, we built and benchmarked the optimized kernel, and we proved the method generalizes across three different models — all real milestones hit. Next, we're going to write up a full evaluation report, package everything so anyone can reproduce it with one command, run an internal workshop to share what we learned, and we're aiming to publish this as a research paper.

---

## Anticipated Q&A

### Q: Why did the other models (Mistral, Llama) beat Qwen?

They didn't beat it across the board — it's a trade-off. Mistral and Llama win on **throughput**, but Qwen still wins on **latency**.

| Model | Throughput | Latency |
|---|---|---|
| Qwen2.5-7B | +5.3% | **+11.4%** (best) |
| Mistral-7B | **+10.1%** (best) | +6.7% |
| Llama-3.1-8B | **+9.4%** | +6.4% |

**Qwen wins on latency because it got the most hand-tuning.** Qwen was our original test case — we spent months on it specifically, running a 31-version automatic search *and* a 256-setting sweep, all focused on making single-request speed as fast as possible. So Qwen's kernels are the most polished for that one specific thing.

**Mistral and Llama win on throughput because of a lucky shape, not more effort.** They got zero manual attention — the system tuned itself automatically in about 12 seconds when it first saw them. But it turns out Mistral's attention calculation (something called "grouped query attention") produces an oddly-shaped multiplication that NVIDIA's standard library is especially bad at — worse than the shapes in Qwen. So there was more "free" speed sitting on the table for our tool to grab. That single shape ended up being our biggest win in the whole project — 1.5 times faster than normal.

**Simple takeaway to say out loud:** "Qwen got the most manual polish, so it wins on single-user speed. But Mistral and Llama, with zero manual work, still found a bigger overall win — because they happened to have a weak spot in the standard library that our automatic system was able to exploit. That's actually a strong result for us: it shows the automatic approach can find wins we didn't even go looking for, sometimes bigger than the ones we spent months hand-tuning."

### Q: Are we running the model online? What does "latency" mean here?

No — this isn't a live public service with real internet users. It's a controlled test on our own machine (a research GPU box, not a public server anyone can reach).

vLLM is the same kind of software that powers things like ChatGPT — it takes a text prompt and generates a response, word by word. We run it ourselves, on our own GPU, and send it fake test requests (from the same machine, not real internet traffic) to measure how fast it responds.

**"Latency" specifically means: how long does it take to generate one word (technically, one "token") of the response, when there's only one request happening at a time.** It's measured in milliseconds per token — e.g., "17.6 ms per word." Think about watching ChatGPT type out an answer on screen, word by word — that speed you see is exactly what this number represents. Lower latency = each word appears faster.

**"Throughput," by contrast, is: how many total words per second the system can produce if 32 people are all using it at the same time.** That's overall capacity when the system is busy serving a crowd, not how fast any one person's text appears.

**Simple takeaway:** we're not measuring real-world internet delay (like "how long did the website take to load"). We're measuring the AI's own raw generation speed, on our hardware, in a controlled repeatable test — as a stand-in for what a real user would experience if this were actually deployed as a live service.
