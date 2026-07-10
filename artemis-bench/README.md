# Benchmark automation for Artemis Discovery

Two scripts that run Discovery campaigns on a list of **benchmark cases**
(one branch of a repo per case, named `benchmark/0/<name>`) and collect the
scores — Paul's "run benchmarks automatically" task.

Validated end-to-end 2026-06-12 on staging (runs `e907d2e2`, `770c1dc2`:
2 cases × 2 candidates, all built/tested/benchmarked/scored).

## Quick start

```bash
cd ~/Desktop/vllm/discovery-handover

# 1. Make sure the runner is alive (ideally inside tmux so it survives SSH drops):
#      tmux new -s runner
#      ~/artemis-tools/.venv/bin/artemis-runner

# 2. Launch all registered cases (cheap model, 2 versions each, serial):
python3 bench_launch.py --runner vishal

# 3. Watch until done and print the scores:
python3 bench_results.py --state campaign_state.json --wait
```

A 2-version campaign takes ~20 min per case.

## Common variations

```bash
python3 bench_launch.py --runner vishal --dry-run            # preview, touch nothing
python3 bench_launch.py --runner vishal --cases skinny-gemm  # specific case(s)
python3 bench_launch.py --runner vishal --versions 10        # bigger budget
python3 bench_launch.py --runner vishal --model high         # Sonnet 4.6 (real campaigns)
python3 bench_results.py --metric speedup                    # per-metric stats table
python3 bench_results.py --format json                       # machine-readable
```

`--model` takes `low` (glm-5-maas — default, cheap), `medium` (GPT-5.4 Mini),
`high` (Claude Sonnet 4.6), or an explicit model-catalogue UUID.

Each launch appends to `campaign_state.json`; already-launched cases are
skipped. Start a fresh campaign over the same cases with `--state <newfile>`.

## Adding a benchmark case

Cases live in **`cases.yaml`** (the registry — data, not code; the scripts
never change). To add one:

1. Push a branch containing: buildable code, a correctness test (exit 0 =
   pass), and a benchmark that writes `artemis_results.json` (flat JSON array,
   identical keys per object) at the repo root.
2. Add an entry to `cases.yaml` (copy an existing one): repo URL, branch,
   key-id, the three commands, a task prompt, and which result key is the
   fitness metric. (Use a different file with `--config <path>`.)
3. `python3 bench_launch.py --runner <runner> --cases <name> --dry-run`,
   then launch for real. After the first launch, pin the printed
   `project_id` into the entry to skip re-import.

The script is generic — it runs whatever cases are in the config. Onboarding a
case is config + a per-repo harness, never a code change.

## Files

| File | Role |
|---|---|
| `bench_launch.py` | case registry + campaign launcher (CLI import + direct Falcon API create) |
| `bench_results.py` | status / fitness / metrics tables (rich) |
| `discovery_lib.py` | shared layer: direct Falcon API + legacy CLI wrappers + patch retrieval |
| `campaign_state.json` | run IDs of the current campaign |
| `analysis/` | reference scripts for statistical analysis of results |
| `START-HERE.md`, `RUN-DISCOVERY-WITH-CLAUDE.md` | original handover kit docs — **partly outdated** (written for the old generations-based CLI) |
| `FALCON_API_REFERENCE.md` | Falcon API reference (still used for candidate code-diff retrieval) |

## Platform gotchas (staging, as of 2026-06-12)

- **Discovery API is versions-based** (`numVersions`, no generations×population)
  and **no released CLI speaks it** — that's why creation/status/metrics go
  through the direct API in `discovery_lib.py`. `artemis project import` and
  `discovery cancel` still work via the CLI.
- **Model list is admin-only** (`/v1/models` → 403); regular users only see
  the 3 presets (`/v1/models/presets`). Runs without an explicit model UUID
  fail in seconds.
- **One campaign at a time per runner** — a second concurrent run *fails*
  rather than queueing, so `bench_launch.py` is serial by default
  (`--no-serial` if you have multiple runners).
- Repos arrive on the runner as **zip extracts without `.git`** — compile
  commands must not depend on git metadata (vLLM needs
  `VLLM_VERSION_OVERRIDE=0.1.0`).
- The runner exports its own `VIRTUAL_ENV` — always pass an explicit target
  to `uv pip install` (`-p .venv`) and call `.venv/bin/python` directly
  (never `uv run`, which re-syncs the project).
- Full Falcon route list: `$FALCON_URL/openapi.json` (don't guess endpoints).
