#!/usr/bin/env python3
"""
discovery_lib.py — Artemis Discovery API wrapper.

Two transports:
  _cli_json(*args) — `artemis` CLI (build 0.1.0+, versions-based API).
  _http(...)       — direct Falcon HTTP for calls the CLI doesn't cover:
                       execute_baseline  (CLI sends no body → 400),
                       project metrics id-lookup (stable /projects/{id}/metrics),
                       LLM model presets (separate turintech-llm-api service),
                       agent-management messages (failure reason lookup).

Env: FALCON_URL, ARTEMIS_API_KEY (auto-loaded from ~/.config/artemis/.env).
The `artemis` CLI must be on PATH and configured (`artemis config setup` once).
"""
from __future__ import annotations
import json, os, re, subprocess, statistics as st, tempfile, time
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlencode
import urllib.request

ENV_FILE = Path.home() / ".config" / "artemis" / ".env"


# --------------------------------------------------------------------------- #
# Transport
# --------------------------------------------------------------------------- #

def _load_env() -> None:
    """Load FALCON_URL / ARTEMIS_API_KEY from ~/.config/artemis/.env if not set."""
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _http(method: str, path: str, payload=None, **params):
    """Direct Falcon HTTP call. Raises on non-2xx."""
    _load_env()
    url = os.environ["FALCON_URL"].rstrip("/") + "/" + path.lstrip("/")
    if params:
        url += "?" + urlencode(params)
    req = urllib.request.Request(url, method=method)
    req.add_header("Authorization", "Bearer " + os.environ["ARTEMIS_API_KEY"])
    data = None
    if payload is not None:
        data = json.dumps(payload).encode()
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, data) as r:
        return json.loads(r.read().decode())


def _cli_json(*args: str):
    """Run `artemis <args> --json` and return parsed output. Raises on non-zero exit."""
    out = subprocess.run(["artemis", *args, "--json"], capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"artemis {' '.join(args)} failed:\n{out.stderr.strip()}")
    return json.loads(out.stdout)


# --------------------------------------------------------------------------- #
# Discovery runs
# --------------------------------------------------------------------------- #

def create_run(project_id: str, task: str, versions: int, runner: str,
               compile_cmd: str = "", test_cmd: str = "", benchmark_cmd: str = "",
               model_id: str | None = None, mode: str = "automatic") -> dict:
    args = [
        "discovery", "create",
        "--project", project_id,
        "--task",    task,
        "--versions", str(versions),
        "--runner",  runner,
        "--mode",    mode,
    ]
    if compile_cmd:   args += ["--compile-cmd",   compile_cmd]
    if test_cmd:      args += ["--test-cmd",       test_cmd]
    if benchmark_cmd: args += ["--benchmark-cmd",  benchmark_cmd]
    if model_id:      args += ["--model",          model_id]
    return _cli_json(*args)


def get_run(run_id: str) -> dict:
    return _cli_json("discovery", "get", run_id)


def execute_baseline(run_id: str) -> dict:
    """Request a fresh baseline measurement for this run.
    Uses direct HTTP: CLI `discovery baseline execute` sends no body → 400."""
    return _http("POST", f"api/code/discovery/runs/{run_id}/baseline/execute", payload={})


def list_runs(project_id: str) -> list:
    d = _cli_json("discovery", "list", "--project", project_id)
    return d.get("docs", d) if isinstance(d, dict) else d


def run_failure_reason(run_id: str) -> str | None:
    """Return the Discovery agent's last explanatory message for a failed run."""
    try:
        agent_run_id = get_run(run_id).get("agentRunId")
        if not agent_run_id:
            return None
        _load_env()
        base = os.environ.get("AGENT_MGMT_URL")
        if not base:
            return None
        req = urllib.request.Request(
            base.rstrip("/") + f"/api/v1/agents/runs/{agent_run_id}/messages")
        req.add_header("Authorization", "Bearer " + os.environ["ARTEMIS_API_KEY"])
        docs = json.loads(urllib.request.urlopen(req).read().decode())
        docs = docs.get("docs", docs) if isinstance(docs, dict) else docs
        deltas = [m["payload"].get("delta", "").strip() for m in docs
                  if m.get("type") == "assistant.delta" and m.get("payload", {}).get("delta")]
        if not deltas:
            return None
        failure_keywords = ("failed", "error", "exited non-zero", "command", "not found", "no such")
        actionable = [d for d in deltas if any(k in d.lower() for k in failure_keywords)]
        return actionable[-1] if actionable else deltas[-1]
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Versions
# --------------------------------------------------------------------------- #

def list_versions(run_id: str, per_page: int = 200) -> list:
    rows, page = [], 1
    while True:
        d = _cli_json("discovery", "versions", "list", run_id,
                      "--per-page", str(per_page), "--page", str(page))
        docs = d.get("docs", d) if isinstance(d, dict) else d
        if not docs:
            break
        rows += docs
        if isinstance(d, dict) and not d.get("hasNextPage"):
            break
        if len(docs) < per_page:
            break
        page += 1
    return rows


def get_version(version_id: str) -> dict:
    return _cli_json("discovery", "versions", "get", version_id)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def metric_values(run_id: str, per_page: int = 1000) -> list:
    """All metric values for a run. Each row gains an `isBaseline` bool tag."""
    base_obs = str(get_run(run_id).get("baselineObservationId") or "")
    rows, page = [], 1
    while True:
        d = _cli_json("discovery", "metrics", run_id,
                      "--per-page", str(per_page), "--page", str(page))
        docs = d.get("docs", d) if isinstance(d, dict) else d
        if not docs:
            break
        for r in docs:
            r["isBaseline"] = bool(base_obs and str(r.get("observationId")) == base_obs)
            rows.append(r)
        if isinstance(d, dict) and not d.get("hasNextPage"):
            break
        if len(docs) < per_page:
            break
        page += 1
    return rows


def _find_metric_id(project_id: str, metric_name: str) -> str | None:
    """Resolve a metric name to its project-level UUID."""
    d = _http("GET", f"api/code/projects/{project_id}/metrics", page=1, perPage=500)
    docs = d.get("docs", d) if isinstance(d, dict) else d
    return next((m["id"] for m in docs if m.get("name") == metric_name), None)


def set_fitness_metric(run_id: str, project_id: str, metric_name: str,
                       higher_is_better: bool = True,
                       wait_secs: int = 0, poll: int = 20) -> None:
    """Attach a single-metric fitness schema so candidates receive scores.

    A metric only registers once a benchmark run has uploaded that key. Pass
    wait_secs > 0 to poll until the baseline registers it (new projects only)."""
    deadline_polls = max(1, wait_secs // poll) if wait_secs else 1
    metric_id = None
    for attempt in range(deadline_polls):
        metric_id = _find_metric_id(project_id, metric_name)
        if metric_id or attempt == deadline_polls - 1:
            break
        time.sleep(poll)
    if not metric_id:
        raise RuntimeError(
            f"Metric '{metric_name}' not found in project {project_id}. "
            "Run the benchmark at least once so the baseline uploads this key, "
            "then re-attach, or pass wait_secs to poll."
        )
    schema = [{"metricId": metric_id, "source": "worker",
               "higherIsBetter": higher_is_better, "importance": 1.0}]
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(schema, f)
        schema_path = f.name
    try:
        _cli_json("discovery", "metrics-schema", run_id, "--file", schema_path)
    finally:
        os.unlink(schema_path)


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #

def available_models() -> list[dict]:
    """All selectable models from the LLM-API presets catalogue."""
    _load_env()
    llm_base = os.environ.get("LLM_API_BASE_URL") or (
        os.environ["FALCON_URL"].split("/turintech-")[0] + "/turintech-llm-api"
    )
    req = urllib.request.Request(llm_base.rstrip("/") + "/v1/models/presets")
    req.add_header("Authorization", "Bearer " + os.environ["ARTEMIS_API_KEY"])
    with urllib.request.urlopen(req) as r:
        d = json.loads(r.read().decode())
    seen, models = set(), []
    for m in (d.get("rest") or []) + [d.get("lowEffort"), d.get("mediumEffort"), d.get("highEffort")]:
        if m and m.get("id") and m["id"] not in seen:
            seen.add(m["id"])
            models.append(m)
    return models


_UUID_RE = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$')


def resolve_model(name_or_uuid: str) -> dict:
    """Resolve a model name substring or full UUID to its catalogue entry.

    Accepts a UUID directly, or a case-insensitive substring of the display name /
    model id (e.g. 'flash', 'glm', 'sonnet', 'opus', 'gpt'). Raises on no match
    or ambiguous match."""
    models = available_models()
    if _UUID_RE.match(name_or_uuid):
        m = next((m for m in models if m["id"] == name_or_uuid), None)
        return m or {"id": name_or_uuid, "displayName": name_or_uuid, "modelId": name_or_uuid}
    q = name_or_uuid.lower()
    hits = [m for m in models
            if q in (m.get("displayName", "") + " " + m.get("modelId", "")).lower()]
    if not hits:
        avail = ", ".join(m.get("modelId", m["id"]) for m in models)
        raise RuntimeError(f"No model matches '{name_or_uuid}'. Available: {avail}")
    if len(hits) > 1:
        names = ", ".join(m.get("displayName") for m in hits)
        raise RuntimeError(f"'{name_or_uuid}' is ambiguous: {names}. Be more specific.")
    return hits[0]


# --------------------------------------------------------------------------- #
# Candidate code diff
# --------------------------------------------------------------------------- #

def version_patch_in_run(run_id: str, version_number: int | None = None) -> str:
    """Return the git diff for a Discovery candidate (highest-numbered by default)."""
    proj = get_run(run_id)["projectId"]
    vers = list_versions(run_id)
    if not vers:
        raise RuntimeError(f"no versions in run {run_id}")
    if version_number is None:
        v = max(vers, key=lambda v: v.get("versionNumber") or 0)
    else:
        v = next((v for v in vers if v.get("versionNumber") == version_number), None)
        if v is None:
            raise RuntimeError(f"version {version_number} not found in run {run_id}")
    cs, head = v["changesetId"], v["versionSha"]
    det = _http("GET", f"api/code/projects/{proj}/changesets/{cs}")
    base = det.get("baseVersionSha") or det.get("baseVersionSHA")
    _load_env()
    url = (os.environ["FALCON_URL"].rstrip("/") +
           f"/api/code/projects/{proj}/changesets/{cs}/{base}/diff/{head}/patch-file")
    req = urllib.request.Request(url)
    req.add_header("Authorization", "Bearer " + os.environ["ARTEMIS_API_KEY"])
    with urllib.request.urlopen(req) as r:
        return r.read().decode()


# --------------------------------------------------------------------------- #
# Analysis helpers
# --------------------------------------------------------------------------- #

def series(run_id: str, metric_substr: str) -> dict:
    """Per-(observationId, metricName) value lists for metrics matching metric_substr."""
    rows = [r for r in metric_values(run_id) if metric_substr in str(r.get("metricName", ""))]
    rows.sort(key=lambda r: str(r.get("createdAt", "")))
    out: dict = defaultdict(list)
    for r in rows:
        out[(str(r["observationId"]), str(r.get("metricName", "")))].append(float(r["value"]))
    return dict(out)


def metric_summary(run_id: str, metric_substr: str) -> dict:
    """Per-(observationId, metricName) stats: mean/std/ste/min/max/n/ci95/cv_pct."""
    res = {}
    for key, vals in series(run_id, metric_substr).items():
        n = len(vals)
        if n == 0:
            continue
        mean = st.mean(vals)
        sample_sd = st.stdev(vals) if n > 1 else 0.0
        ste = sample_sd / (n ** 0.5) if n > 1 else 0.0
        res[key] = dict(
            n=n, mean=mean, std=sample_sd, ste=ste,
            min=min(vals), max=max(vals),
            cv_pct=(st.pstdev(vals) / mean * 100) if mean else 0.0,
            ci95=(mean - 1.96 * ste, mean + 1.96 * ste),
        )
    return res


def _infer_higher_is_better(metric_name: str) -> bool:
    n = metric_name.lower()
    if any(t in n for t in ("tok_s", "tflop", "throughput", "req", "speedup")):
        return True
    if any(t in n for t in ("_ms", "runtime", "latency", "memory", "time")):
        return False
    return False


def ledger(run_id: str, metric_substr: str, higher_better: bool | None = None) -> dict:
    """Classify each candidate as WIN / REGRESSION / NEUTRAL / BROKE / NO-DATA
    relative to the run baseline, using 95% CI overlap tests.

    Returns dict(metric, baseline, higher_better, rows, carry_forward)."""
    summ = metric_summary(run_id, metric_substr)
    versions = list_versions(run_id)
    cand = []
    for v in versions:
        vd = get_version(v["id"])
        cand.append(dict(
            id=v["id"], gen=vd.get("generation"), num=vd.get("versionNumber"),
            obs=str(vd.get("observationId")), status=vd.get("executionStatus"),
            fitness=v.get("fitnessScore"), rationale=(vd.get("llmRationale") or "").strip(),
        ))
    cand_obs = {c["obs"] for c in cand}
    metric_name = next((n for (_, n) in summ), metric_substr)
    base_keys = [(o, n) for (o, n) in summ if o not in cand_obs and metric_substr in n]
    base = summ[base_keys[0]] if base_keys else None
    hib = _infer_higher_is_better(metric_name) if higher_better is None else higher_better

    rows = []
    for c in cand:
        key = next(((o, n) for (o, n) in summ if o == c["obs"] and metric_substr in n), None)
        s = summ.get(key) if key else None
        verdict, pct = "NO-DATA", None
        correct = (c["status"] == "success")
        if s and base:
            lo, hi = s["ci95"]
            bmean = base["mean"]
            pct = ((s["mean"] - bmean) / bmean * 100) * (1 if hib else -1)
            if (lo > bmean) if hib else (hi < bmean):
                verdict = "WIN"
            elif (hi < bmean) if hib else (lo > bmean):
                verdict = "REGRESSION"
            else:
                verdict = "NEUTRAL"
        if not correct:
            verdict = "BROKE"
        rows.append(dict(**c, verdict=verdict, pct=pct, mean=(s["mean"] if s else None)))

    rows.sort(key=lambda r: (r["pct"] is None, -(r["pct"] or -1e9)))
    return dict(
        metric=metric_name, baseline=(base["mean"] if base else None),
        higher_better=hib, rows=rows,
        carry_forward=dict(
            dead_ends=[r for r in rows if r["verdict"] in ("NEUTRAL", "REGRESSION")],
            promising=[r for r in rows if r["verdict"] == "WIN"],
        ),
    )
