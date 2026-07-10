#!/usr/bin/env python3
"""
launch.py — run an Artemis Discovery campaign on any repo.

    python launch.py --git-url https://github.com/org/repo.git \\
                     --task    "Optimise the kernel" \\
                     --runner  my-runner

The tool calls scripts/artemis_compile.sh, scripts/artemis_test.sh,
scripts/artemis_bench.sh in your repo by default — add those scripts once
and you never pass build commands again. Override with --compile-cmd etc.
only when your repo uses different paths.
"""
from __future__ import annotations

import argparse, re, subprocess, sys, time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from discovery_lib import (
    _load_env, _http, create_run, get_run, list_versions,
    metric_values, set_fitness_metric, resolve_model, run_failure_reason,
)

try:
    from rich.console import Console
    from rich.table import Table
    from rich import box
except ImportError:
    sys.exit("pip install rich")

console = Console()
_UUID    = re.compile(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}')
TERMINAL = frozenset({"completed", "failed", "cancelled", "error"})


# ── Data ───────────────────────────────────────────────────────────────────────

@dataclass
class RunConfig:
    git_url:          str
    task:             str
    runner:           str
    branch:           str  = "main"
    key_id:           str  = ""
    fitness_metric:   str  = ""
    higher_is_better: bool = True
    versions:         int  = 2
    model:            str  = "flash"
    compile_cmd:      str  = "bash scripts/artemis_compile.sh"
    test_cmd:         str  = "bash scripts/artemis_test.sh"
    benchmark_cmd:    str  = "bash scripts/artemis_bench.sh"


# ── API ────────────────────────────────────────────────────────────────────────

class ArtemisClient:
    """Project find/import + model resolution on top of discovery_lib."""

    def find_project(self, git_url: str, branch: str) -> str | None:
        try:
            docs = _http("GET", "api/code/projects", page=1, perPage=200)
            docs = docs.get("docs", docs) if isinstance(docs, dict) else docs
            return next(
                (p["id"] for p in docs
                 if p.get("gitUrl") == git_url and p.get("gitBranch") == branch),
                None,
            )
        except Exception:
            return None

    def import_project(self, git_url: str, branch: str, key_id: str) -> str:
        if not key_id:
            raise ValueError("project not imported yet — pass --key-id "
                             "(get yours: artemis key list)")
        r = subprocess.run(
            ["artemis", "project", "import",
             "--git-url", git_url, "--branch", branch, "--key-id", key_id],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            raise RuntimeError(r.stderr.strip() or r.stdout.strip())
        uuids = _UUID.findall(r.stdout)
        if not uuids:
            raise RuntimeError(f"import succeeded but no UUID in output:\n{r.stdout}")
        return uuids[-1]

    def resolve_project(self, cfg: RunConfig) -> str:
        pid = self.find_project(cfg.git_url, cfg.branch)
        if pid:
            console.print(f"  project  [dim]{pid}[/dim]")
            return pid
        console.print("  [dim]project not found — importing…[/dim]")
        pid = self.import_project(cfg.git_url, cfg.branch, cfg.key_id)
        console.print(f"  project  [dim]{pid}[/dim]  [dim](imported)[/dim]")
        return pid

    def resolve_model(self, name_or_uuid: str) -> tuple[str, str]:
        m = resolve_model(name_or_uuid)
        return m["id"], m.get("displayName", name_or_uuid)


# ── Orchestration ──────────────────────────────────────────────────────────────

class DiscoveryRunner:

    def __init__(self, client: ArtemisClient):
        self.client = client

    def run(self, cfg: RunConfig) -> None:
        _load_env()
        console.rule("[bold cyan]Artemis Discovery[/bold cyan]")

        project_id, run_id = self._launch(cfg)
        status = self._poll(run_id)

        console.print()
        if status == "completed":
            console.print(f"[bold green]✓[/bold green] run [bold]{run_id}[/bold] completed")
            self._show_results(run_id, project_id, cfg)
        else:
            reason = run_failure_reason(run_id)
            console.print(f"[bold red]✗[/bold red] run [bold]{run_id}[/bold] — {status}")
            if reason:
                console.print(f"  [dim]{reason}[/dim]")

    def _launch(self, cfg: RunConfig) -> tuple[str, str]:
        project_id    = self.client.resolve_project(cfg)
        model_id, name = self.client.resolve_model(cfg.model)
        console.print(f"  model    [dim]{name}[/dim]")

        run = create_run(
            project_id=project_id, task=cfg.task, versions=cfg.versions,
            runner=cfg.runner, model_id=model_id,
            compile_cmd=cfg.compile_cmd, test_cmd=cfg.test_cmd,
            benchmark_cmd=cfg.benchmark_cmd,
        )
        run_id = run["id"]
        console.print(f"  run      [bold]{run_id}[/bold]  "
                      f"[dim](versions={cfg.versions})[/dim]")

        if cfg.fitness_metric:
            try:
                set_fitness_metric(run_id, project_id,
                                   cfg.fitness_metric, cfg.higher_is_better)
                console.print(f"  fitness  [dim]{cfg.fitness_metric}[/dim]")
            except RuntimeError:
                console.print(f"  [yellow]fitness metric '{cfg.fitness_metric}' will attach "
                              "after first benchmark upload[/yellow]")

        return project_id, run_id

    def _poll(self, run_id: str, interval: int = 30) -> str:
        console.print()
        with console.status("[cyan]waiting for run to finish…[/cyan]") as spinner:
            while True:
                state = get_run(run_id).get("status", "unknown")
                spinner.update(f"[cyan]{state}[/cyan]  "
                               f"[dim]artemis discovery get {run_id}[/dim]")
                if state in TERMINAL:
                    return state
                time.sleep(interval)

    def _show_results(self, run_id: str, project_id: str, cfg: RunConfig) -> None:
        if not cfg.fitness_metric:
            return

        # Lazily attach fitness metric if it was deferred (first run)
        try:
            set_fitness_metric(run_id, project_id,
                               cfg.fitness_metric, cfg.higher_is_better)
        except RuntimeError:
            pass

        run      = get_run(run_id)
        base_obs = str(run.get("baselineObservationId") or "")
        versions = list_versions(run_id)
        scores   = {str(r["observationId"]): float(r["value"])
                    for r in metric_values(run_id)
                    if r.get("metricName") == cfg.fitness_metric}

        baseline = scores.get(base_obs)

        t = Table(box=box.SIMPLE_HEAD,
                  title=f"[bold]{cfg.fitness_metric}[/bold]",
                  title_justify="left")
        t.add_column("version")
        t.add_column("score",      justify="right")
        t.add_column("Δ baseline", justify="right")

        def _delta(val: float) -> str:
            if not baseline:
                return "–"
            pct   = (val - baseline) / baseline * 100
            color = "green" if pct > 0.0 else ("red" if pct < -0.01 else "dim")
            return f"[{color}]{pct:+.2f}%[/{color}]"

        if baseline is not None:
            t.add_row("[bold]baseline[/bold]", f"{baseline:.4f}", "")

        best: tuple[float, str] | None = None
        for v in sorted(versions, key=lambda v: v.get("versionNumber") or 0):
            obs = str(v.get("observationId", ""))
            if obs == base_obs:
                continue
            val = scores.get(obs)
            num = v.get("versionNumber", "?")
            ok  = v.get("executionStatus") == "success"
            lbl = f"v{num}" if ok else f"[red]v{num}[/red] [dim]({v.get('executionStatus')})[/dim]"
            t.add_row(lbl,
                      f"{val:.4f}" if val is not None else "[dim]–[/dim]",
                      _delta(val)  if val is not None else "–")
            if val is not None and baseline:
                pct = (val - baseline) / baseline * 100
                if best is None or pct > best[0]:
                    best = (pct, f"v{num}")

        console.print(t)
        if best:
            color = "green" if best[0] > 0 else "red"
            console.print(f"best  {best[1]}  [{color}]{best[0]:+.2f}% vs baseline[/{color}]")


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--git-url",          required=True)
    p.add_argument("--task",             required=True)
    p.add_argument("--runner",           required=True)
    p.add_argument("--branch",           default="main")
    p.add_argument("--key-id",           default="",
                   help="Required only if project is not yet imported "
                        "(artemis key list)")
    p.add_argument("--fitness-metric",   default="",
                   help="Key written to artemis_results.json")
    p.add_argument("--higher-is-better", default=True, type=lambda v: v.lower() != "false")
    p.add_argument("--versions",         default=2, type=int)
    p.add_argument("--model",            default="flash",
                   help="Name substring or UUID — flash|glm|sonnet|opus|gpt|pro")
    p.add_argument("--compile-cmd",   default="bash scripts/artemis_compile.sh",
                   help="Override the repo's build script")
    p.add_argument("--test-cmd",      default="bash scripts/artemis_test.sh",
                   help="Override the repo's test script")
    p.add_argument("--benchmark-cmd", default="bash scripts/artemis_bench.sh",
                   help="Override the repo's benchmark script")
    return p


def main() -> None:
    args = _parser().parse_args()
    cfg  = RunConfig(
        git_url          = args.git_url,
        task             = args.task,
        runner           = args.runner,
        branch           = args.branch,
        key_id           = args.key_id,
        fitness_metric   = args.fitness_metric,
        higher_is_better = args.higher_is_better,
        versions         = args.versions,
        model            = args.model,
        compile_cmd      = args.compile_cmd,
        test_cmd         = args.test_cmd,
        benchmark_cmd    = args.benchmark_cmd,
    )
    DiscoveryRunner(ArtemisClient()).run(cfg)


if __name__ == "__main__":
    main()
