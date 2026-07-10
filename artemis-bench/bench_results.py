#!/usr/bin/env python3
"""
bench_results.py — summarise Artemis Discovery results from a campaign state file.

Usage:
    python bench_results.py [--state campaign_state.json]
    python bench_results.py --state campaign_state.json --metric speedup
    python bench_results.py --state campaign_state.json --wait
    python bench_results.py --state campaign_state.json --format json

The state file is written by bench_launch.py. Each campaign entry carries its
own fitness_metric and higher_is_better, so this script has no dependency on
cases.yaml or bench_launch.py.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from discovery_lib import (
    _load_env, get_run, list_versions, metric_summary, metric_values,
    set_fitness_metric, run_failure_reason,
)

try:
    from rich.console import Console
    from rich.table import Table
    from rich import box
except ImportError:
    sys.exit("rich required:  pip install rich")

console = Console()
TERMINAL = {"completed", "failed", "cancelled", "error"}
STATUS_STYLE = {
    "completed": "bold green", "running": "yellow", "failed": "bold red",
    "cancelled": "dim", "error": "bold red",
}


# --------------------------------------------------------------------------- #
# Per-campaign data collection
# --------------------------------------------------------------------------- #

def _collect(c: dict, metric: str) -> dict:
    run_id = c["run_id"]
    try:
        run = get_run(run_id)
        status = run.get("status", "unknown")
    except Exception as exc:
        return {**c, "status": "error", "error": str(exc),
                "best_fitness": None, "versions": [], "metrics": {}}

    try:
        versions = list_versions(run_id)
    except Exception:
        versions = []
    scores = [v["fitnessScore"] for v in versions if v.get("fitnessScore") is not None]

    fm = c.get("fitness_metric")
    hib = c.get("higher_is_better", True)

    # Self-heal: if a completed run has all-zero fitness, the schema was never
    # attached (first run on a new project). Attach it now and re-read.
    if (fm and status == "completed" and versions
            and all((v.get("fitnessScore") or 0) == 0 for v in versions)):
        try:
            set_fitness_metric(run_id, run["projectId"], fm, hib)
            versions = list_versions(run_id)
            scores = [v["fitnessScore"] for v in versions if v.get("fitnessScore") is not None]
        except Exception:
            pass

    metrics: dict = {}
    if metric:
        try:
            metrics = metric_summary(run_id, metric)
        except Exception as exc:
            metrics = {"_error": str(exc)}

    fm_values: dict = {}
    fm_baseline = None
    if fm and status in TERMINAL:
        try:
            for r in metric_values(run_id):
                if r["metricName"] == fm:
                    fm_values[str(r["observationId"])] = float(r["value"])
                    if r.get("isBaseline"):
                        fm_baseline = float(r["value"])
        except Exception:
            pass

    reason = run_failure_reason(run_id) if status == "failed" else None

    return {**c, "status": status,
            "best_fitness": max(scores) if scores else None,
            "versions": versions, "metrics": metrics, "reason": reason,
            "fm_values": fm_values, "fm_baseline": fm_baseline}


# --------------------------------------------------------------------------- #
# Rich output
# --------------------------------------------------------------------------- #

def _campaign_table(rows: list[dict]) -> Table:
    t = Table(title="Discovery campaigns", box=box.ROUNDED, title_justify="left")
    t.add_column("Case", style="bold cyan")
    t.add_column("Status")
    t.add_column("Fitness metric", style="dim")
    t.add_column("Best candidate", justify="right")
    t.add_column("Δ vs baseline", justify="right")
    t.add_column("Candidates", justify="right")
    t.add_column("Run ID", style="dim")
    for r in rows:
        status = r["status"]
        ok = sum(1 for v in r["versions"] if v.get("executionStatus") == "success")
        cands = f"{ok}/{len(r['versions'])}" if r["versions"] else "–"
        err = f" [red]({r['error'][:40]})[/red]" if r.get("error") else ""

        fm_values = r.get("fm_values") or {}
        version_obs = {str(v.get("observationId")) for v in r["versions"]}
        cand_vals = [fm_values[o] for o in version_obs if o in fm_values]
        baseline = r.get("fm_baseline") or next(
            (val for obs, val in fm_values.items() if obs not in version_obs), None)
        best = max(cand_vals) if cand_vals else None

        if best is not None and baseline:
            pct = (best - baseline) / baseline * 100
            style = "green" if pct > 0 else ("red" if pct < -0.01 else "dim")
            delta = f"[{style}]{pct:+.2f}%[/{style}]"
        else:
            delta = "[dim]–[/dim]"

        t.add_row(
            r["case"],
            f"[{STATUS_STYLE.get(status, 'white')}]{status}[/]{err}",
            r.get("fitness_metric") or "?",
            f"{best:.4f}" if best is not None else "[dim]–[/dim]",
            delta, cands, r["run_id"][:8] + "…",
        )
    return t


def _versions_table(r: dict) -> Table | None:
    if not r["versions"]:
        return None
    fm = r.get("fitness_metric")
    fm_values = r.get("fm_values") or {}
    version_obs = {str(v.get("observationId")) for v in r["versions"]}
    baseline = r.get("fm_baseline") or next(
        (val for obs, val in fm_values.items() if obs not in version_obs), None)

    t = Table(title=f"[bold]{r['case']}[/bold] — candidates",
              box=box.SIMPLE_HEAD, title_justify="left")
    t.add_column("#", justify="right")
    t.add_column("Execution")
    if fm:
        t.add_column(fm, justify="right")
        t.add_column("Δ vs baseline", justify="right")

    def delta(val):
        if val is None or baseline is None or baseline == 0:
            return "[dim]–[/dim]"
        pct = (val - baseline) / baseline * 100
        style = "green" if pct > 0 else ("red" if pct < -0.01 else "dim")
        return f"[{style}]{pct:+.2f}%[/{style}]"

    for v in sorted(r["versions"], key=lambda v: v.get("versionNumber") or 0):
        ok = v.get("executionStatus") == "success"
        row = [str(v.get("versionNumber", "?")),
               "[green]success[/green]" if ok else f"[red]{v.get('executionStatus')}[/red]"]
        if fm:
            val = fm_values.get(str(v.get("observationId")))
            row += [f"{val:.4f}" if val is not None else "[dim]–[/dim]", delta(val)]
        t.add_row(*row)
    if fm and baseline is not None:
        t.caption = f"baseline {fm} = {baseline:.4f}"
    return t


def _metrics_table(r: dict, metric: str) -> Table | None:
    m = r.get("metrics") or {}
    if "_error" in m:
        console.print(f"[red]metrics error for {r['case']}: {m['_error']}[/red]")
        return None
    if not m:
        return None
    labels = {str(v.get("observationId")): f"v{v.get('versionNumber')}"
              for v in r.get("versions", [])}
    t = Table(title=f"[bold]{r['case']}[/bold] — metrics matching '{metric}'",
              box=box.SIMPLE_HEAD, title_justify="left")
    t.add_column("Version", style="dim")
    t.add_column("Metric", style="cyan")
    t.add_column("n", justify="right")
    t.add_column("Mean", justify="right")
    t.add_column("95% CI", justify="right")
    t.add_column("CV %", justify="right")

    def sort_key(kv):
        (obs, name), _ = kv
        lab = labels.get(obs, "baseline")
        return (name, lab != "baseline", lab)

    for (obs, name), s in sorted(m.items(), key=sort_key):
        lo, hi = s["ci95"]
        lab = labels.get(obs, "[bold]baseline[/bold]")
        t.add_row(lab, name, str(s["n"]), f"{s['mean']:.4f}",
                  f"[{lo:.4f}, {hi:.4f}]", f"{s['cv_pct']:.2f}")
    return t


def _render(rows: list[dict], metric: str) -> None:
    console.print(_campaign_table(rows))
    for r in rows:
        if r.get("reason"):
            console.print(f"[red]✗ {r['case']}[/red] [dim]({r['run_id'][:8]}…):[/dim] {r['reason']}")
    for r in rows:
        vt = _versions_table(r)
        if vt:
            console.print(vt)
        if metric:
            mt = _metrics_table(r, metric)
            if mt:
                console.print(mt)


# --------------------------------------------------------------------------- #
# JSON output
# --------------------------------------------------------------------------- #

def _to_json_safe(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        r2 = dict(r)
        r2["metrics"] = {f"{o[:8]}:{n}": dict(v, ci95=list(v["ci95"]))
                         for (o, n), v in r.get("metrics", {}).items()
                         if isinstance(v, dict) and "ci95" in v}
        out.append(r2)
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--state",         default="campaign_state.json",
                   help="State file from bench_launch.py (default: campaign_state.json)")
    p.add_argument("--metric",        default="",
                   help="Metric name substring to show per-observation stats (e.g. 'speedup')")
    p.add_argument("--wait",          action="store_true",
                   help="Poll until all campaigns reach a terminal state")
    p.add_argument("--poll-interval", type=int, default=60,
                   help="Seconds between polls in --wait mode (default: 60)")
    p.add_argument("--format",        choices=["table", "json"], default="table")
    args = p.parse_args()

    state_path = Path(args.state)
    if not state_path.exists():
        sys.exit(f"State file not found: {args.state}\n"
                 "Run bench_launch.py first, or pass --state <path>.")

    _load_env()

    while True:
        campaigns = json.loads(state_path.read_text()).get("campaigns", [])
        if not campaigns:
            console.print("[yellow]No campaigns in state file.[/yellow]")
            break

        if args.format == "table" and args.wait:
            with console.status("[cyan]fetching run status…[/cyan]"):
                rows = [_collect(c, args.metric) for c in campaigns]
        else:
            rows = [_collect(c, args.metric) for c in campaigns]

        if args.format == "json":
            print(json.dumps(_to_json_safe(rows), indent=2, default=str))
        else:
            _render(rows, args.metric)

        pending = [r for r in rows if r["status"] not in TERMINAL]
        if not args.wait or not pending:
            if args.wait and not pending:
                console.print(f"\n[bold green]All {len(rows)} campaign(s) finished.[/bold green]")
            break

        console.print(f"\n[dim]{len(pending)}/{len(rows)} still running — "
                      f"next check in {args.poll_interval}s (Ctrl-C to stop)[/dim]\n")
        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
