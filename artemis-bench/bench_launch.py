#!/usr/bin/env python3
"""
bench_launch.py — launch Artemis Discovery campaigns on benchmark cases.

Each case is defined in cases.yaml: a repo/branch + compile/test/benchmark
commands + a fitness metric. The script imports the repo into Artemis (unless
already imported) and triggers a Discovery run.

Usage:
    # Run all cases in cases.yaml, or specific ones:
    python bench_launch.py --runner vishal
    python bench_launch.py --runner vishal --cases skinny-gemm --versions 10 --model sonnet

    # One-off repo via CLI args (no cases.yaml entry needed):
    python bench_launch.py --runner vishal \\
        --git-url https://github.com/org/repo.git --branch dev --key-id <KEY> \\
        --compile-cmd "make" --test-cmd "make test" --benchmark-cmd "./bench.sh" \\
        --fitness-metric speedup

    # Preview without touching Artemis:
    python bench_launch.py --runner vishal --dry-run

Run IDs are appended to --state (default: campaign_state.json). Re-running
with the same state file skips already-launched cases.

Afterwards:
    python bench_results.py --state campaign_state.json [--wait]
"""
from __future__ import annotations
import argparse, json, re, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("PyYAML required:  pip install pyyaml")

sys.path.insert(0, str(Path(__file__).parent))
from discovery_lib import create_run, get_run, execute_baseline, resolve_model, set_fitness_metric

try:
    from rich.console import Console
    from rich.table import Table
    from rich import box
except ImportError:
    sys.exit("rich required:  pip install rich")

console = Console()

DEFAULT_CASES_FILE = Path(__file__).parent / "cases.yaml"
UUID_RE = re.compile(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}')


def load_cases(path: Path) -> dict:
    if not path.exists():
        sys.exit(f"Cases file not found: {path}\nCreate it — see cases.yaml for the schema.")
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict) or not data:
        sys.exit(f"{path} is empty or malformed — expected a mapping of case-name → case.")
    return data


def _run(cmd: list[str], dry: bool) -> str:
    console.print(f"    [dim]$ {' '.join(cmd)}[/dim]")
    if dry:
        return "<dry-run>"
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{r.stderr.strip()}")
    return r.stdout.strip()


def _extract_uuid(text: str, exclude: set[str] | None = None) -> str | None:
    found = UUID_RE.findall(text)
    for u in reversed(found):
        if u not in (exclude or set()):
            return u
    return found[-1] if found else None


def import_project(case: dict, dry: bool) -> str:
    out = _run([
        "artemis", "project", "import",
        "--git-url", case["git_url"],
        "--branch",  case["branch"],
        "--key-id",  case["key_id"],
    ], dry)
    if dry:
        return "<dry-project-id>"
    pid = _extract_uuid(out)
    if not pid:
        raise RuntimeError(f"Could not parse project ID from output:\n{out}")
    return pid


def launch_discovery(project_id: str, case: dict, args: argparse.Namespace, dry: bool) -> str:
    task = args.task or case["task"]
    console.print(f"    [dim]artemis discovery create  versions={args.versions}[/dim]")
    if dry:
        return "<dry-run-id>"
    run = create_run(
        project_id=project_id, task=task, versions=args.versions,
        runner=args.runner, compile_cmd=case["compile_cmd"],
        test_cmd=case["test_cmd"], benchmark_cmd=case["benchmark_cmd"],
        model_id=args.model,
    )
    run_id = run.get("id")
    if not run_id:
        raise RuntimeError(f"No run ID in API response:\n{json.dumps(run)[:500]}")
    if args.execute_baseline:
        try:
            execute_baseline(run_id)
            console.print("    [dim]baseline execution requested[/dim]")
        except Exception as e:
            console.print(f"      [yellow]WARNING: baseline execute failed: {e}[/yellow]")
    if case.get("fitness_metric"):
        try:
            set_fitness_metric(run_id, project_id, case["fitness_metric"],
                               case.get("higher_is_better", True))
        except RuntimeError as e:
            console.print(f"      [yellow]WARNING: could not set fitness metric: {e}[/yellow]")
    return run_id


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config",  default=str(DEFAULT_CASES_FILE),
                   help="Benchmark-case registry YAML (default: cases.yaml)")
    p.add_argument("--cases",   nargs="+", default=None,
                   help="Case names to launch (default: all cases in config)")
    p.add_argument("--runner",  required=True, help="Artemis runner name")

    adhoc = p.add_argument_group(
        "ad-hoc case",
        "Define a one-off case inline instead of via --config. "
        "--git-url triggers this path; all four command flags are then required."
    )
    adhoc.add_argument("--git-url",        help="Repository to optimise")
    adhoc.add_argument("--branch",         default="main", help="Branch (default: main)")
    adhoc.add_argument("--key-id",         help="Artemis GitHub key ID for the repo")
    adhoc.add_argument("--project-id",     default=None,
                       help="Existing Artemis project UUID (skip re-import)")
    adhoc.add_argument("--compile-cmd",    help="Build command")
    adhoc.add_argument("--test-cmd",       help="Correctness test command")
    adhoc.add_argument("--benchmark-cmd",  help="Benchmark command (writes artemis_results.json)")
    adhoc.add_argument("--fitness-metric", default=None,
                       help="Key from artemis_results.json to use as the fitness score")

    p.add_argument("--model",    default="flash",
                   help="Model name substring ('flash', 'glm', 'sonnet', 'opus', 'gpt', 'pro') "
                        "or a catalogue UUID. Default: flash (cheap; good for exploration).")
    p.add_argument("--versions", type=int, default=2,
                   help="Candidate versions to generate per run (default: 2)")
    p.add_argument("--task",     default=None,
                   help="Discovery prompt override (default: each case's own task)")
    p.add_argument("--state",    default="campaign_state.json",
                   help="File run IDs are appended to (default: campaign_state.json)")
    p.add_argument("--dry-run",  action="store_true",
                   help="Print commands without executing them")
    p.add_argument("--no-serial", action="store_true",
                   help="Launch all cases at once. Default is serial (wait for each run to "
                        "finish first) because concurrent runs on a single runner fail rather "
                        "than queue.")
    p.add_argument("--execute-baseline", action="store_true",
                   help="Force each run to measure its own baseline (paired comparison). "
                        "Recommended for formal model comparisons; adds one benchmark execution.")
    args = p.parse_args()

    if args.git_url:
        missing = [flag for flag, val in [
            ("--key-id", args.key_id), ("--compile-cmd", args.compile_cmd),
            ("--test-cmd", args.test_cmd), ("--benchmark-cmd", args.benchmark_cmd),
        ] if not val]
        if missing:
            sys.exit(f"--git-url requires: {', '.join(missing)}")
        name = re.sub(r'\W+', '-', args.git_url.rstrip('/').split('/')[-1].replace('.git', ''))
        cases = {name: {
            "git_url": args.git_url, "branch": args.branch, "key_id": args.key_id,
            "project_id": args.project_id, "compile_cmd": args.compile_cmd,
            "test_cmd": args.test_cmd, "benchmark_cmd": args.benchmark_cmd,
            "task": args.task or f"Optimise {name} for performance.",
            "fitness_metric": args.fitness_metric, "higher_is_better": True,
        }}
        selected = [name]
        console.print(f"[dim]Ad-hoc case '{name}' from CLI args[/dim]")
    else:
        cases = load_cases(Path(args.config))
        selected = args.cases or sorted(cases)
        unknown = [n for n in selected if n not in cases]
        if unknown:
            sys.exit(f"Unknown case(s): {', '.join(unknown)}. "
                     f"Available: {', '.join(sorted(cases))}")

    m = resolve_model(args.model)
    model_name = m["displayName"]
    console.print(f"[bold]Model:[/bold] [cyan]{model_name}[/cyan] [dim]({m['id']})[/dim]")
    args.model = m["id"]

    state_path = Path(args.state)
    state = (json.loads(state_path.read_text()) if state_path.exists()
             else {"created_at": datetime.now(timezone.utc).isoformat(), "campaigns": []})
    already = {c["case"] for c in state["campaigns"]}

    results = []
    launched = skipped = failed = 0

    for name in selected:
        case = cases[name]
        if name in already:
            console.print(f"\n[bold cyan]{name}[/bold cyan]  "
                          f"[yellow]already in {args.state} — skipping[/yellow]")
            results.append((name, "skipped", next(
                c["run_id"] for c in state["campaigns"] if c["case"] == name)))
            skipped += 1
            continue

        console.print(f"\n[bold cyan]{name}[/bold cyan]  "
                      f"[dim]{case['git_url']} @ {case['branch']}[/dim]")
        try:
            project_id = case.get("project_id")
            if project_id:
                console.print(f"  project: [dim]{project_id}[/dim]")
            else:
                console.print("  importing into Artemis…")
                project_id = import_project(case, args.dry_run)
                console.print(f"  project: {project_id}  "
                              f"[dim](pin as project_id in {args.config})[/dim]")

            run_id = launch_discovery(project_id, case, args, args.dry_run)
            console.print(f"  [green]✓ launched[/green]  run [bold]{run_id}[/bold]")
        except Exception as e:
            console.print(f"  [bold red]✗ FAILED:[/bold red] {e}")
            results.append((name, "failed", "—"))
            failed += 1
            continue

        results.append((name, "launched", run_id))
        state["campaigns"].append({
            "case":             name,
            "git_url":          case["git_url"],
            "branch":           case["branch"],
            "project_id":       project_id,
            "run_id":           run_id,
            "model":            args.model,
            "versions":         args.versions,
            "task":             args.task or case["task"],
            "fitness_metric":   case.get("fitness_metric"),
            "higher_is_better": case.get("higher_is_better", True),
            "launched_at":      datetime.now(timezone.utc).isoformat(),
        })
        launched += 1
        if not args.dry_run:
            state_path.write_text(json.dumps(state, indent=2))

        remaining = selected[selected.index(name) + 1:]
        if remaining and not args.no_serial and not args.dry_run:
            with console.status(f"[cyan]waiting for {name} ({run_id[:8]}…) "
                                f"before launching {len(remaining)} more…[/cyan]"):
                status = "running"
                while status not in ("completed", "failed", "cancelled"):
                    time.sleep(60)
                    status = get_run(run_id).get("status")
            console.print(f"  [dim]{name} finished: {status}[/dim]")

    t = Table(box=box.ROUNDED, title="Launch summary", title_justify="left")
    t.add_column("Case", style="bold cyan")
    t.add_column("Outcome")
    t.add_column("Run ID", style="dim")
    style_map = {"launched": "[green]launched[/green]",
                 "skipped":  "[yellow]skipped[/yellow]",
                 "failed":   "[bold red]failed[/bold red]"}
    for name, outcome, run_id in results:
        t.add_row(name, style_map[outcome], run_id)
    console.print()
    console.print(t)
    console.print(f"[bold]{launched}[/bold] launched, {skipped} skipped, {failed} failed   "
                  f"[dim]model: {model_name}, versions: {args.versions}[/dim]")
    if launched and not args.dry_run:
        console.print(f"\nWatch progress:  [bold]python3 bench_results.py "
                      f"--state {args.state} --wait[/bold]")


if __name__ == "__main__":
    main()
