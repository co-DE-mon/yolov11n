"""
Display model comparison results with rich formatting.

Usage:
    python scripts/display_results.py --dataset TXL
    python scripts/display_results.py --dataset TXL --file comparison_20251128_072755.csv
"""

import argparse
import csv
import re
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

PROJECT_ROOT = Path(__file__).parent.parent


def find_latest_csv(logs_dir: Path) -> Path | None:
    """Find the most recent comparison CSV file."""
    csv_files = sorted(logs_dir.glob("comparison_*.csv"), reverse=True)
    return csv_files[0] if csv_files else None


def load_results(csv_path: Path) -> list[dict]:
    """Load results from CSV file."""
    results = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                "batch": int(row["batch"]),
                "model_size": row.get("model_size", "n"),  # Default to "n" for backwards compatibility
                "model_family": row.get("model_family", "yolo11"),
                "prune_pct": float(row["prune_pct"]),
                "params_m": float(row["params_m"]),
                "gflops": float(row["gflops"]),
                "flops_reduction_pct": float(row["flops_reduction_pct"]),
                "pytorch_ms": float(row["pytorch_ms"]),
                "onnx_ms": float(row["onnx_ms"]),
                "pytorch_speedup": float(row["pytorch_speedup"]),
                "onnx_speedup": float(row["onnx_speedup"]),
            })
    return results


def load_metrics_from_runs(runs_dir: Path, dataset: str) -> dict:
    """
    Load mAP and other metrics from training results.csv files.
    Returns: {(model_size, batch, prune_pct): {precision, recall, mAP50, mAP50-95}}
    """
    pattern_baseline = re.compile(rf"{re.escape(dataset)}_([nslmx])_batch(\d+)_baseline$")
    pattern_pruned = re.compile(rf"{re.escape(dataset)}_([nslmx])_batch(\d+)_pruned(\d+)$")

    metrics = {}

    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir():
            continue

        results_csv = run_dir / "results.csv"
        if not results_csv.exists():
            continue

        name = run_dir.name

        # Parse model_size, batch and prune_pct from directory name
        if m := pattern_baseline.match(name):
            model_size = m.group(1)
            batch = int(m.group(2))
            prune_pct = 0.0
        elif m := pattern_pruned.match(name):
            model_size = m.group(1)
            batch = int(m.group(2))
            prune_pct = float(m.group(3))
        else:
            continue

        # Read the last epoch's metrics from results.csv
        with open(results_csv, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if not rows:
                continue

            # Get best metrics (last row typically, or we could find max mAP)
            last_row = rows[-1]

            # Handle column name variations (may have spaces)
            def get_col(row, *names):
                for n in names:
                    for key in row:
                        if n in key:
                            return float(row[key])
                return 0.0

            metrics[(model_size, batch, prune_pct)] = {
                "precision": get_col(last_row, "precision"),
                "recall": get_col(last_row, "recall"),
                "mAP50": get_col(last_row, "mAP50(B)", "mAP50"),
                "mAP50-95": get_col(last_row, "mAP50-95(B)", "mAP50-95"),
            }

    return metrics


def colorize_speedup(value: float) -> Text:
    """Return colored text based on speedup value."""
    text = f"{value:.2f}x"
    if value >= 1.5:
        return Text(text, style="bold green")
    elif value >= 1.1:
        return Text(text, style="green")
    elif value >= 0.95:
        return Text(text, style="yellow")
    else:
        return Text(text, style="red")


def colorize_reduction(value: float) -> Text:
    """Return colored text based on reduction percentage."""
    text = f"{value:.1f}%"
    if value <= -50:
        return Text(text, style="bold green")
    elif value <= -30:
        return Text(text, style="green")
    elif value <= -10:
        return Text(text, style="yellow")
    else:
        return Text(text, style="dim")


def colorize_map(value: float, baseline: float | None = None) -> Text:
    """Return colored text based on mAP value and drop from baseline."""
    text = f"{value:.3f}"
    if baseline is not None and baseline > 0:
        drop = (baseline - value) / baseline * 100
        if drop <= 1:
            return Text(text, style="bold green")
        elif drop <= 3:
            return Text(text, style="green")
        elif drop <= 5:
            return Text(text, style="yellow")
        else:
            return Text(text, style="red")
    return Text(text, style="white")


def display_results(results: list[dict], metrics: dict, dataset: str, console: Console):
    """Display results using rich tables."""

    # Group by model_size, then batch size
    model_sizes = {}
    for r in results:
        model_size = r["model_size"]
        batch = r["batch"]
        model_sizes.setdefault(model_size, {})
        model_sizes[model_size].setdefault(batch, []).append(r)

    # Header panel
    console.print()
    console.print(Panel(
        f"[bold cyan]{dataset}[/bold cyan] Model Comparison Results",
        box=box.DOUBLE,
        style="bold white"
    ))
    console.print()

    # Summary statistics
    if results:
        best_speedup_pt = max(r["pytorch_speedup"] for r in results)
        best_speedup_onnx = max(r["onnx_speedup"] for r in results)
        best_reduction = min(r["flops_reduction_pct"] for r in results)

        # Find best mAP among pruned models
        pruned_maps = [metrics.get((r["model_size"], r["batch"], r["prune_pct"]), {}).get("mAP50-95", 0)
                       for r in results if r["prune_pct"] > 0]
        best_pruned_map = max(pruned_maps) if pruned_maps else 0

        summary = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
        summary.add_column("Metric", style="dim")
        summary.add_column("Value", style="bold")

        summary.add_row("Best PyTorch Speedup", f"{best_speedup_pt:.2f}x")
        summary.add_row("Best ONNX Speedup", f"{best_speedup_onnx:.2f}x")
        summary.add_row("Max FLOPs Reduction", f"{best_reduction:.1f}%")
        if best_pruned_map > 0:
            summary.add_row("Best Pruned mAP50-95", f"{best_pruned_map:.3f}")

        console.print(Panel(summary, title="[bold]Summary", box=box.ROUNDED))
        console.print()

    # Table per model size and batch size
    for model_size in sorted(model_sizes.keys()):
        batches = model_sizes[model_size]

        # Print model size header
        if len(model_sizes) > 1:
            console.print(f"[bold yellow]Model: yolo11{model_size}[/bold yellow]")
            console.print()

        for batch_size in sorted(batches.keys()):
            batch_results = batches[batch_size]

            # Get baseline metrics for this model size and batch
            baseline_metrics = metrics.get((model_size, batch_size, 0.0), {})
            baseline_map50 = baseline_metrics.get("mAP50", 0)
            baseline_map50_95 = baseline_metrics.get("mAP50-95", 0)

            # Update title to include model size
            title = f"[bold magenta]Model {model_size} - Batch Size {batch_size}"

            table = Table(
                title=title,
                box=box.ROUNDED,
                show_header=True,
                header_style="bold cyan",
                title_justify="left",
            )

            table.add_column("Pruning", justify="right", style="bold")
            table.add_column("Params (M)", justify="right")
            table.add_column("FLOPs Δ", justify="right")
            table.add_column("mAP50", justify="right")
            table.add_column("mAP50-95", justify="right")
            table.add_column("Precision", justify="right")
            table.add_column("Recall", justify="right")
            table.add_column("PT Speedup", justify="right")
            table.add_column("ONNX Speedup", justify="right")

            # Add baseline row
            table.add_row(
                "0% (base)",
                "-",
                Text("-", style="dim"),
                f"{baseline_map50:.3f}" if baseline_map50 else "-",
                f"{baseline_map50_95:.3f}" if baseline_map50_95 else "-",
                f"{baseline_metrics.get('precision', 0):.3f}" if baseline_metrics.get('precision') else "-",
                f"{baseline_metrics.get('recall', 0):.3f}" if baseline_metrics.get('recall') else "-",
                Text("1.00x", style="dim"),
                Text("1.00x", style="dim"),
            )

            for r in sorted(batch_results, key=lambda x: x["prune_pct"]):
                m = metrics.get((r["model_size"], r["batch"], r["prune_pct"]), {})
                map50 = m.get("mAP50", 0)
                map50_95 = m.get("mAP50-95", 0)

                table.add_row(
                    f"{r['prune_pct']:.0f}%",
                    f"{r['params_m']:.2f}",
                    colorize_reduction(r["flops_reduction_pct"]),
                    colorize_map(map50, baseline_map50) if map50 else Text("-", style="dim"),
                    colorize_map(map50_95, baseline_map50_95) if map50_95 else Text("-", style="dim"),
                    f"{m.get('precision', 0):.3f}" if m.get('precision') else "-",
                    f"{m.get('recall', 0):.3f}" if m.get('recall') else "-",
                    colorize_speedup(r["pytorch_speedup"]),
                    colorize_speedup(r["onnx_speedup"]),
                )

            console.print(table)
            console.print()

    # Legend
    legend = Table(show_header=False, box=None, padding=(0, 1))
    legend.add_column()
    legend.add_column()
    legend.add_row(
        Text("Speedup: ", style="dim"),
        Text("≥1.5x", style="bold green") + Text(" | ", style="dim") +
        Text("≥1.1x", style="green") + Text(" | ", style="dim") +
        Text("≥0.95x", style="yellow") + Text(" | ", style="dim") +
        Text("<0.95x", style="red")
    )
    legend.add_row(
        Text("mAP drop: ", style="dim"),
        Text("≤1%", style="bold green") + Text(" | ", style="dim") +
        Text("≤3%", style="green") + Text(" | ", style="dim") +
        Text("≤5%", style="yellow") + Text(" | ", style="dim") +
        Text(">5%", style="red")
    )
    console.print(Panel(legend, title="[dim]Legend", box=box.SIMPLE))


def main():
    parser = argparse.ArgumentParser(description="Display model comparison results")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., TXL)")
    parser.add_argument("--file", type=str, help="Specific CSV file to load (filename only)")
    parser.add_argument("--logs-dir", type=Path, default=None, help="Logs directory")
    parser.add_argument("--runs-dir", type=Path, default=None, help="Runs directory for metrics")
    args = parser.parse_args()

    console = Console()

    # Determine directories
    logs_dir = args.logs_dir or (PROJECT_ROOT / "logs" / args.dataset)
    runs_dir = args.runs_dir or (PROJECT_ROOT / "runs")

    if not logs_dir.exists():
        console.print(f"[red]Error:[/red] Logs directory not found: {logs_dir}")
        console.print(f"[dim]Run 'python -m src.run_comparison --dataset {args.dataset}' first[/dim]")
        return 1

    # Find CSV file
    if args.file:
        csv_path = logs_dir / args.file
    else:
        csv_path = find_latest_csv(logs_dir)

    if not csv_path or not csv_path.exists():
        console.print(f"[red]Error:[/red] No comparison CSV found in {logs_dir}")
        return 1

    console.print(f"[dim]Loading: {csv_path.name}[/dim]")

    results = load_results(csv_path)

    if not results:
        console.print("[yellow]Warning:[/yellow] No results found in CSV")
        return 1

    # Load metrics from runs
    metrics = {}
    if runs_dir.exists():
        metrics = load_metrics_from_runs(runs_dir, args.dataset)
        console.print(f"[dim]Loaded metrics from {len(metrics)} runs[/dim]")

    display_results(results, metrics, args.dataset, console)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
