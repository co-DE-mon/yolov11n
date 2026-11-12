"""
Compare all trained models: discover runs, benchmark, generate tables.

Usage:
    python -m src.run_comparison --dataset TXL
    python -m src.run_comparison --dataset TXL --runs-dir runs
"""

import argparse
import csv
import re
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).parent.parent


# Note: ONNX models must be pre-exported using: python -m src.run_export --dataset <name>


def discover_models(runs_dir: Path, dataset: str) -> dict:
    """
    Find all finetuned and pruned models for a specific dataset.
    Returns: {batch_size: {'baseline': Path, 'pruned': {pct: Path, ...}}}
    """
    # Match: {dataset}_batch{N}_baseline
    pattern_baseline = re.compile(rf"{re.escape(dataset)}_batch(\d+)_baseline$")
    # Match: {dataset}_batch{N}_pruned{pct}
    pattern_pruned = re.compile(rf"{re.escape(dataset)}_batch(\d+)_pruned(\d+)$")

    result = {}
    for child in runs_dir.iterdir():
        if not child.is_dir():
            continue

        weights = child / "weights" / "best.pt"
        if not weights.exists():
            continue

        name = child.name

        if m := pattern_baseline.match(name):
            batch = int(m.group(1))
            result.setdefault(batch, {"baseline": None, "pruned": {}})
            result[batch]["baseline"] = weights

        elif m := pattern_pruned.match(name):
            batch = int(m.group(1))
            pct = float(m.group(2))
            result.setdefault(batch, {"baseline": None, "pruned": {}})
            result[batch]["pruned"][pct] = weights

    return result


def get_model_stats(model_path: Path, imgsz: int = 640) -> tuple[float, float]:
    """Get (params_millions, gflops) for a model."""
    model = YOLO(str(model_path)).model
    params_m = sum(p.numel() for p in model.parameters()) / 1e6

    # Simple FLOPs estimation via forward hooks
    flops = [0]

    def hook_conv(m, inp, out):
        if isinstance(out, torch.Tensor):
            b, c_out, h, w = out.shape
            flops[0] += m.kernel_size[0] * m.kernel_size[1] * (m.in_channels / m.groups) * c_out * h * w

    hooks = []
    for m in model.modules():
        if m.__class__.__name__ == "Conv2d":
            hooks.append(m.register_forward_hook(hook_conv))

    with torch.inference_mode():
        model(torch.randn(1, 3, imgsz, imgsz))

    for h in hooks:
        h.remove()

    return params_m, flops[0] / 1e9


def benchmark_pytorch(model_path: Path, imgsz: int, batch: int, iters: int, device: str) -> float:
    """Returns avg ms per batch."""
    model = YOLO(str(model_path)).model.to(device).eval()
    inp = torch.randn(batch, 3, imgsz, imgsz, device=device)

    with torch.inference_mode():
        for _ in range(10):
            model(inp)

    if device.startswith("cuda"):
        torch.cuda.synchronize()

    t0 = perf_counter()
    with torch.inference_mode():
        for _ in range(iters):
            model(inp)

    if device.startswith("cuda"):
        torch.cuda.synchronize()

    return (perf_counter() - t0) * 1000 / iters


def benchmark_onnx(model_path: Path, imgsz: int, batch: int, iters: int) -> tuple[float, str]:
    """Benchmark pre-exported ONNX model. Returns (avg ms per image, execution provider actually used)."""
    import onnxruntime as ort

    # Look for pre-exported ONNX file
    onnx_path = model_path.with_suffix(".onnx")
    if not onnx_path.exists():
        print(f"    [WARN] ONNX not found: {onnx_path.name}, run 'python -m src.run_export' first")
        return float("nan"), "none"

    providers = []
    if "CUDAExecutionProvider" in ort.get_available_providers():
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")

    session = ort.InferenceSession(str(onnx_path), providers=providers)
    # ORT silently falls back to CPU if CUDA fails to initialize, so report what the session really uses
    provider = session.get_providers()[0]
    input_name = session.get_inputs()[0].name
    inp = np.random.randn(batch, 3, imgsz, imgsz).astype(np.float32)

    for _ in range(10):
        session.run(None, {input_name: inp})

    t0 = perf_counter()
    for _ in range(iters):
        session.run(None, {input_name: inp})

    return (perf_counter() - t0) * 1000 / iters / batch, provider


def main():
    parser = argparse.ArgumentParser(description="Compare all models and generate results")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., TXL)")
    parser.add_argument("--runs-dir", type=Path, default=PROJECT_ROOT / "runs")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Output to logs/{dataset}/
    output_dir = PROJECT_ROOT / "logs" / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"=== Model Comparison ===")
    print(f"Dataset: {args.dataset}")
    print(f"Discovering models in {args.runs_dir}...")

    models = discover_models(args.runs_dir, args.dataset)
    if not models:
        print("No models found.")
        return

    print(f"Found batch sizes: {sorted(models.keys())}")

    results = []
    md_tables = []

    for batch_size in sorted(models.keys()):
        info = models[batch_size]
        baseline = info["baseline"]

        if not baseline:
            print(f"[SKIP] Batch {batch_size}: no baseline")
            continue

        print(f"\n[Batch {batch_size}] Benchmarking...")

        # Baseline stats
        base_params, base_gflops = get_model_stats(baseline, args.imgsz)
        base_pt_ms = benchmark_pytorch(baseline, args.imgsz, args.batch, args.iters, args.device)
        base_onnx_ms, onnx_provider = benchmark_onnx(baseline, args.imgsz, args.batch, args.iters)
        print(f"  PyTorch device: {args.device} | ONNX provider: {onnx_provider}")

        md_rows = [
            f"### Batch size {batch_size}",
            f"| Pruning % | Params (M) | FLOPs Reduction | PyTorch Speedup ({args.device}) | ONNX Speedup ({onnx_provider}) |",
            "|-----------|-----------:|----------------:|----------------:|-------------:|",
            f"| 0% (baseline) | {base_params:.2f} | - | 1.00x | 1.00x |",
        ]

        for pct in sorted(info["pruned"].keys()):
            pruned_path = info["pruned"][pct]
            print(f"  Pruning {pct}%...")

            params, gflops = get_model_stats(pruned_path, args.imgsz)
            pt_ms = benchmark_pytorch(pruned_path, args.imgsz, args.batch, args.iters, args.device)
            onnx_ms, _ = benchmark_onnx(pruned_path, args.imgsz, args.batch, args.iters)

            flops_red = (gflops - base_gflops) / base_gflops * 100 if base_gflops else 0
            pt_speedup = base_pt_ms / pt_ms if pt_ms else 0
            onnx_speedup = base_onnx_ms / onnx_ms if onnx_ms and not np.isnan(onnx_ms) else 0

            results.append({
                "batch": batch_size,
                "prune_pct": pct,
                "params_m": params,
                "gflops": gflops,
                "flops_reduction_pct": flops_red,
                "pytorch_ms": pt_ms,
                "onnx_ms": onnx_ms,
                "pytorch_speedup": pt_speedup,
                "onnx_speedup": onnx_speedup,
                "pytorch_device": args.device,
                "onnx_provider": onnx_provider,
            })

            md_rows.append(
                f"| {pct:.0f}% | {params:.2f} | {flops_red:.0f}% | {pt_speedup:.2f}x | {onnx_speedup:.2f}x |"
            )

        md_tables.append("\n".join(md_rows))

    # Write CSV
    csv_path = output_dir / f"comparison_{timestamp}.csv"
    if results:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nCSV: {csv_path}")

    # Write Markdown
    md_path = output_dir / f"summary_{timestamp}.md"
    with open(md_path, "w") as f:
        f.write(f"# {args.dataset} Model Comparison\n\n")
        f.write("\n\n".join(md_tables) + "\n")
    print(f"Markdown: {md_path}")

    print(f"\n=== Done ===")


if __name__ == "__main__":
    main()
