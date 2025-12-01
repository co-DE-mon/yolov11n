"""
Run training experiments: finetune baseline + prune at various ratios.

Usage:
    python -m src.run_train                          # Use config/experiment.yaml
    python -m src.run_train --config config/my_exp.yaml
    python -m src.run_train --batch-sizes 4 8        # Override config values
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

# Model utilities (inline for now)
VALID_SIZES = {"n", "s", "m", "l", "x"}
VALID_FAMILIES = {"yolo11"}


def validate_model_config(model_size: str, model_family: str) -> None:
    """Validate model size and family."""
    if model_size not in VALID_SIZES:
        raise ValueError(
            f"Invalid model size: {model_size}. Valid: {sorted(VALID_SIZES)}"
        )
    if model_family not in VALID_FAMILIES:
        raise ValueError(
            f"Invalid model family: {model_family}. Valid: {sorted(VALID_FAMILIES)}"
        )


def construct_model_path(model_family: str, model_size: str) -> str:
    """Construct model path from family and size."""
    validate_model_config(model_size, model_family)
    return f"{model_family}{model_size}.pt"


PROJECT_ROOT = Path(__file__).parent.parent


def load_config_with_inheritance(config_path: Path) -> dict:
    """Load configuration with inheritance from default.yaml."""
    # Load default config first
    default_config_path = PROJECT_ROOT / "config/default.yaml"
    with open(default_config_path) as f:
        config = yaml.safe_load(f)

    # Load dataset-specific config and merge
    if config_path and config_path.exists():
        with open(config_path) as f:
            dataset_config = yaml.safe_load(f)

        # Merge dataset config over defaults (dataset config takes precedence)
        config.update(dataset_config)

    return config


def run(cmd: list[str], env: dict | None = None) -> int:
    """Run command with live output."""
    proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr, env=env)
    proc.wait()
    return proc.returncode


def extract_dataset_name(data_path: str) -> str:
    """Extract dataset name from data.yaml path (e.g., data/TXL/data.yaml -> TXL)."""
    path = Path(data_path)
    # Parent of data.yaml is the dataset folder
    return path.parent.name


def main():
    parser = argparse.ArgumentParser(description="Run finetune + prune experiments")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config/experiment.yaml",
        help="Path to experiment config YAML",
    )
    parser.add_argument(
        "--model-size",
        default=None,
        choices=["n", "s", "m", "l", "x"],
        help="Model size: n, s, m, l, x",
    )
    parser.add_argument("--model-family", default=None, help="Model family: yolo11")
    parser.add_argument("--data", default=None)
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset name (auto-extracted from --data if not provided)",
    )
    parser.add_argument(
        "--epochs-pre", type=int, default=None, help="Baseline finetune epochs"
    )
    parser.add_argument(
        "--epochs-post", type=int, default=None, help="Post-prune finetune epochs"
    )
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=None)
    parser.add_argument("--prune-ratios", type=int, nargs="+", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    # Load config file with inheritance
    config = load_config_with_inheritance(args.config)
    print(f"Loaded config: {args.config} (inherits from config/default.yaml)")

    # CLI args override config values
    model_family = args.model_family or config.get("model_family", "yolo11")

    # Determine model sizes to train
    if args.model_size:
        # CLI specified a single model size
        model_sizes = [args.model_size]
    else:
        # Get all sizes from config for the model family
        family_config = config.get("model_families", {}).get(model_family, {})
        model_sizes = family_config.get("sizes", [config.get("model_size", "n")])

    data = args.data or config.get("data", str(PROJECT_ROOT / "data/TXL/data.yaml"))
    dataset = args.dataset or config.get("dataset") or extract_dataset_name(data)
    epochs_pre = (
        args.epochs_pre
        if args.epochs_pre is not None
        else config.get("epochs_pre", 100)
    )
    epochs_post = (
        args.epochs_post
        if args.epochs_post is not None
        else config.get("epochs_post", 50)
    )
    batch_sizes = args.batch_sizes or config.get("batch_sizes", [4, 8])
    prune_ratios = args.prune_ratios or config.get("prune_ratios", [20, 50])
    device = args.device or config.get("device", "0")

    # Resolve relative data path
    if not Path(data).is_absolute():
        data = str(PROJECT_ROOT / data)

    cfg = str(PROJECT_ROOT / "config/default.yaml")
    runs = str(PROJECT_ROOT / "runs")

    print(f"=== Training Experiment ===")
    print(f"Dataset: {dataset}")
    print(f"Model family: {model_family}")
    print(f"Model sizes: {model_sizes}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Prune ratios: {prune_ratios}")
    print(f"Epochs (pre/post): {epochs_pre}/{epochs_post}")
    print(f"Started: {datetime.now()}")

    for model_size in model_sizes:
        model = construct_model_path(model_family, model_size)
        print(f"\n{'='*60}")
        print(f"Training with model size: {model_size} ({model})")
        print(f"{'='*60}")

        for batch in batch_sizes:
            print(f"\n[{dataset}] [Model {model_size}] [Batch {batch}] Finetuning baseline...")

            # Set environment for subprocess
            env = {"PYTHONPATH": str(PROJECT_ROOT), **os.environ}

            status = run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "src/training/finetune.py"),
                    "--model",
                    model,
                    "--model-size",
                    model_size,
                    "--model-family",
                    model_family,
                    "--data",
                    data,
                    "--epochs",
                    str(epochs_pre),
                    "--batch-size",
                    str(batch),
                    "--device",
                    device,
                    "--project",
                    runs,
                    "--name",
                    f"{dataset}_{model_size}_batch{batch}_baseline",
                ],
                env=env,
            )

            if status != 0:
                print(f"[ERROR] Finetune failed for batch {batch}")
                continue

            weights = f"{runs}/{dataset}_{model_size}_batch{batch}_baseline/weights/best.pt"

            for ratio in prune_ratios:
                print(f"\n[{dataset}] [Model {model_size}] [Batch {batch}] Pruning {ratio}%...")

                status = run(
                    [
                        sys.executable,
                        str(PROJECT_ROOT / "src/training/prune.py"),
                        "--model",
                        weights,
                        "--model-size",
                        model_size,
                        "--model-family",
                        model_family,
                        "--data",
                        data,
                        "--cfg",
                        cfg,
                        "--postprune_epochs",
                        str(epochs_post),
                        "--batch_size",
                        str(batch),
                        "--target_prune_rate",
                        str(ratio / 100),
                        "--dataset",
                        dataset,
                    ]
                )

                if status != 0:
                    print(f"[ERROR] Prune {ratio}% failed")

    print(f"\n=== Training Done: {datetime.now()} ===")


if __name__ == "__main__":
    main()
