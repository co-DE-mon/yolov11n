"""
Run training experiments: finetune baseline + prune at various ratios.

Usage:
    python -m src.run_train --config config/datasets/TXL.yaml
    python -m src.run_train --config config/datasets/TXL.yaml --batch-sizes 4 8   # Override config values
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).parent.parent


def load_config(config_path: Path) -> dict:
    """Load experiment configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def run(cmd: list[str]) -> int:
    """Run command with live output."""
    proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
    proc.wait()
    return proc.returncode


def extract_dataset_name(data_path: str) -> str:
    """Extract dataset name from data.yaml path (e.g., data/TXL/data.yaml -> TXL)."""
    path = Path(data_path)
    # Parent of data.yaml is the dataset folder
    return path.parent.name


def main():
    parser = argparse.ArgumentParser(description="Run finetune + prune experiments")
    parser.add_argument("--config", type=Path, required=True,
                        help="Path to experiment config YAML (e.g. config/datasets/TXL.yaml)")
    parser.add_argument("--model", default=None)
    parser.add_argument("--data", default=None)
    parser.add_argument("--dataset", default=None, help="Dataset name (auto-extracted from --data if not provided)")
    parser.add_argument("--epochs-pre", type=int, default=None, help="Baseline finetune epochs")
    parser.add_argument("--epochs-post", type=int, default=None, help="Post-prune finetune epochs")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=None)
    parser.add_argument("--prune-ratios", type=int, nargs="+", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    # Load config file
    config = load_config(args.config)
    print(f"Loaded config: {args.config}")

    # CLI args override config values
    model = args.model or config.get("model", "yolo11n.pt")
    data = args.data or config.get("data", str(PROJECT_ROOT / "data/TXL/data.yaml"))
    dataset = args.dataset or config.get("dataset") or extract_dataset_name(data)
    epochs_pre = args.epochs_pre if args.epochs_pre is not None else config.get("epochs_pre", 100)
    epochs_post = args.epochs_post if args.epochs_post is not None else config.get("epochs_post", 50)
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
    print(f"Batch sizes: {batch_sizes}")
    print(f"Prune ratios: {prune_ratios}")
    print(f"Epochs (pre/post): {epochs_pre}/{epochs_post}")
    print(f"Started: {datetime.now()}")

    for batch in batch_sizes:
        print(f"\n[{dataset}] [Batch {batch}] Finetuning baseline...")

        status = run([
            sys.executable, str(PROJECT_ROOT / "src/training/finetune.py"),
            "--model", model,
            "--data", data,
            "--epochs", str(epochs_pre),
            "--batch-size", str(batch),
            "--device", device,
            "--project", runs,
            "--name", f"{dataset}_batch{batch}_baseline",
        ])

        if status != 0:
            print(f"[ERROR] Finetune failed for batch {batch}")
            continue

        weights = f"{runs}/{dataset}_batch{batch}_baseline/weights/best.pt"

        for ratio in prune_ratios:
            print(f"\n[{dataset}] [Batch {batch}] Pruning {ratio}%...")

            status = run([
                sys.executable, str(PROJECT_ROOT / "src/training/prune.py"),
                "--model", weights,
                "--data", data,
                "--cfg", cfg,
                "--postprune_epochs", str(epochs_post),
                "--batch_size", str(batch),
                "--target_prune_rate", str(ratio / 100),
                "--dataset", dataset,
            ])

            if status != 0:
                print(f"[ERROR] Prune {ratio}% failed")

    print(f"\n=== Training Done: {datetime.now()} ===")


if __name__ == "__main__":
    main()
