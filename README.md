# yolov11n

A lightweight YOLOv11n workflow for training, pruning, benchmarking, and exporting a YOLO model, including scripts to measure FLOPs/params, latency, FPS, and inference time.

## Project Structure

- `c3k2_v2.py`: Custom module to replace specific YOLO blocks (C3k2) with a modified `c3k2_v2` variant used during pruning/training.
- `compare_all_models.py`: Benchmark harness to compare multiple model checkpoints/configs and aggregate metrics.
- `export_onnx.py`: Exports trained/pruned YOLO model to ONNX for deployment.
- `exp_1.sh`: Shell script for running a predefined experiment (Linux/macOS). On Windows, translate commands to PowerShell.
- `finetune.py`: Fine-tuning utility for post-training or post-pruning adjustments on a dataset.
- `metrics.py`: Utility functions to compute FLOPs, MACs, parameter counts, latency, FPS, inference time, and model size.
- `prune.py`: Iterative pruning pipeline for YOLOv11n with device-safe synchronization, pre/post-pruning metrics, and optional fine-tuning.
- `TABLE.md`: Human-readable summary of experiments or benchmarks (e.g., consolidated metrics/results).
- `train_v2.py`: Overrides parts of Ultralytics training to support pruning workflows and safe checkpoint saving; provides `train_v2()` entry.
- `yolo11n.pt`: Base YOLOv11n weights file used as a starting point for training/pruning.
- `logs/compare_all/…`: CSV/JSON/Markdown benchmark outputs collected by `compare_all_models.py` or related scripts.

## End-to-End Workflow

1. Setup environment
   - Install Python 3.10+ and CUDA if using GPU.
   - Install dependencies (Ultralytics, torch, torch_pruning, and any local requirements).

2. Download dataset (TXL‑PBC)
   - Fetch from Figshare (link below) and prepare `data.yaml` for Ultralytics.

3. Train or fine-tune
   - Use `train_v2.py` or `finetune.py` with your `data.yaml` to produce a `best.pt`.

4. Measure baseline metrics
   - Use `metrics.py` helpers from within scripts to report FLOPs/params, latency, FPS, etc.

5. Prune iteratively
   - Run `prune.py` to prune the model in steps; it saves pre-finetune pruned weights and then fine-tunes them.

6. Compare/benchmark
   - Optionally run `compare_all_models.py` to produce aggregate reports in `logs/compare_all`.

7. Export for deployment
   - Use `export_onnx.py` to produce an ONNX artifact.

## Quick Start (Windows PowerShell)

```powershell
# 1) Create and activate venv
python -m venv .venv; .\.venv\Scripts\Activate.ps1

# 2) Install core dependencies
pip install ultralytics torch torchvision torch_pruning onnx onnxruntime

# 3) Download TXL-PBC dataset
# Visit the Figshare page and download the archive. Unpack to, e.g., datasets\TXL_PBC
# Then create a data.yaml pointing to the images/labels per Ultralytics spec.

# 4) Train/Fine-tune (example)
python .\train_v2.py --cfg .\default.yaml --data .\datasets\TXL_PBC\data.yaml --epochs 100 --batch 16

# or fine-tune an existing model
python .\finetune.py --model .\yolo11n.pt --data .\datasets\TXL_PBC\data.yaml --epochs 50

# 5) Prune (iterative + post-pruning finetune)
python .\prune.py --model .\runs\fine_tuned_yolo11n\weights\best.pt --data .\datasets\TXL_PBC\data.yaml --cfg .\default.yaml --iterative_steps 2 --postprune_epochs 2 --batch_size 4 --target_prune_rate 0.5

# 6) Export ONNX
python .\export_onnx.py --weights .\runs\pruned_models\best.pt --imgsz 640 --dynamic

# 7) Compare models (optional, adjust script args inside)
python .\compare_all_models.py
```

## Experiment Script: `exp_1.sh`

- Purpose: Automates a full pruning experiment for YOLO11n across multiple batch sizes and pruning ratios.
- High-level flow:
   - Creates/activates a venv (`ultravenv`), installs dependencies.
   - Pre-trains Model A for `EPOCHS_PRE` epochs per batch size (no pruning).
   - Iterates pruning percentages (`PRUNE_LIST`) and runs `prune.py` with `--target_prune_rate` and post-pruning fine-tuning (`EPOCHS_POST`).
   - Logs every stage to `logs/` with timestamped files; final summary appended to the master log.
- Key parameters inside the script:
   - `EPOCHS_PRE=25`: epochs for the initial fine-tune without pruning.
   - `EPOCHS_POST=25`: epochs for post-pruning fine-tuning.
   - `batch_sizes=(2 4 8 16)`: batch sizes explored.
   - `PRUNE_LIST=(5 10 20 30 40 50 75)`: pruning percentages.
   - `DATA_PATH`: points to TXL-PBC `data.yaml` (adjust path to your environment).
   - `BASE_MODEL="yolo11n.pt"`: initial weights.

### Run on Linux/macOS

```bash
chmod +x ./exp_1.sh
./exp_1.sh
```

Logs are written to `logs/`, models and training artifacts to `runs/`.

### Run on Windows (PowerShell equivalent)

This script uses Bash-specific features (arrays, `source`), so run it in Git Bash or WSL. If you prefer native PowerShell, replicate the steps:

```powershell
# Create venv and install deps
python -m venv .\ultravenv; .\ultravenv\Scripts\Activate.ps1
pip install -U pip
pip install ultralytics torch torch-pruning

# Set variables
$SCRIPTPATH = (Get-Location).Path
$DATA_PATH = Join-Path $SCRIPTPATH "27073186/TXL-PBC/TXL-PBC/data.yaml"
$BASE_MODEL = Join-Path $SCRIPTPATH "yolo11n.pt"
$LOG_DIR = "logs"
$TIMESTAMP = Get-Date -Format "yyyyMMdd_HHmmss"
New-Item -ItemType Directory -Path $LOG_DIR -Force | Out-Null

# Example: one batch size and one prune ratio
$BATCH = 8
$EPOCHS_PRE = 25
$EPOCHS_POST = 25
$PRUNE_PCT = 30
$PRUNE_RATE = [math]::Round($PRUNE_PCT/100.0,4)

# 1) Pre-train Model A (no pruning)
python .\finetune.py `
   --model $BASE_MODEL `
   --data $DATA_PATH `
   --epochs $EPOCHS_PRE `
   --batch-size $BATCH `
   --device 0 `
   --project (Join-Path $SCRIPTPATH "runs") `
   --name "finetune_b$BATCH"

# 2) Prune + post-finetune
$MODEL_A_WEIGHTS = Join-Path $SCRIPTPATH "runs/finetune_b$BATCH/weights/best.pt"
python .\prune.py `
   --model $MODEL_A_WEIGHTS `
   --data $DATA_PATH `
   --cfg (Join-Path $SCRIPTPATH "default.yaml") `
   --iterative_steps 2 `
   --postprune_epochs $EPOCHS_POST `
   --batch_size $BATCH `
   --target_prune_rate $PRUNE_RATE

# Deactivate when done
.\ultravenv\Scripts\deactivate.bat
```

For full factorial runs (multiple batch sizes and prune ratios), use Git Bash/WSL to execute `exp_1.sh` directly or port the loops into PowerShell.

## How `prune.py` Works (From Scratch)

- Loads a YOLO model from `--model` and binds a custom `train_v2()` that supports pruning/fine-tuning.
- Reads pruning config (`--cfg`), injects dataset path (`--data`), epochs (`--postprune_epochs`), batch size (`--batch_size`), and learning rate.
- Computes baseline metrics using `metrics.py` (`get_flops_and_params`, `get_latency`, `get_fps`, `get_inference_time`, `get_model_size`).
- Calculates a per-iteration pruning ratio from a target prune rate that excludes parameters in attention layers.
- Uses `torch_pruning` (GroupNormPruner with GroupMagnitudeImportance) to prune across `iterative_steps`.
- Saves intermediate pruned weights, then fine-tunes via `model.train_v2(pruning=True, **cfg)`.
- Recomputes metrics after pruning and prints before/after comparisons.

## TXL‑PBC Dataset (Brief)

- Source: https://figshare.com/articles/dataset/TXL-PBC_Dataset/27073186/8
- Domain: Peripheral blood cell imagery (microscopy), curated for classification/detection tasks.
- Contents: Multiple classes of blood cells with split-able images/labels suitable for training object detectors.
- Usage: Unpack the dataset and build a Ultralytics-style `data.yaml` (train/val/test paths, class names). Ensure images and labels follow YOLO format (labels in `.txt` with normalized coordinates).

## Notes

- Paths and defaults in `prune.py` may refer to example locations (e.g., `examples/yolov11n/...`); adjust to your local paths.
- GPU strongly recommended for training/pruning; ensure CUDA and matching PyTorch build are installed.
- Benchmark outputs are written under `logs/compare_all/` or `runs/` depending on the script.
