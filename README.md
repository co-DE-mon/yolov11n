# YOLOv11n Pruning Experiments

Multi-model-size training and pruning experiments for YOLOv11 with automatic ONNX export and performance comparison.

## Quick Start

```bash
# Install dependencies
pip install -e .

# Run complete workflow (trains all model sizes: n, s)
./scripts/run_experiment.sh Cellpose

# This will:
# 1. Train yolo11n and yolo11s with batch sizes [1, 2]
# 2. Apply pruning ratios [20%, 50%] to each
# 3. Export all models to ONNX
# 4. Generate performance comparison reports
```

## Run Experiments

The training system automatically trains **all model sizes** defined in `config/default.yaml`:

```bash
# Full pipeline (train -> export -> compare)
# This trains ALL model sizes (n, s) defined in model_families.yolo11.sizes
./scripts/run_experiment.sh Cellpose

# Individual steps
./scripts/run_experiment.sh Cellpose train    # Train all model sizes
./scripts/run_experiment.sh Cellpose export   # Export ONNX for all models
./scripts/run_experiment.sh Cellpose compare  # Compare all models
```

### **Training Workflow**

When you run training without specifying `--model-size`, the system trains **all sizes** in the config:

```bash
./scripts/run_experiment.sh Cellpose train
```

This will:
1. Train `yolo11n` (nano) with batch sizes [1, 2]
2. Train `yolo11s` (small) with batch sizes [1, 2]
3. Apply pruning ratios [20%, 50%] to each model

**Output structure:**
```
runs/
├── Cellpose_n_batch1_baseline/
├── Cellpose_n_batch1_pruned20/
├── Cellpose_n_batch1_pruned50/
├── Cellpose_n_batch2_baseline/
├── Cellpose_n_batch2_pruned20/
├── Cellpose_n_batch2_pruned50/
├── Cellpose_s_batch1_baseline/
├── Cellpose_s_batch1_pruned20/
├── Cellpose_s_batch1_pruned50/
├── Cellpose_s_batch2_baseline/
├── Cellpose_s_batch2_pruned20/
└── Cellpose_s_batch2_pruned50/
```

### **Train Specific Model Size**

To train only one model size:

```bash
# Train only nano model
./scripts/run_experiment.sh Cellpose train --model-size n

# Train only small model
./scripts/run_experiment.sh Cellpose train --model-size s
```

## Configuration

The project uses a hierarchical configuration system with inheritance:

### **Configuration Structure**

```
config/
├── default.yaml              # Global defaults and model configurations
└── datasets/
    ├── TXL.yaml           # Dataset-specific overrides
    └── Cellpose.yaml       # Dataset-specific overrides
```

### **Model Sizes**

All YOLOv11 model sizes are supported:

- **n** (nano) - Fastest, smallest model
- **s** (small) - Balanced speed and accuracy
- **m** (medium) - Good accuracy, moderate speed
- **l** (large) - High accuracy, slower
- **x** (extra large) - Highest accuracy, slowest

### **Configuration Inheritance**

Dataset configs inherit from `config/default.yaml`:

```yaml
# config/default.yaml (base defaults)
model_families:
  yolo11:
    sizes: [n, s]          # Model sizes to train (trains ALL by default)
    default_size: n

model_family: yolo11       # Model family
epochs_pre: 1              # Pre-pruning training epochs
epochs_post: 1             # Post-pruning training epochs
batch_sizes: [1, 2]        # Batch sizes to test
prune_ratios: [20, 50]     # Pruning percentages
device: "0"                # GPU device
```

**Important**: When running training without `--model-size`, the system trains **all sizes** listed in `model_families.yolo11.sizes`.

### **Usage Examples**

#### **Default Behavior: Train All Model Sizes**
```bash
# Trains all sizes defined in model_families.yolo11.sizes: [n, s]
python -m src.run_train --config config/datasets/Cellpose.yaml
```

#### **Train Specific Model Size**
```bash
# Train only nano model
python -m src.run_train --config config/datasets/Cellpose.yaml --model-size n

# Train only small model
python -m src.run_train --config config/datasets/Cellpose.yaml --model-size s
```

#### **Configure Model Sizes to Train**

Edit `config/default.yaml` to change which sizes train by default:

```yaml
model_families:
  yolo11:
    sizes: [n, s, m]       # Now trains nano, small, AND medium
    default_size: n
```

Or override in dataset config `config/datasets/TXL.yaml`:

```yaml
dataset: TXL
data: data/TXL/data.yaml

# Override which model sizes to train for this dataset
model_families:
  yolo11:
    sizes: [m, l]          # Only train medium and large for TXL

# Other parameters inherited from default.yaml
```

#### **Complete Workflow Example**

```bash
# 1. Train all model sizes (n, s) with all batch sizes and pruning ratios
./scripts/run_experiment.sh Cellpose train

# 2. Export all trained models to ONNX format
./scripts/run_experiment.sh Cellpose export

# 3. Generate performance comparison reports
./scripts/run_experiment.sh Cellpose compare

# Results saved to:
# - CSV: logs/Cellpose/comparison_*.csv
# - Markdown: logs/Cellpose/summary_*.md
```

#### **Available Parameters**
- `model_families.{family}.sizes`: List of model sizes to train (e.g., [n, s, m])
- `model_size`: n, s, m, l, x (single model size for CLI override)
- `model_family`: yolo11 (model family)
- `epochs_pre`: Pre-pruning training epochs
- `epochs_post`: Post-pruning training epochs
- `batch_sizes`: List of batch sizes to test
- `prune_ratios`: List of pruning percentages
- `device`: GPU device ID or "cpu"

## Adding a Dataset

1. Create `data/<DATASET_NAME>/` with YOLO structure:
   ```
   data/<DATASET_NAME>/
   ├── images/{train,val,test}/
   ├── labels/{train,val,test}/
   └── data.yaml
   ```

2. Create `config/datasets/<DATASET_NAME>.yaml` (inherits from `config/default.yaml`):
   ```yaml
   # Dataset Configuration (inherits from config/default.yaml)
   dataset: <DATASET_NAME>
   data: data/<DATASET_NAME>/data.yaml

   # Optional: Override which model sizes to train
   # model_families:
   #   yolo11:
   #     sizes: [n]           # Only train nano for this dataset

   # Optional: Override training parameters
   # epochs_pre: 50           # Custom pre-pruning epochs
   # batch_sizes: [2, 4]      # Custom batch sizes
   ```

   **Note**: By default, trains **all sizes** from `config/default.yaml` unless overridden.

3. Run complete workflow:
   ```bash
   # Train all model sizes defined in config
   ./scripts/run_experiment.sh <DATASET_NAME> train

   # Or train specific size only
   ./scripts/run_experiment.sh <DATASET_NAME> train --model-size n

   # Export to ONNX
   ./scripts/run_experiment.sh <DATASET_NAME> export

   # Generate comparison reports
   ./scripts/run_experiment.sh <DATASET_NAME> compare

   # Or run full pipeline at once
   ./scripts/run_experiment.sh <DATASET_NAME>
   ```
