# Cellpose: Cell Instance Segmentation Dataset

## Overview

Cellpose is a generalist cell segmentation dataset originally designed for instance segmentation. For this project, the dataset has been converted to YOLO object detection format with bounding boxes extracted from instance masks.

The dataset contains microscopy images of cells with a single class:
- Cell

## Dataset Statistics

| Split | Images | Annotations |
|-------|--------|-------------|
| Train | 432    | ~15,000+    |
| Val   | 108    | ~4,000+     |
| Test  | 68     | ~2,500+     |
| **Total** | **608** | **~21,500+** |

## Directory Structure

```
Cellpose/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
├── data.yaml
├── train.zip          # Original archive (backup)
└── test.zip           # Original archive (backup)
```

- `images/`: Contains all images in PNG format, split into train/val/test.
- `labels/`: YOLO-format annotation files (.txt), split into train/val/test.
- `data.yaml`: YOLO configuration file.
- `*.zip`: Original archived data (kept as backup).

## Data Format

### Original Format
- Images: `{id}_img.png` - Microscopy cell images
- Masks: `{id}_masks.png` - Instance segmentation masks where each unique pixel value represents a different cell instance

### Converted Format (YOLO)
- Images: `{id}.png`
- Labels: `{id}.txt` - One line per cell with format: `class_id x_center y_center width height` (normalized 0-1)

## Conversion Process

The dataset was converted using `scripts/convert_cellpose.py`:

1. Instance masks are read as grayscale images
2. Each unique non-zero pixel value represents one cell instance
3. Bounding boxes are computed for each instance
4. Coordinates are normalized to YOLO format
5. Training data is split 80/20 into train/val sets

To re-run conversion:
```bash
python scripts/convert_cellpose.py
```

## Usage

Update `config/experiment.yaml` to use this dataset:

```yaml
dataset: Cellpose
data: data/Cellpose/data.yaml
```

Then run experiments:
```bash
./scripts/run_experiment.sh
```

## Citation

If you use this dataset, please cite the original Cellpose paper:

```
Stringer, C., Wang, T., Michaelos, M., & Pachitariu, M. (2021).
Cellpose: a generalist algorithm for cellular segmentation.
Nature Methods, 18(1), 100-106.
```

## License

Please refer to the original Cellpose dataset license for usage terms.
