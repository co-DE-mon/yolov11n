# Cellpose: Cell Instance Segmentation Dataset

## Overview
The Cellpose dataset is formatted for YOLO object detection training. It contains instance segmentation masks converted to YOLO bounding box format, with a single class: `cell`.

## Download

The original Cellpose dataset and code are available at:
- [Cellpose Official Website](https://www.cellpose.org/)
- [Cellpose GitHub Repository](https://github.com/MouseLand/cellpose)

To use the YOLO-ready version for this project:
1. Download the original Cellpose dataset from the links above.
2. Place the raw images and masks in `data/Cellpose/train/` and `data/Cellpose/test/` as `{id}_img.png` and `{id}_masks.png` pairs.
3. Run the conversion script:
   ```bash
   python scripts/convert_cellpose.py
   ```
4. The script will split and convert the data into YOLO format under `data/Cellpose/` as shown below.

## Directory Structure

```
data/Cellpose/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

- Each split (`train`, `valid`, `test`) contains its own `images/` and `labels/` subfolders.
- Label files are in YOLO format, one per image, with bounding boxes for each cell instance.

## Data Format

- Images: `{id}.png`
- Labels: `{id}.txt` - One line per cell with format: `class_id x_center y_center width height` (normalized 0-1)

## data.yaml Example
```yaml
train: train/images
val: valid/images
test: test/images
nc: 1
names: ['cell']
```

## Conversion Process

The dataset was converted using `scripts/convert_cellpose.py`.

## Usage

Update your experiment config to use this dataset:

```yaml
dataset: Cellpose
data: data/Cellpose/data.yaml
```

Then run experiments:
```bash
./scripts/run_experiment.sh Cellpose
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
