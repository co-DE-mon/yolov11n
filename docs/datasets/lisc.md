# LISC: Leukocyte Image Segmentation and Classification

## Overview

LISC is a leukocyte (white blood cell) dataset adapted to YOLO object detection format. It contains images of blood smears with annotations for leukocyte instances. In this conversion, it is represented as a single-class detection:
- WBC (White Blood Cell)

## Download

You can download the LISC dataset from Roboflow Universe:

- [LISC on Roboflow Universe](https://universe.roboflow.com/raufderakhshandeh/wbc-lisc-aug)

## Directory Structure

```
data/LISC/
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
- `data.yaml`: YOLO dataset config with class names (`names`) and count (`nc`).

## Usage

Set the dataset in your experiment config:

```yaml
dataset: LISC
data: data/LISC/data.yaml
```

Run experiments:

```bash
python src/run_train.py --model yolo11n.pt --data data/LISC/data.yaml --cfg config/default.yaml
python src/training/prune.py --model yolo11n.pt --data data/LISC/data.yaml --cfg config/default.yaml --target_prune_rate 0.2 --dataset LISC
```

## Notes

- Ensure `nc` equals the number of classes listed in `names`.
- Labels must be normalized; verify formatting with a small sample.

## License

Refer to the original dataset license and usage terms.
