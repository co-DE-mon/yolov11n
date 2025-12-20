# BCCD: Blood Cell Count and Detection

## Overview

BCCD is a blood cell detection dataset containing microscopy images annotated for object detection in YOLO format. It includes three classes:
- Platelets
- RBC (Red Blood Cell)
- WBC (White Blood Cell)

## Download

You can download the BCCD dataset from Roboflow Universe:

- [BCCD on Roboflow Universe](https://universe.roboflow.com/joseph-nelson/bccd)

## Directory Structure

```
data/BCCD/
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
- `data.yaml`: YOLO dataset config (paths, splits, class names).

## Usage

Update your config to point at this dataset:

```yaml
dataset: BCCD
data: data/BCCD/data.yaml
```

Then run training or pruning:

```bash
python src/run_train.py --model yolo11n.pt --data data/BCCD/data.yaml --cfg config/default.yaml
python src/training/prune.py --model yolo11n.pt --data data/BCCD/data.yaml --cfg config/default.yaml --target_prune_rate 0.2 --dataset BCCD
```

## Citation

Please cite the original BCCD creators if you use this dataset.

## License

Refer to the original dataset license for usage terms.
