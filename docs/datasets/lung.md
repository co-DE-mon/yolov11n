# Lung: Lung Cell Detection Dataset

## Overview

The Lung dataset contains annotated images for lung cell detection and classification in YOLO format. It is designed for training and evaluating models to detect and classify lung cells or related structures in medical images.

## Download

You can download the Lung dataset from Roboflow Universe:

- [Lung Cell Dataset on Roboflow Universe](https://universe.roboflow.com/mehmet-fatih-akca/yolotransfer/dataset/2)


## Directory Structure

```
data/Lung/
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
dataset: Lung
data: data/Lung/data.yaml
```

Then run training or pruning:

```bash
python src/run_train.py --model yolo11n.pt --data data/Lung/data.yaml --cfg config/default.yaml
python src/training/prune.py --model yolo11n.pt --data data/Lung/data.yaml --cfg config/default.yaml --target_prune_rate 0.2 --dataset Lung
```

## Citation

Please cite the original Lung dataset creators or Roboflow Universe if you use this dataset.

## License

Refer to the original dataset license for usage terms.
