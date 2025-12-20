# S_tools: Surgical Tools Detection Dataset

## Overview

S_tools is a surgical tool detection dataset containing annotated images of surgical instruments for object detection in YOLO format. It is designed for training and evaluating models to detect and classify surgical tools in medical images.

## Download

You can download the S_tools dataset from Roboflow Universe:

- [Surgical Tools Dataset on Roboflow Universe](https://universe.roboflow.com/surgical-tools/surgical-tools-dataset/dataset/2)
- Direct download (YOLO format):
  https://universe.roboflow.com/ds/1vQw1Qw7vA?key=YOUR_KEY

Replace `YOUR_KEY` with your Roboflow API key if required. After downloading, extract the archive into `data/S_tools/` and ensure the folder structure matches below.

## Directory Structure

```
data/S_tools/
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
- Label files are in YOLO format, one per image, with bounding boxes for each tool instance.
- `data.yaml`: YOLO dataset config (paths, splits, class names).

## Usage

Update your config to point at this dataset:

```yaml
dataset: S_tools
data: data/S_tools/data.yaml
```

Then run training or pruning:

```bash
python src/run_train.py --model yolo11n.pt --data data/S_tools/data.yaml --cfg config/default.yaml
python src/training/prune.py --model yolo11n.pt --data data/S_tools/data.yaml --cfg config/default.yaml --target_prune_rate 0.2 --dataset S_tools
```

## Citation

Please cite the original S_tools dataset creators or Roboflow Universe if you use this dataset.

## License

Refer to the original dataset license for usage terms.
