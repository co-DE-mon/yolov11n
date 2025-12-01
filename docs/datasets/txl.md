# TXL-PBC: A Curated and Re-annotated Peripheral Blood Cell Dataset

## Overview

TXL-PBC is a curated and re-annotated peripheral blood cell dataset constructed by integrating four publicly available resources:
- Blood Cell Count and Detection (BCCD)
- Blood Cell Detection Dataset (BCDD)
- Peripheral Blood Cells (PBC)
- Raabin White Blood Cell (Raabin-WBC)

The dataset contains 1,260 images and 18,143 bounding box annotations for three major blood cell types:
- White blood cells (WBC)
- Red blood cells (RBC)
- Platelets (PC)

All images are annotated in YOLO format and split into training, validation, and test sets.

## Directory Structure

```
TXL-PBC/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
├── data.yaml
├── BCCD_selection.xlsx
├── metadata_file.xlsx
└── annotation_protocol.pdf
```

- `images/`: Contains all images, split into train/val/test.
- `labels/`: YOLO-format annotation files, split into train/val/test.
- `data.yaml`: YOLO configuration file.
- `BCCD_selection.xlsx`: List of selected and excluded BCCD images.
- `metadata.csv`: Mapping of image filenames to source datasets.
- `annotation_protocol.pdf`: Manual annotation guidelines.

## Configuration

The TXL dataset uses `config/datasets/TXL.yaml` which inherits from `config/default.yaml`. 

To use different model sizes, modify the inherited `model_size` parameter:

```yaml
# In config/datasets/TXL.yaml
model_size: s  # Override default model size
```

Or use command line:
```bash
python -m src.run_train --config config/datasets/TXL.yaml --model-size s
```

Available model sizes: n (nano), s (small), m (medium), l (large), x (extra large)

## Usage

- The dataset can be used directly with object detection frameworks such as YOLO.
- Recommended preprocessing: image normalization and data augmentation.
- Suitable for training, validation, and benchmarking of blood cell detection models.
- Can be combined with other datasets for cross-validation or transfer learning.

## Citation

If you use this dataset in your research, please cite:

```
Lu Gan, Xi Li, Xichun Wang. TXL-PBC: A Curated and Re-annotated Peripheral Blood Cell Dataset Integrating Four Public Resources. 2024.
```

## License

This dataset is released for academic and non-commercial use.
