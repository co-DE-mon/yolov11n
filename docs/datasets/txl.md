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

## Download

You can download the TXL-PBC dataset from the following Link:

- [TXL-PBC Datset](https://figshare.com/articles/dataset/TXL-PBC_Dataset/27073186/8)

## Directory Structure

```
data/TXL/
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
- `data.yaml`: YOLO configuration file.
- Additional files (e.g., `BCCD_selection.xlsx`, `metadata_file.xlsx`, `annotation_protocol.pdf`) may be present for reference.

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
