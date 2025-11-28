#!/usr/bin/env python3
"""
Convert Cellpose instance segmentation masks to YOLO bounding box format.

Cellpose masks use unique integer values for each cell instance.
This script extracts bounding boxes from each instance and saves them in YOLO format.
"""

import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np


def mask_to_yolo_bboxes(mask_path: str, class_id: int = 0) -> list[str]:
    """
    Convert instance segmentation mask to YOLO bounding boxes.

    Args:
        mask_path: Path to the mask PNG file
        class_id: Class ID for all instances (default 0 for single class)

    Returns:
        List of YOLO format strings: "class_id x_center y_center width height"
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        return []

    h, w = mask.shape[:2]

    # Get unique instance IDs (0 is background)
    instance_ids = np.unique(mask)
    instance_ids = instance_ids[instance_ids != 0]

    yolo_lines = []
    for inst_id in instance_ids:
        # Create binary mask for this instance
        inst_mask = (mask == inst_id).astype(np.uint8)

        # Find bounding box
        coords = np.where(inst_mask > 0)
        if len(coords[0]) == 0:
            continue

        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()

        # Convert to YOLO format (normalized center coordinates and dimensions)
        x_center = ((x_min + x_max) / 2) / w
        y_center = ((y_min + y_max) / 2) / h
        bbox_width = (x_max - x_min) / w
        bbox_height = (y_max - y_min) / h

        # Validate bounds
        if bbox_width > 0 and bbox_height > 0:
            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}")

    return yolo_lines


def process_dataset(
    raw_dir: Path,
    images_dir: Path,
    labels_dir: Path,
    split_ratio: float = None,
    val_images_dir: Path = None,
    val_labels_dir: Path = None,
    seed: int = 42
):
    """
    Process raw Cellpose data and convert to YOLO format.

    Args:
        raw_dir: Directory containing *_img.png and *_masks.png files
        images_dir: Output directory for images
        labels_dir: Output directory for labels
        split_ratio: If provided, split data into train/val (e.g., 0.8 for 80% train)
        val_images_dir: Validation images directory (required if split_ratio provided)
        val_labels_dir: Validation labels directory (required if split_ratio provided)
        seed: Random seed for reproducible splits
    """
    # Find all image files
    img_files = sorted(raw_dir.glob("*_img.png"))

    if not img_files:
        print(f"No images found in {raw_dir}")
        return 0, 0

    # Determine train/val split
    if split_ratio:
        random.seed(seed)
        random.shuffle(img_files)
        split_idx = int(len(img_files) * split_ratio)
        train_files = img_files[:split_idx]
        val_files = img_files[split_idx:]
    else:
        train_files = img_files
        val_files = []

    processed = 0
    skipped = 0

    # Process training files
    for img_path in train_files:
        base_name = img_path.stem.replace("_img", "")
        mask_path = raw_dir / f"{base_name}_masks.png"

        if not mask_path.exists():
            print(f"Warning: No mask found for {img_path.name}, skipping")
            skipped += 1
            continue

        # Convert mask to YOLO labels
        yolo_lines = mask_to_yolo_bboxes(str(mask_path))

        if not yolo_lines:
            print(f"Warning: No instances found in {mask_path.name}, skipping")
            skipped += 1
            continue

        # Copy image
        out_img_path = images_dir / f"{base_name}.png"
        shutil.copy2(img_path, out_img_path)

        # Write labels
        out_label_path = labels_dir / f"{base_name}.txt"
        with open(out_label_path, "w") as f:
            f.write("\n".join(yolo_lines))

        processed += 1

    # Process validation files
    val_processed = 0
    if val_files and val_images_dir and val_labels_dir:
        for img_path in val_files:
            base_name = img_path.stem.replace("_img", "")
            mask_path = raw_dir / f"{base_name}_masks.png"

            if not mask_path.exists():
                skipped += 1
                continue

            yolo_lines = mask_to_yolo_bboxes(str(mask_path))

            if not yolo_lines:
                skipped += 1
                continue

            out_img_path = val_images_dir / f"{base_name}.png"
            shutil.copy2(img_path, out_img_path)

            out_label_path = val_labels_dir / f"{base_name}.txt"
            with open(out_label_path, "w") as f:
                f.write("\n".join(yolo_lines))

            val_processed += 1

    return processed, val_processed, skipped


def main():
    base_dir = Path(__file__).parent.parent / "data" / "Cellpose"

    print("Converting Cellpose dataset to YOLO format...")
    print(f"Base directory: {base_dir}")

    # Process training data with 80/20 split
    print("\n[1/2] Processing training data (80/20 split)...")
    train_processed, val_processed, train_skipped = process_dataset(
        raw_dir=base_dir / "train",
        images_dir=base_dir / "images" / "train",
        labels_dir=base_dir / "labels" / "train",
        split_ratio=0.8,
        val_images_dir=base_dir / "images" / "val",
        val_labels_dir=base_dir / "labels" / "val",
        seed=42
    )
    print(f"  Train: {train_processed} images processed")
    print(f"  Val: {val_processed} images processed")
    if train_skipped:
        print(f"  Skipped: {train_skipped} images")

    # Process test data (no split)
    print("\n[2/2] Processing test data...")
    test_processed, _, test_skipped = process_dataset(
        raw_dir=base_dir / "test",
        images_dir=base_dir / "images" / "test",
        labels_dir=base_dir / "labels" / "test"
    )
    print(f"  Test: {test_processed} images processed")
    if test_skipped:
        print(f"  Skipped: {test_skipped} images")

    # Summary
    print("\n" + "=" * 50)
    print("Conversion complete!")
    print(f"  Train: {train_processed} images")
    print(f"  Val: {val_processed} images")
    print(f"  Test: {test_processed} images")
    print(f"  Total: {train_processed + val_processed + test_processed} images")
    print("=" * 50)


if __name__ == "__main__":
    main()
