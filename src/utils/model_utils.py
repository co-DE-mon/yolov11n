"""Model size utilities for YOLOv11."""

from pathlib import Path
from typing import Tuple

VALID_SIZES = {"n", "s", "m", "l", "x"}
VALID_FAMILIES = {"yolo11"}


def validate_model_config(model_size: str, model_family: str) -> None:
    """Validate model size and family."""
    if model_size not in VALID_SIZES:
        raise ValueError(
            f"Invalid model size: {model_size}. Valid: {sorted(VALID_SIZES)}"
        )
    if model_family not in VALID_FAMILIES:
        raise ValueError(
            f"Invalid model family: {model_family}. Valid: {sorted(VALID_FAMILIES)}"
        )


def construct_model_path(model_family: str, model_size: str) -> str:
    """Construct model path from family and size."""
    validate_model_config(model_size, model_family)
    return f"{model_family}{model_size}.pt"


def extract_model_info(model_path: str | Path) -> Tuple[str, str]:
    """Extract model family and size from model path."""
    model_path = Path(model_path)
    stem = model_path.stem  # e.g., "yolo11n"

    # Try to match pattern like "yolo11n", "yolo11s", etc.
    for family in VALID_FAMILIES:
        if stem.startswith(family):
            size = stem[len(family) :]
            if size in VALID_SIZES:
                return family, size

    # Default fallback
    return "yolo11", "n"


def extract_model_size(model_path: str | Path) -> str:
    """Extract model size from model path."""
    return extract_model_info(model_path)[1]


def extract_model_family(model_path: str | Path) -> str:
    """Extract model family from model path."""
    return extract_model_info(model_path)[0]
