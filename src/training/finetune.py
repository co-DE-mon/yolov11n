from ultralytics import YOLO
from pathlib import Path
import argparse


def fine_tune_yolov11(
    model_path: str,
    data_yaml: str,
    epochs: int = 50,
    batch_size: int = 4,
    device: str = "0",
    project_dir: str | None = None,
    run_name: str | None = None,
    model_size: str = "n",
    model_family: str = "yolo11",
):
    """
    Fine-tune a YOLOv11 model.

    Args:
        model_path: Path to a .pt file (e.g. "yolo11n.pt" or checkpoint)
        data_yaml: Path to dataset YAML (with train/val paths and classes)
        epochs: Number of epochs for fine-tuning
        batch_size: Training batch size
        device: GPU id (e.g. "0") or "cpu"
        project_dir: Directory to save runs (defaults to `<repo>/runs`)
        run_name: Name of the training run (defaults to `finetune_b{batch_size}`)
        model_size: Model size (n, s, m, l, x) for validation/logging
        model_family: Model family (yolo11) for validation/logging
    Returns:
        Path to the best weights file (may not exist if training didn't save).
    """

    base_dir = Path(__file__).resolve().parent
    project_path = Path(project_dir) if project_dir else (base_dir / "runs")
    run_name = run_name or f"finetune_b{batch_size}"
    project_path.mkdir(parents=True, exist_ok=True)

    # Load model (this downloads or loads the YOLOv11n architecture + weights)
    model_path = str(
        (base_dir / model_path) if not Path(model_path).is_absolute() else model_path
    )
    print(f"🔧 Loading model from: {model_path}")
    model = YOLO(model_path)

    # Fine-tune / train
    model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        device=device,
        name=run_name,
        project=str(project_path),
        exist_ok=True,
    )

    # Verify weights file was created
    weights_path = project_path / run_name / "weights" / "best.pt"
    if weights_path.is_file():
        print(f"Successfully created weights file at: {weights_path}")
    else:
        print(f"Warning: Weights file not found at expected location: {weights_path}")

    return weights_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv11 on a dataset")
    parser.add_argument("--model", default="yolo11n.pt", help="Path to model .pt file")
    parser.add_argument(
        "--model-size",
        choices=["n", "s", "m", "l", "x"],
        help="Model size (for validation)",
    )
    parser.add_argument("--model-family", help="Model family (for validation)")
    parser.add_argument("--data", default="BCCD/data.yaml", help="Path to dataset YAML")
    parser.add_argument("--epochs", default=50, type=int, help="Number of epochs")
    parser.add_argument(
        "--batch-size", dest="batch_size", default=4, type=int, help="Batch size"
    )
    parser.add_argument("--device", default="0", help="Device id (e.g. '0') or 'cpu'")
    parser.add_argument(
        "--project", dest="project_dir", default=None, help="Project directory for runs"
    )
    parser.add_argument("--name", dest="run_name", default=None, help="Run name")
    args = parser.parse_args()

    weights_path = fine_tune_yolov11(
        model_path=args.model,
        data_yaml=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        project_dir=args.project_dir,
        run_name=args.run_name,
        model_size=args.model_size or "n",
        model_family=args.model_family or "yolo11",
    )
    print(f"Finished. Best weights expected at: {weights_path}")
