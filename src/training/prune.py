import argparse
from pathlib import Path

import torch
import torch_pruning as tp

from ultralytics import YOLO
from ultralytics.nn.modules import Attention
from ultralytics.utils import YAML
from ultralytics.utils.checks import check_yaml

from src.models.c3k2 import replace_c3k2_with_c3k2_v2
from src.training.trainer import train_v2


def prune(args: argparse.Namespace):
    """Prune the YOLO model and finetune.

    Args:
        args: Parsed CLI arguments with fields: model, data, cfg,
              postprune_epochs, batch_size, target_prune_rate, dataset.
    """
    # Load the model - get project root (two levels up from src/training/)
    base_dir = Path(__file__).resolve().parent.parent.parent

    model = YOLO(args.model)

    # Replace the default training method with a custom one that supports pruning
    model.__setattr__("train_v2", train_v2.__get__(model))

    # Loads training configuration from YAML file
    cfg_path = Path(args.cfg)
    if not cfg_path.is_absolute():
        cfg_path = base_dir / cfg_path
    pruning_cfg = YAML.load(check_yaml(str(cfg_path)))
    pruning_cfg["data"] = args.data
    pruning_cfg["epochs"] = args.postprune_epochs
    pruning_cfg["lr0"] = 0.001  # Lower initial learning rate
    pruning_cfg["batch"] = args.batch_size
    pruning_cfg["model"] = args.model

    model.model.train()  # Set to training mode
    replace_c3k2_with_c3k2_v2(model.model)

    for name, param in model.model.named_parameters():
        param.requires_grad = True

    # Establish a single device early and ensure model + example inputs match
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.model.to(device)
    example_inputs = torch.randn(
        1, 3, pruning_cfg["imgsz"], pruning_cfg["imgsz"], device=device
    )

    # Count parameters for pruning ratio calculation
    nparams = sum(p.numel() for p in model.model.parameters())

    fixed_params = 0
    for m in model.model.modules():
        if isinstance(m, Attention):
            fixed_params += sum(p.numel() for p in m.parameters(recurse=True))

    target_prunable = (nparams * args.target_prune_rate) / (nparams - fixed_params)
    if target_prunable >= 1.0:
        target_prunable = 0.99
    pruning_ratio = target_prunable

    # Pruning step
    model.model.train()
    for name, param in model.model.named_parameters():
        param.requires_grad = True

    ignored_layers = []
    unwrapped_parameters = []
    for m in model.model.modules():
        if isinstance(m, Attention):
            ignored_layers.append(m)

    # Guarantee example_inputs resides on same device as model weights before tracing
    model_device = next(model.model.parameters()).device
    if example_inputs.device != model_device:
        example_inputs = example_inputs.to(model_device)

    pruner = tp.pruner.GroupNormPruner(
        model.model,
        example_inputs,
        importance=tp.importance.GroupMagnitudeImportance(),
        iterative_steps=1,
        pruning_ratio=pruning_ratio,
        ignored_layers=ignored_layers,
        unwrapped_parameters=unwrapped_parameters,
    )
    pruner.step()

    # COMPREHENSIVE DEVICE SYNCHRONIZATION
    # Reaffirm model on target device (should already be) and keep example_inputs consistent
    model.model = model.model.to(device)
    example_inputs = example_inputs.to(device)

    # Ensure all modules and their attributes are on the correct device
    def move_module_to_device(module, target_device):
        module.to(target_device)
        for attr_name in dir(module):
            if not attr_name.startswith("_") and not callable(
                getattr(module, attr_name)
            ):
                attr_value = getattr(module, attr_name)
                if isinstance(attr_value, torch.Tensor):
                    setattr(module, attr_name, attr_value.to(target_device))

    for module in model.model.modules():
        move_module_to_device(module, device)

    # Handle loss function specifically
    if hasattr(model.model, "criterion") and model.model.criterion is not None:
        criterion = model.model.criterion
        if hasattr(criterion, "to"):
            move_module_to_device(criterion, device)
        else:
            for attr_name in dir(criterion):
                if not attr_name.startswith("_") and not callable(
                    getattr(criterion, attr_name)
                ):
                    attr_value = getattr(criterion, attr_name)
                    if isinstance(attr_value, torch.Tensor):
                        setattr(criterion, attr_name, attr_value.to(device))

    save_dir_before = base_dir / "runs" / "pruned_models"
    save_dir_after = base_dir / "runs"
    save_dir_before.mkdir(parents=True, exist_ok=True)
    pruned_model_path = (
        save_dir_before
        / f"{args.dataset}_batch{args.batch_size}_pruned{int(args.target_prune_rate * 100)}_before_finetune.pt"
    )
    torch.save(model.model.state_dict(), pruned_model_path)

    # fine-tuning
    for name, param in model.model.named_parameters():
        param.requires_grad = True
    pruning_cfg["name"] = (
        f"{args.dataset}_batch{args.batch_size}_pruned{int(args.target_prune_rate * 100)}"
    )
    pruning_cfg["epochs"] = args.postprune_epochs
    pruning_cfg["batch"] = args.batch_size
    pruning_cfg["project"] = save_dir_after
    pruning_cfg["exist_ok"] = True

    # Filter out custom config keys that are not valid ultralytics arguments
    invalid_keys = {
        "epochs_pre",
        "prune_method",
        "model_presets",
        "model_families",
        "model_family",
        "model_size",
        "batch_sizes",
        "prune_ratios",
        "prune_target",
    }

    # Create a filtered config dict with only valid ultralytics arguments
    filtered_cfg = {k: v for k, v in pruning_cfg.items() if k not in invalid_keys}

    model.train_v2(pruning=True, **filtered_cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="examples/yolov11n/runs/fine_tuned_yolo11n/weights/best.pt",
        help="Model path",
    )
    parser.add_argument(
        "--data", default="examples/yolov11n/BCCD/data.yaml", help="DATA path"
    )
    parser.add_argument(
        "--cfg",
        default="config/default.yaml",
        help="Pruning config file."
        " This file should have same format with ultralytics/yolo/cfg/default.yaml",
    )
    parser.add_argument(
        "--postprune_epochs",
        default=2,
        type=int,
        help="Number of epochs for post pruning finetuning",
    )
    parser.add_argument("--batch_size", default=4, type=int, help="Batch Size")
    parser.add_argument(
        "--target_prune_rate", default=0.5, type=float, help="Target pruning rate"
    )
    parser.add_argument(
        "--dataset", default="dataset", type=str, help="Dataset name for output naming"
    )
    parser.add_argument("--model-size", default=None, help="Model size (n, s, m, l, x)")
    parser.add_argument(
        "--model-family", default=None, help="Model family (e.g., yolo11)"
    )

    args = parser.parse_args()
    pruned_model = prune(args)
