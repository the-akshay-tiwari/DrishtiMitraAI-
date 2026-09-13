"""Train an experimental diabetic-retinopathy grade classifier on the bundled IDRiD data.

The CSV already defines a patient-level training/test split via image names: image IDs ending
in ``test`` are held out and are never used to select the model. This script makes a
stratified validation split from the remaining training images and saves the best validation
checkpoint plus a human-readable metrics file.

This is research software, not a validated clinical decision-support system.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import train_test_split
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
from torchvision.transforms import v2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "archive" / "Imagenes" / "Imagenes"
LABELS_PATH = PROJECT_ROOT / "archive" / "idrid_labels.csv"
APTOS_DIR = PROJECT_ROOT / "data" / "aptos2019" / "extracted"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CLASS_NAMES = ["No apparent DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]
IMAGE_SIZE = 384
NORMALIZATION = {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}


@dataclass(frozen=True)
class Sample:
    image_path: Path
    grade: int


class FundusDataset(Dataset[tuple[Tensor, int]]):
    def __init__(self, samples: list[Sample], transform: v2.Compose):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        sample = self.samples[index]
        with Image.open(sample.image_path) as image:
            rgb_image = image.convert("RGB")
        return self.transform(rgb_image), sample.grade


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["auto", "aptos", "idrid"], default="auto", help="Dataset to train on")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--workers", type=int, default=0, help="0 is most reliable on Windows")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--no-pretrained", action="store_true", help="Train from random weights (not recommended)")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available to PyTorch.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_aptos_image_path(split_prefix: str, id_code: str) -> Path | None:
    """Robustly resolve image path across flat or nested folder extraction structures."""
    candidates = [
        APTOS_DIR / f"{split_prefix}_images" / f"{id_code}.png",
        APTOS_DIR / f"{split_prefix}_images" / f"{id_code}.jpg",
        APTOS_DIR / f"{split_prefix}_images" / f"{split_prefix}_images" / f"{id_code}.png",
        APTOS_DIR / f"{split_prefix}_images" / f"{split_prefix}_images" / f"{id_code}.jpg",
        APTOS_DIR / f"{id_code}.png",
        APTOS_DIR / f"{id_code}.jpg",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def load_aptos_samples() -> tuple[list[Sample], list[Sample], list[Sample]]:
    """Load APTOS 2019 dataset pre-split into train (2930), validation (366), and test (366) sets."""
    def read_csv_samples(csv_path: Path, split_prefix: str, require_diagnosis: bool = True) -> list[Sample]:
        samples: list[Sample] = []
        if not csv_path.is_file():
            return samples
        with csv_path.open(newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                image_id = (row.get("id_code") or "").strip()
                grade_text = (row.get("diagnosis") or "").strip()
                if not image_id:
                    continue
                if require_diagnosis and grade_text not in {"0", "1", "2", "3", "4"}:
                    continue
                img_path = resolve_aptos_image_path(split_prefix, image_id)
                if img_path is None:
                    continue
                grade_val = int(grade_text) if grade_text in {"0", "1", "2", "3", "4"} else -1
                samples.append(Sample(image_path=img_path, grade=grade_val))
        return samples

    train_samples = read_csv_samples(APTOS_DIR / "train_1.csv", "train", require_diagnosis=True)
    val_samples = read_csv_samples(APTOS_DIR / "valid.csv", "val", require_diagnosis=True)
    test_samples = read_csv_samples(APTOS_DIR / "test.csv", "test", require_diagnosis=False)
    return train_samples, val_samples, test_samples


def load_idrid_samples() -> tuple[list[Sample], list[Sample]]:
    if not DATASET_DIR.is_dir() or not LABELS_PATH.is_file():
        raise FileNotFoundError("Expected archive/Imagenes/Imagenes and archive/idrid_labels.csv in the project root.")

    train_samples: list[Sample] = []
    test_samples: list[Sample] = []
    with LABELS_PATH.open(newline="", encoding="utf-8-sig") as labels_file:
        for row in csv.DictReader(labels_file):
            image_id = (row.get("id_code") or "").strip()
            grade_text = (row.get("diagnosis") or "").strip()
            if not image_id or grade_text not in {"0", "1", "2", "3", "4"}:
                continue
            image_path = DATASET_DIR / f"{image_id}.jpg"
            if not image_path.is_file():
                raise FileNotFoundError(f"Missing labelled image: {image_path}")
            sample = Sample(image_path=image_path, grade=int(grade_text))
            (test_samples if image_id.lower().endswith("test") else train_samples).append(sample)

    if len(train_samples) != 375 or len(test_samples) != 80:
        raise ValueError(f"Expected 375 train and 80 test samples; found {len(train_samples)} and {len(test_samples)}.")
    return train_samples, test_samples


def load_samples(dataset_choice: str = "auto") -> tuple[list[Sample], list[Sample], list[Sample], str]:
    if dataset_choice in {"auto", "aptos"} and (APTOS_DIR / "train_1.csv").is_file():
        tr, va, te = load_aptos_samples()
        return tr, va, te, "APTOS 2019 Blindness Detection Dataset"
    train_samples, test_samples = load_idrid_samples()
    train_indices, val_indices = train_test_split(
        np.arange(len(train_samples)),
        test_size=0.2,
        random_state=42,
        stratify=[sample.grade for sample in train_samples],
    )
    return [train_samples[i] for i in train_indices], [train_samples[i] for i in val_indices], test_samples, "IDRiD Indian Diabetic Retinopathy Image Dataset"




def make_transforms() -> tuple[v2.Compose, v2.Compose]:
    normalize = v2.Normalize(mean=NORMALIZATION["mean"], std=NORMALIZATION["std"])
    train_transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        v2.RandomRotation(180),
        v2.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.10, hue=0.04),
        v2.ToDtype(torch.float32, scale=True),
        normalize,
    ])
    evaluation_transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        normalize,
    ])
    return train_transform, evaluation_transform


def build_model(pretrained: bool) -> nn.Module:
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=0.35),
        nn.Linear(in_features, len(CLASS_NAMES)),
    )
    return model


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[dict[str, float], list[int], list[int]]:
    model.eval()
    targets: list[int] = []
    predictions: list[int] = []
    for images, labels in loader:
        logits = model(images.to(device, non_blocking=True))
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
        targets.extend(labels.tolist())
    metrics = {
        "accuracy": float(accuracy_score(targets, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "quadratic_weighted_kappa": float(cohen_kappa_score(targets, predictions, weights="quadratic")),
    }
    return metrics, targets, predictions


def class_weights(samples: Iterable[Sample]) -> Tensor:
    counts = np.bincount([sample.grade for sample in samples], minlength=len(CLASS_NAMES)).astype(np.float32)
    weights = counts.sum() / (len(CLASS_NAMES) * counts)
    return torch.tensor(weights, dtype=torch.float32)


def ordinal_loss(logits: Tensor, targets: Tensor, weights: Tensor) -> Tensor:
    ce = nn.functional.cross_entropy(logits, targets, weight=weights, label_smoothing=0.05)
    probs = nn.functional.softmax(logits, dim=1)
    class_vals = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0], device=logits.device)
    expected_val = (probs * class_vals).sum(dim=1)
    mse = nn.functional.mse_loss(expected_val, targets.float())
    return ce + 0.75 * mse


def make_checkpoint(model: nn.Module, args: argparse.Namespace, metrics: dict[str, float]) -> dict[str, object]:
    return {
        "model_name": "efficientnet_b0",
        "model_state_dict": model.state_dict(),
        "class_names": CLASS_NAMES,
        "image_size": IMAGE_SIZE,
        "normalization": NORMALIZATION,
        "metrics": metrics,
        "training_config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "pretrained": not args.no_pretrained,
        },
        "medical_disclaimer": "Experimental research model only. Not validated for clinical use or autonomous diagnosis.",
    }


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive.")
    set_seed(args.seed)
    device = resolve_device(args.device)
    training_samples, validation_samples, test_samples, dataset_name = load_samples(args.dataset)
    train_transform, evaluation_transform = make_transforms()
    pin_memory = device.type == "cuda"

    # Balanced WeightedRandomSampler
    counts = np.bincount([sample.grade for sample in training_samples], minlength=len(CLASS_NAMES)).astype(np.float32)
    sample_weights = torch.tensor([1.0 / counts[sample.grade] for sample in training_samples], dtype=torch.double)
    sampler = torch.utils.data.WeightedRandomSampler(weights=sample_weights, num_samples=len(training_samples), replacement=True)

    train_loader = DataLoader(FundusDataset(training_samples, train_transform), batch_size=args.batch_size, sampler=sampler, num_workers=args.workers, pin_memory=pin_memory)
    validation_loader = DataLoader(FundusDataset(validation_samples, evaluation_transform), batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=pin_memory)
    test_loader = DataLoader(FundusDataset(test_samples, evaluation_transform), batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=pin_memory)

    model = build_model(pretrained=not args.no_pretrained).to(device)
    weights_tensor = class_weights(training_samples).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    checkpoint_path = ARTIFACTS_DIR / "drishtimitra_efficientnet_b0.pt"
    best_validation_kappa = float("-inf")
    best_epoch = 0
    started_at = time.perf_counter()
    print(f"Training on {device} (IMAGE_SIZE={IMAGE_SIZE}): {len(training_samples)} train / {len(validation_samples)} val / {len(test_samples)} test images [{dataset_name}]")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images.to(device, non_blocking=True))
                loss = ordinal_loss(logits, labels.to(device, non_blocking=True), weights_tensor)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * labels.size(0)
        scheduler.step()

        validation_metrics, _, _ = evaluate(model, validation_loader, device)
        average_loss = total_loss / len(training_samples)
        print(f"Epoch {epoch:02d}/{args.epochs} | loss {average_loss:.4f} | val accuracy {validation_metrics['accuracy']:.3f} | val QWK {validation_metrics['quadratic_weighted_kappa']:.3f}")
        if validation_metrics["quadratic_weighted_kappa"] > best_validation_kappa:
            best_validation_kappa = validation_metrics["quadratic_weighted_kappa"]
            best_epoch = epoch
            torch.save(make_checkpoint(model, args, validation_metrics), checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Check if test set is labeled for ground-truth test evaluation
    has_test_labels = all(s.grade >= 0 for s in test_samples)
    if has_test_labels:
        test_metrics, test_targets, test_predictions = evaluate(model, test_loader, device)
        test_report_metrics = test_metrics
        test_class_counts = {str(grade): test_targets.count(grade) for grade in range(len(CLASS_NAMES))}
    else:
        # Generate predictions for unlabeled test set
        model.eval()
        test_preds: list[int] = []
        with torch.inference_mode():
            for images, _ in test_loader:
                logits = model(images.to(device))
                test_preds.extend(logits.argmax(dim=1).cpu().tolist())
        test_predictions = test_preds
        test_report_metrics = "Unlabeled test set (Inference predictions generated)"
        test_class_counts = "N/A (Unlabeled test set)"

    report = {
        "model": "EfficientNet-B0 ImageNet transfer learning",
        "dataset": dataset_name,
        "split": {"training": len(training_samples), "validation": len(validation_samples), "test": len(test_samples)},
        "best_epoch": best_epoch,
        "best_validation_metrics": checkpoint["metrics"],
        "held_out_test_metrics": test_report_metrics,
        "held_out_test_class_counts": test_class_counts,
        "held_out_test_predictions": {str(grade): test_predictions.count(grade) for grade in range(len(CLASS_NAMES))},
        "elapsed_minutes": round((time.perf_counter() - started_at) / 60, 2),
        "medical_disclaimer": "Experimental research result. It must undergo independent external and clinical validation before any clinical use.",
    }
    metrics_path = ARTIFACTS_DIR / "training_metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nBest validation checkpoint: epoch {best_epoch}, QWK {best_validation_kappa:.3f}")
    print("Report saved successfully: " + str(metrics_path))



if __name__ == "__main__":
    main()
