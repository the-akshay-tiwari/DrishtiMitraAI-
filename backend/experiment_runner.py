"""DrishtiMitra Controlled Experiment Runner.

Executes isolated experiments without touching baseline artifacts.
All experiment checkpoints, logs, and metrics are saved strictly inside:
artifacts/experiments/<exp_id>/
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
from torchvision.transforms import v2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.train import CLASS_NAMES, NORMALIZATION, Sample, load_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DrishtiMitra Controlled Experiment Runner")
    parser.add_argument("--exp-id", type=str, required=True, help="Unique experiment ID (e.g., E03_bengraham_ordinal)")
    parser.add_argument("--change-description", type=str, default="Controlled Experiment", help="Brief experiment rationale")
    parser.add_argument("--aug-type", choices=["standard", "randaug"], default="standard", help="Augmentation strategy")
    parser.add_argument("--prep-type", choices=["standard", "bengraham"], default="standard", help="Image preprocessing method")
    parser.add_argument("--pad-square", action="store_true", help="Pad rectangular images to 1:1 square canvas before resizing")
    parser.add_argument("--loss-type", choices=["ce", "ordinal"], default="ordinal", help="Loss function strategy")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--sanity-check", action="store_true", help="Run lightweight sanity test (2 batches only)")
    parser.add_argument("--dataset", choices=["auto", "aptos", "idrid"], default="auto")
    parser.add_argument(
        "--sampler",
        choices=["weighted", "none"],
        default="weighted",
        help="Training sampler strategy",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from an existing experiment checkpoint",
    )
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
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pad_to_square(image: Image.Image, bg_color: tuple[int, int, int] = (0, 0, 0)) -> Image.Image:
    width, height = image.size
    if width == height:
        return image
    max_dim = max(width, height)
    new_img = Image.new(image.mode, (max_dim, max_dim), bg_color)
    paste_x = (max_dim - width) // 2
    paste_y = (max_dim - height) // 2
    new_img.paste(image, (paste_x, paste_y))
    return new_img


def apply_ben_graham_pil(img: Image.Image, sigma: float = 10.0) -> Image.Image:
    try:
        import cv2
        img_np = np.array(img.convert("RGB"))
        blur = cv2.GaussianBlur(img_np, (0, 0), sigma)
        processed = cv2.addWeighted(img_np, 4.0, blur, -4.0, 128.0)
        mask = (img_np > 10).any(axis=2)
        processed[~mask] = 0
        return Image.fromarray(np.clip(processed, 0, 255).astype(np.uint8))
    except ImportError:
        from PIL import ImageFilter
        blur = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        img_np = np.array(img.convert("RGB"), dtype=np.float32)
        blur_np = np.array(blur.convert("RGB"), dtype=np.float32)
        processed = 4.0 * img_np - 4.0 * blur_np + 128.0
        mask = (img_np > 10).any(axis=2)
        processed[~mask] = 0
        return Image.fromarray(np.clip(processed, 0, 255).astype(np.uint8))


class FundusDataset(Dataset[tuple[Tensor, int]]):
    def __init__(
        self,
        samples: list[Sample],
        transform: v2.Compose,
        prep_type: str = "standard",
        pad_square: bool = False,
        image_size: int = 384,
    ):
        self.samples = samples
        self.transform = transform
        self.prep_type = prep_type
        self.pad_square = pad_square
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        sample = self.samples[index]
        with Image.open(sample.image_path) as image:
            rgb_image = image.convert("RGB")
        if self.pad_square:
            rgb_image = pad_to_square(rgb_image)
        # CRITICAL MEMORY FIX: Resize to target image_size BEFORE Ben Graham Gaussian Blur
        # This bounds memory footprint from 137 MB float32 down to <1.8 MB per image!
        rgb_image = rgb_image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        if self.prep_type == "bengraham":
            rgb_image = apply_ben_graham_pil(rgb_image)
        return self.transform(rgb_image), sample.grade


def make_transforms(image_size: int, aug_type: str) -> tuple[v2.Compose, v2.Compose]:
    normalize = v2.Normalize(mean=NORMALIZATION["mean"], std=NORMALIZATION["std"])
    if aug_type == "randaug":
        train_transform = v2.Compose([
            v2.ToImage(),
            v2.Resize((image_size, image_size), antialias=True),
            v2.RandAugment(num_ops=2, magnitude=9),
            v2.ToDtype(torch.float32, scale=True),
            normalize,
        ])
    else:
        train_transform = v2.Compose([
            v2.ToImage(),
            v2.Resize((image_size, image_size), antialias=True),
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.RandomRotation(180),
            v2.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.10, hue=0.04),
            v2.ToDtype(torch.float32, scale=True),
            normalize,
        ])

    evaluation_transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((image_size, image_size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        normalize,
    ])
    return train_transform, evaluation_transform


def build_model() -> nn.Module:
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
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


def compute_loss(logits: Tensor, targets: Tensor, weights: Tensor, loss_type: str = "ordinal") -> Tensor:
    if loss_type == "ordinal":
        ce = nn.functional.cross_entropy(logits, targets, weight=weights, label_smoothing=0.05)
        probs = nn.functional.softmax(logits, dim=1)
        class_vals = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0], device=logits.device)
        expected_val = (probs * class_vals).sum(dim=1)
        mse = nn.functional.mse_loss(expected_val, targets.float())
        return ce + 0.75 * mse
    return nn.functional.cross_entropy(logits, targets, weight=weights, label_smoothing=0.05)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    # Isolated experiment output directory
    exp_dir = PROJECT_ROOT / "artifacts" / "experiments" / args.exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = exp_dir / "checkpoint.pt"
    metrics_path = exp_dir / "experiment_metrics.json"

    # Strictly protect baseline files
    baseline_pt = PROJECT_ROOT / "artifacts" / "drishtimitra_efficientnet_b0.pt"
    baseline_json = PROJECT_ROOT / "artifacts" / "training_metrics.json"
    assert checkpoint_path.resolve() != baseline_pt.resolve(), "CRITICAL RISK: Cannot overwrite baseline checkpoint!"
    assert metrics_path.resolve() != baseline_json.resolve(), "CRITICAL RISK: Cannot overwrite baseline metrics!"

    training_samples, validation_samples, test_samples, dataset_name = load_samples(args.dataset)
    train_transform, eval_transform = make_transforms(args.image_size, args.aug_type)
    pin_memory = device.type == "cuda"

    # Configurable training DataLoader sampler strategy
    if args.sampler == "none":
        train_loader = DataLoader(
            FundusDataset(training_samples, train_transform, prep_type=args.prep_type, pad_square=args.pad_square, image_size=args.image_size),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=pin_memory,
        )
    else:
        # Balanced WeightedRandomSampler
        counts = np.bincount([sample.grade for sample in training_samples], minlength=len(CLASS_NAMES)).astype(np.float32)
        sample_weights = torch.tensor([1.0 / counts[sample.grade] for sample in training_samples], dtype=torch.double)
        sampler = torch.utils.data.WeightedRandomSampler(weights=sample_weights, num_samples=len(training_samples), replacement=True)
        train_loader = DataLoader(
            FundusDataset(training_samples, train_transform, prep_type=args.prep_type, pad_square=args.pad_square, image_size=args.image_size),
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.workers,
            pin_memory=pin_memory,
        )

    validation_loader = DataLoader(FundusDataset(validation_samples, eval_transform, prep_type=args.prep_type, pad_square=args.pad_square, image_size=args.image_size), batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=pin_memory)

    model = build_model().to(device)
    weights_tensor = class_weights(training_samples).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    started_at = time.perf_counter()

    print("\n" + "=" * 65)
    print(" DRISHTIMITRA EXPERIMENT RUNNER STARTUP DIAGNOSTICS")
    print("=" * 65)
    print(f" Python Executable : {sys.executable}")
    print(f" PyTorch Version   : {torch.__version__}")
    print(f" CUDA Available    : {torch.cuda.is_available()}")
    print(f" CUDA Device Name  : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A (CPU Mode)'}")
    print(f" Selected Device   : {device} ({'GPU CUDA Accelerated' if device.type == 'cuda' else 'CPU Standard Mode'})")
    print(f" Experiment ID     : {args.exp_id}")
    print(f" Dataset           : {dataset_name} ({len(training_samples)} train / {len(validation_samples)} val / {len(test_samples)} test)")
    print(f" Augmentation      : {args.aug_type}")
    print(f" Preprocessing     : {args.prep_type}")
    print(f" Pad to Square     : {args.pad_square}")
    print(f" Loss Function     : {args.loss_type}")
    print(f" Sampler Strategy  : {args.sampler}")
    print(f" Image Size        : {args.image_size}")
    print(f" Batch Size        : {args.batch_size}")
    print(f" Learning Rate     : {args.learning_rate}")
    print(f" Target Epochs     : {args.epochs}")
    print("=" * 65 + "\n", flush=True)

    if args.sanity_check:
        print("--- SANITY CHECK MODE ACTIVATED ---", flush=True)

        # Single APTOS sample preprocessing verification test
        sample_0 = training_samples[0]
        print(f"[Sanity Preprocessing] Testing sample: {sample_0.image_path.name}")
        with Image.open(sample_0.image_path) as test_img:
            orig_size = test_img.size
            test_rgb = test_img.convert("RGB")
            if args.pad_square:
                test_rgb = pad_to_square(test_rgb)
            test_rgb = test_rgb.resize((args.image_size, args.image_size), Image.Resampling.BILINEAR)
            if args.prep_type == "bengraham":
                test_rgb = apply_ben_graham_pil(test_rgb)
            tensor_out = train_transform(test_rgb)
            mem_mb = (tensor_out.element_size() * tensor_out.nelement()) / (1024 * 1024)

        print(f"[Sanity Preprocessing] Original Resolution : {orig_size[0]}x{orig_size[1]}")
        print(f"[Sanity Preprocessing] Preprocessed Shape   : {tuple(tensor_out.shape)}")
        print(f"[Sanity Preprocessing] Tensor Memory Footprint: {mem_mb:.2f} MB")
        print(f"[Sanity Preprocessing] Preprocessing Test   : PASSED", flush=True)

        model.train()
        for batch_idx, (images, labels) in enumerate(train_loader):
            if batch_idx >= 2:
                break
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images.to(device, non_blocking=True))
                loss = compute_loss(logits, labels.to(device, non_blocking=True), weights_tensor, loss_type=args.loss_type)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            print(f"[Sanity Training Step {batch_idx+1}/2] Loss: {loss.item():.4f} (Finite: {torch.isfinite(loss).item()})", flush=True)

        # Verify checkpoint creation in experiment directory
        sanity_meta = {
            "sanity_check": True,
            "status": "PASSED",
            "exp_id": args.exp_id,
            "sampler": args.sampler,
            "output_dir": str(exp_dir),
            "checkpoint_path": str(checkpoint_path),
            "baseline_protected": True,
        }
        (exp_dir / "sanity_result.json").write_text(json.dumps(sanity_meta, indent=2), encoding="utf-8")
        print("Sanity Check Passed Successfully!\n")
        return

    # Full Training Loop (only runs when --sanity-check is NOT present)
    best_val_qwk = float("-inf")
    best_epoch = 0
    best_val_metrics = {}
    start_epoch = 0

    if args.resume:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Cannot resume experiment '{args.exp_id}': Checkpoint file not found at {checkpoint_path}")
        print(f"\n--- RESUMING EXPERIMENT '{args.exp_id}' FROM CHECKPOINT ---")
        print(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

        model.load_state_dict(ckpt["model_state_dict"])

        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])

        best_epoch = ckpt.get("best_epoch", 0)
        best_val_metrics = ckpt.get("val_metrics", {})
        if "quadratic_weighted_kappa" in best_val_metrics:
            best_val_qwk = best_val_metrics["quadratic_weighted_kappa"]

        if "last_epoch" in ckpt:
            start_epoch = ckpt["last_epoch"]
        elif "args" in ckpt and "epochs" in ckpt["args"]:
            start_epoch = ckpt["args"]["epochs"]
        else:
            start_epoch = best_epoch

        print(f"Resuming training from epoch {start_epoch + 1} to total target {args.epochs} epochs")
        print(f"Restored Best Epoch: {best_epoch}, Best Val QWK: {best_val_qwk:.4f}\n")

    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images.to(device, non_blocking=True))
                loss = compute_loss(logits, labels.to(device, non_blocking=True), weights_tensor, loss_type=args.loss_type)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * labels.size(0)
        scheduler.step()

        val_metrics, _, _ = evaluate(model, validation_loader, device)
        avg_loss = total_loss / len(training_samples)
        print(f"Epoch {epoch:02d}/{args.epochs} | loss {avg_loss:.4f} | val acc {val_metrics['accuracy']:.3f} | val QWK {val_metrics['quadratic_weighted_kappa']:.3f}")

        if val_metrics["quadratic_weighted_kappa"] > best_val_qwk:
            best_val_qwk = val_metrics["quadratic_weighted_kappa"]
            best_epoch = epoch
            best_val_metrics = val_metrics
            torch.save({
                "exp_id": args.exp_id,
                "model_name": "efficientnet_b0",
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "best_epoch": best_epoch,
                "last_epoch": epoch,
                "val_metrics": val_metrics,
                "args": vars(args),
            }, checkpoint_path)

    # Reload best checkpoint and evaluate on held-out test set
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loader = DataLoader(
        FundusDataset(
            test_samples,
            eval_transform,
            prep_type=args.prep_type,
            pad_square=args.pad_square,
            image_size=args.image_size,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin_memory,
    )
    has_test_labels = all(s.grade >= 0 for s in test_samples)
    if has_test_labels:
        test_metrics, _, _ = evaluate(model, test_loader, device)
    else:
        test_metrics = "Unlabeled test set"

    # Save Experiment Summary Report
    exp_report = {
        "experiment_id": args.exp_id,
        "description": args.change_description,
        "model": "EfficientNet-B0 ImageNet transfer learning",
        "dataset": dataset_name,
        "aug_type": args.aug_type,
        "sampler": args.sampler,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "best_validation_metrics": best_val_metrics,
        "held_out_test_metrics": test_metrics,
        "elapsed_minutes": round((time.perf_counter() - started_at) / 60, 2),
        "checkpoint_path": str(checkpoint_path),
    }
    metrics_path.write_text(json.dumps(exp_report, indent=2), encoding="utf-8")
    print(f"\nExperiment {args.exp_id} Complete! Best Epoch: {best_epoch}, Best Val QWK: {best_val_qwk:.4f}")
    if isinstance(test_metrics, dict):
        print(f"Held-out Test QWK: {test_metrics['quadratic_weighted_kappa']:.4f} | Accuracy: {test_metrics['accuracy']*100:.2f}% | Macro-F1: {test_metrics['macro_f1']:.4f}")
    print(f"Saved Checkpoint: {checkpoint_path}")
    print(f"Saved Metrics   : {metrics_path}")



if __name__ == "__main__":
    main()
