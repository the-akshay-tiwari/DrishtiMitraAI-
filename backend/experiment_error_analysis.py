"""Experiment-Specific Error Analysis Script for DrishtiMitra.

Evaluates trained experiment checkpoints (e.g., artifacts/experiments/E01_randaug/checkpoint.pt)
without retraining and generates all experiment-specific error analysis artifacts under:
artifacts/experiments/<exp-id>/analysis/
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.train import CLASS_NAMES, FundusDataset, build_model, load_samples, make_transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DrishtiMitra Experiment Error Analysis")
    parser.add_argument("--exp-id", type=str, required=True, help="Experiment ID (e.g. E01_randaug)")
    parser.add_argument("--dataset", choices=["auto", "aptos", "idrid"], default="aptos", help="Dataset choice")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for inference")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    return parser.parse_args()


def draw_confusion_matrix_png(cm: np.ndarray, labels: list[str], title: str, output_path: Path, normalize: bool = False) -> None:
    cell_size = 90
    left_margin = 160
    top_margin = 80
    right_margin = 40
    bottom_margin = 60

    width = left_margin + len(labels) * cell_size + right_margin
    height = top_margin + len(labels) * cell_size + bottom_margin

    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Title
    draw.text((width // 2 - len(title) * 4, 25), title, fill=(20, 20, 20))

    if normalize:
        cm_data = cm.astype("float") / np.maximum(cm.sum(axis=1)[:, np.newaxis], 1e-12)
        max_val = 1.0
    else:
        cm_data = cm.astype("float")
        max_val = max(1.0, float(cm.max()))

    for i in range(len(labels)):
        # Y-axis label (True Grade)
        y_pos = top_margin + i * cell_size + cell_size // 2 - 6
        draw.text((15, y_pos), f"True: {labels[i]}", fill=(30, 30, 30))

        for j in range(len(labels)):
            # X-axis label (Predicted Grade)
            if i == 0:
                x_pos = left_margin + j * cell_size + 10
                draw.text((x_pos, top_margin - 30), f"Pred {j}", fill=(30, 30, 30))

            val = cm_data[i, j]
            raw_val = cm[i, j]
            ratio = float(val / max_val)

            # Color gradient: Light blue to dark navy
            r = int(240 - ratio * (240 - 25))
            g = int(248 - ratio * (248 - 80))
            b = int(255 - ratio * (255 - 165))

            x0 = left_margin + j * cell_size
            y0 = top_margin + i * cell_size
            x1 = x0 + cell_size
            y1 = y0 + cell_size

            draw.rectangle([x0, y0, x1, y1], fill=(r, g, b), outline=(210, 210, 210), width=1)

            text_str = f"{val:.2f}" if normalize else f"{raw_val}"
            txt_color = (255, 255, 255) if ratio > 0.45 else (20, 20, 20)
            draw.text((x0 + cell_size // 2 - len(text_str) * 3, y0 + cell_size // 2 - 6), text_str, fill=txt_color)

    img.save(output_path)


def compute_metrics(t_arr: np.ndarray, p_arr: np.ndarray):
    cm = confusion_matrix(t_arr, p_arr, labels=list(range(5)))
    acc = float(accuracy_score(t_arr, p_arr))
    bal_acc = float(balanced_accuracy_score(t_arr, p_arr))
    macro_f1 = float(f1_score(t_arr, p_arr, average="macro", zero_division=0))
    qwk = float(cohen_kappa_score(t_arr, p_arr, weights="quadratic"))
    prec, rec, f1, supp = precision_recall_fscore_support(t_arr, p_arr, labels=list(range(5)), zero_division=0)
    return cm, acc, bal_acc, macro_f1, qwk, prec, rec, f1, supp


def compute_ordinal_stats(t_arr: np.ndarray, p_arr: np.ndarray):
    diffs = np.abs(t_arr - p_arr)
    total = len(t_arr)
    exact = float(np.sum(diffs == 0) / total) if total > 0 else 0.0
    within_1 = float(np.sum(diffs <= 1) / total) if total > 0 else 0.0
    severe = float(np.sum(diffs > 1) / total) if total > 0 else 0.0
    diff_counts = {str(k): int(v) for k, v in zip(*np.unique(diffs, return_counts=True))}
    return {
        "total_samples": total,
        "exact_prediction_rate": round(exact, 6),
        "within_one_grade_rate": round(within_1, 6),
        "severe_error_rate": round(severe, 6),
        "grade_difference_counts": diff_counts,
        "prediction_distribution": {str(g): int(np.sum(p_arr == g)) for g in range(5)},
        "true_label_distribution": {str(g): int(np.sum(t_arr == g)) for g in range(5)},
    }


def main() -> None:
    args = parse_args()

    # Determine device
    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Checkpoint path verification
    exp_dir = PROJECT_ROOT / "artifacts" / "experiments" / args.exp_id
    checkpoint_path = exp_dir / "checkpoint.pt"
    if not checkpoint_path.is_file():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)

    analysis_dir = exp_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    train_samples, val_samples, test_samples, dataset_name = load_samples(args.dataset)

    # Pre-execution Verification Prints
    print(f"\n=======================================================")
    print(f" EXPERIMENT ERROR ANALYSIS: {args.exp_id}")
    print(f"=======================================================")
    print(f"Device Selection    : {device}")
    print(f"Dataset Name        : {dataset_name}")
    print(f"Checkpoint Path     : {checkpoint_path}")
    print(f"Validation Samples  : {len(val_samples)}")
    print(f"Test Samples        : {len(test_samples)}")
    print(f"Analysis Output Dir : {analysis_dir}\n")

    # Load Model Checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(pretrained=False)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device).eval()

    _, eval_transform = make_transforms()

    val_loader = DataLoader(FundusDataset(val_samples, eval_transform), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(FundusDataset(test_samples, eval_transform), batch_size=args.batch_size, shuffle=False)

    # Inference Runner using Batch Processing
    def run_batch_inference(loader: DataLoader):
        targets: list[int] = []
        predictions: list[int] = []
        confidences: list[float] = []

        with torch.inference_mode():
            for images, labels in loader:
                logits = model(images.to(device))
                probs = F.softmax(logits, dim=1).cpu()
                preds = probs.argmax(dim=1)
                confs = probs.max(dim=1).values

                predictions.extend(preds.tolist())
                targets.extend(labels.tolist())
                confidences.extend(confs.tolist())

        return np.array(targets), np.array(predictions), np.array(confidences)

    val_targets, val_preds, val_confs = run_batch_inference(val_loader)
    test_targets, test_preds, test_confs = run_batch_inference(test_loader)

    # Compute Metrics
    val_cm, val_acc, val_bal_acc, val_macro_f1, val_qwk, val_prec, val_rec, val_f1, val_supp = compute_metrics(val_targets, val_preds)
    test_cm, test_acc, test_bal_acc, test_macro_f1, test_qwk, test_prec, test_rec, test_f1, test_supp = compute_metrics(test_targets, test_preds)

    # 1. metrics.json
    metrics_data = {
        "experiment_id": args.exp_id,
        "dataset": dataset_name,
        "validation": {
            "accuracy": round(val_acc, 6),
            "balanced_accuracy": round(val_bal_acc, 6),
            "macro_f1": round(val_macro_f1, 6),
            "quadratic_weighted_kappa": round(val_qwk, 6),
        },
        "held_out_test": {
            "accuracy": round(test_acc, 6),
            "balanced_accuracy": round(test_bal_acc, 6),
            "macro_f1": round(test_macro_f1, 6),
            "quadratic_weighted_kappa": round(test_qwk, 6),
        },
    }
    (analysis_dir / "metrics.json").write_text(json.dumps(metrics_data, indent=2), encoding="utf-8")

    # 2-5. Confusion Matrices (raw and normalized for Val and Test)
    draw_confusion_matrix_png(val_cm, CLASS_NAMES, f"Validation Confusion Matrix ({args.exp_id})", analysis_dir / "confusion_matrix_val.png", normalize=False)
    draw_confusion_matrix_png(test_cm, CLASS_NAMES, f"Held-out Test Confusion Matrix ({args.exp_id})", analysis_dir / "confusion_matrix_test.png", normalize=False)
    draw_confusion_matrix_png(val_cm, CLASS_NAMES, f"Validation Normalized Confusion Matrix ({args.exp_id})", analysis_dir / "normalized_confusion_matrix_val.png", normalize=True)
    draw_confusion_matrix_png(test_cm, CLASS_NAMES, f"Held-out Test Normalized Confusion Matrix ({args.exp_id})", analysis_dir / "normalized_confusion_matrix_test.png", normalize=True)

    # 6-7. Per-Class Metrics JSON
    per_class_val = {
        CLASS_NAMES[i]: {
            "precision": round(float(val_prec[i]), 6),
            "recall": round(float(val_rec[i]), 6),
            "f1_score": round(float(val_f1[i]), 6),
            "support": int(val_supp[i]),
        } for i in range(5)
    }
    per_class_test = {
        CLASS_NAMES[i]: {
            "precision": round(float(test_prec[i]), 6),
            "recall": round(float(test_rec[i]), 6),
            "f1_score": round(float(test_f1[i]), 6),
            "support": int(test_supp[i]),
        } for i in range(5)
    }
    (analysis_dir / "per_class_metrics_val.json").write_text(json.dumps(per_class_val, indent=2), encoding="utf-8")
    (analysis_dir / "per_class_metrics_test.json").write_text(json.dumps(per_class_test, indent=2), encoding="utf-8")

    # 8-9. Error Analysis JSON
    val_ordinal_stats = compute_ordinal_stats(val_targets, val_preds)
    test_ordinal_stats = compute_ordinal_stats(test_targets, test_preds)

    (analysis_dir / "error_analysis_val.json").write_text(json.dumps(val_ordinal_stats, indent=2), encoding="utf-8")
    (analysis_dir / "error_analysis_test.json").write_text(json.dumps(test_ordinal_stats, indent=2), encoding="utf-8")

    # 10-11. Misclassified Samples CSV (Val and Test)
    def save_misclassified_csv(csv_path: Path, samples: list, t_arr: np.ndarray, p_arr: np.ndarray, c_arr: np.ndarray):
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["image_id", "true_grade", "predicted_grade", "absolute_error", "confidence", "image_path"])
            for i, sample in enumerate(samples):
                if t_arr[i] != p_arr[i]:
                    abs_err = int(abs(t_arr[i] - p_arr[i]))
                    writer.writerow([
                        sample.image_path.stem,
                        int(t_arr[i]),
                        int(p_arr[i]),
                        abs_err,
                        round(float(c_arr[i]), 6),
                        str(sample.image_path),
                    ])

    save_misclassified_csv(analysis_dir / "misclassified_samples_val.csv", val_samples, val_targets, val_preds, val_confs)
    save_misclassified_csv(analysis_dir / "misclassified_samples_test.csv", test_samples, test_targets, test_preds, test_confs)

    # Concise Final Summary Printout
    print("=======================================================")
    print("              EXPERIMENT ANALYSIS SUMMARY              ")
    print("=======================================================")
    print(f"Experiment ID            : {args.exp_id}")
    print(f"Checkpoint path          : {checkpoint_path}")
    print(f"Validation QWK           : {val_qwk:.6f}")
    print(f"Test QWK                 : {test_qwk:.6f}")
    print(f"Validation Accuracy      : {val_acc * 100:.2f}%")
    print(f"Test Accuracy            : {test_acc * 100:.2f}%")
    print(f"Validation Macro-F1      : {val_macro_f1:.6f}")
    print(f"Test Macro-F1            : {test_macro_f1:.6f}")
    print(f"Exact prediction rate    : Val {val_ordinal_stats['exact_prediction_rate']*100:.2f}% | Test {test_ordinal_stats['exact_prediction_rate']*100:.2f}%")
    print(f"Within ±1 grade rate     : Val {val_ordinal_stats['within_one_grade_rate']*100:.2f}% | Test {test_ordinal_stats['within_one_grade_rate']*100:.2f}%")
    print(f"Severe error rate        : Val {val_ordinal_stats['severe_error_rate']*100:.2f}% | Test {test_ordinal_stats['severe_error_rate']*100:.2f}%")
    print(f"Output directory         : {analysis_dir}")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
