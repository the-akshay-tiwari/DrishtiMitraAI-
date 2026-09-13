"""Phase 2: Baseline Error Analysis Script.

Evaluates the existing baseline checkpoint (artifacts/drishtimitra_efficientnet_b0.pt)
without retraining and generates all Phase 2 error analysis artifacts.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.train import CLASS_NAMES, FundusDataset, build_model, load_samples, make_transforms


def plot_confusion_matrix(cm: np.ndarray, labels: list[str], title: str, output_path: Path, normalize: bool = False) -> None:
    plt.figure(figsize=(8, 6))
    if normalize:
        cm_float = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_float, annot=True, fmt=".2f", cmap="Blues", xticklabels=labels, yticklabels=labels)
    else:
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel("True Grade")
    plt.xlabel("Predicted Grade")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = PROJECT_ROOT / "artifacts" / "drishtimitra_efficientnet_b0.pt"

    if not checkpoint_path.is_file():
        print(f"Error: Baseline checkpoint not found at {checkpoint_path}")
        sys.exit(1)

    print(f"Loading baseline checkpoint: {checkpoint_path} on {device}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = build_model(pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    train_samples, val_samples, test_samples, dataset_name = load_samples("auto")
    _, eval_transform = make_transforms()

    val_loader = DataLoader(FundusDataset(val_samples, eval_transform), batch_size=1, shuffle=False)
    test_loader = DataLoader(FundusDataset(test_samples, eval_transform), batch_size=1, shuffle=False)

    def run_eval(loader: DataLoader, samples: list):
        targets: list[int] = []
        predictions: list[int] = []
        confidences: list[float] = []

        with torch.inference_mode():
            for i, (images, labels) in enumerate(loader):
                logits = model(images.to(device))
                probs = F.softmax(logits, dim=1)[0].cpu()
                pred = int(probs.argmax().item())
                conf = float(probs[pred].item())

                predictions.append(pred)
                targets.append(labels.item())
                confidences.append(conf)

        return np.array(targets), np.array(predictions), np.array(confidences)

    val_targets, val_preds, val_confs = run_eval(val_loader, val_samples)
    test_targets, test_preds, test_confs = run_eval(test_loader, test_samples)

    # Calculate metrics
    def compute_all_metrics(t_arr: np.ndarray, p_arr: np.ndarray):
        cm = confusion_matrix(t_arr, p_arr, labels=list(range(5)))
        acc = float(accuracy_score(t_arr, p_arr))
        bal_acc = float(balanced_accuracy_score(t_arr, p_arr))
        macro_f1 = float(f1_score(t_arr, p_arr, average="macro", zero_division=0))
        qwk = float(cohen_kappa_score(t_arr, p_arr, weights="quadratic"))
        prec, rec, f1, supp = precision_recall_fscore_support(t_arr, p_arr, labels=list(range(5)), zero_division=0)
        return cm, acc, bal_acc, macro_f1, qwk, prec, rec, f1, supp

    val_cm, val_acc, val_bal_acc, val_macro_f1, val_qwk, val_prec, val_rec, val_f1, val_supp = compute_all_metrics(val_targets, val_preds)
    test_cm, test_acc, test_bal_acc, test_macro_f1, test_qwk, test_prec, test_rec, test_f1, test_supp = compute_all_metrics(test_targets, test_preds)

    # Save artifacts/baseline_metrics.json
    baseline_metrics = {
        "model": "EfficientNet-B0 ImageNet transfer learning",
        "dataset": dataset_name,
        "best_epoch": checkpoint.get("best_epoch", 29),
        "validation_metrics": {
            "accuracy": val_acc,
            "balanced_accuracy": val_bal_acc,
            "macro_f1": val_macro_f1,
            "quadratic_weighted_kappa": val_qwk,
        },
        "held_out_test_metrics": {
            "accuracy": test_acc,
            "balanced_accuracy": test_bal_acc,
            "macro_f1": test_macro_f1,
            "quadratic_weighted_kappa": test_qwk,
        },
    }
    (PROJECT_ROOT / "artifacts" / "baseline_metrics.json").write_text(json.dumps(baseline_metrics, indent=2), encoding="utf-8")

    # Generate Confusion Matrix Plots (Validation Set & Test Set)
    plot_confusion_matrix(test_cm, CLASS_NAMES, f"Held-out Test Confusion Matrix (QWK: {test_qwk:.4f})", PROJECT_ROOT / "artifacts" / "confusion_matrix.png", normalize=False)
    plot_confusion_matrix(test_cm, CLASS_NAMES, f"Held-out Test Normalized Confusion Matrix (QWK: {test_qwk:.4f})", PROJECT_ROOT / "artifacts" / "normalized_confusion_matrix.png", normalize=True)

    # Per-Class Metrics JSON
    per_class_metrics = {
        "validation": {
            CLASS_NAMES[i]: {
                "precision": float(val_prec[i]),
                "recall_sensitivity": float(val_rec[i]),
                "f1_score": float(val_f1[i]),
                "support": int(val_supp[i]),
            } for i in range(5)
        },
        "test": {
            CLASS_NAMES[i]: {
                "precision": float(test_prec[i]),
                "recall_sensitivity": float(test_rec[i]),
                "f1_score": float(test_f1[i]),
                "support": int(test_supp[i]),
            } for i in range(5)
        },
    }
    (PROJECT_ROOT / "artifacts" / "per_class_metrics.json").write_text(json.dumps(per_class_metrics, indent=2), encoding="utf-8")

    # Misclassified Samples List (from Validation Set for development & Test Set)
    misclassified_csv = PROJECT_ROOT / "artifacts" / "misclassified_samples.csv"
    misclassified_count = 0
    with misclassified_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "image_id", "true_grade", "predicted_grade", "confidence", "image_path"])
        for split_name, samples, t_arr, p_arr, c_arr in [("val", val_samples, val_targets, val_preds, val_confs), ("test", test_samples, test_targets, test_preds, test_confs)]:
            for i, sample in enumerate(samples):
                if t_arr[i] != p_arr[i]:
                    misclassified_count += 1
                    writer.writerow([
                        split_name,
                        sample.image_path.stem,
                        int(t_arr[i]),
                        int(p_arr[i]),
                        round(float(c_arr[i]), 4),
                        str(sample.image_path),
                    ])

    # Ordinal Error Statistics
    def compute_ordinal_stats(t_arr: np.ndarray, p_arr: np.ndarray):
        diffs = np.abs(t_arr - p_arr)
        total = len(t_arr)
        exact = float(np.sum(diffs == 0) / total)
        within_1 = float(np.sum(diffs <= 1) / total)
        greater_than_1 = float(np.sum(diffs > 1) / total)
        return {
            "total_samples": total,
            "exact_prediction_rate": round(exact, 4),
            "within_1_grade_accuracy": round(within_1, 4),
            "errors_greater_than_1_grade": round(greater_than_1, 4),
            "grade_difference_counts": {str(k): int(v) for k, v in zip(*np.unique(diffs, return_counts=True))},
        }

    val_ordinal = compute_ordinal_stats(val_targets, val_preds)
    test_ordinal = compute_ordinal_stats(test_targets, test_preds)

    error_analysis = {
        "validation_ordinal_stats": val_ordinal,
        "test_ordinal_stats": test_ordinal,
        "prediction_distribution_test": {str(g): int(np.sum(test_preds == g)) for g in range(5)},
        "true_label_distribution_test": {str(g): int(np.sum(test_targets == g)) for g in range(5)},
        "total_misclassified_test_samples": int(np.sum(test_targets != test_preds)),
    }
    (PROJECT_ROOT / "artifacts" / "error_analysis.json").write_text(json.dumps(error_analysis, indent=2), encoding="utf-8")

    print("\nPhase 2 Baseline Error Analysis Complete!")
    print(f"Validation QWK: {val_qwk:.4f} | Test QWK: {test_qwk:.4f}")
    print(f"Held-out Test Exact Prediction Rate: {test_ordinal['exact_prediction_rate']*100:.2f}%")
    print(f"Held-out Test ±1 Grade Accuracy: {test_ordinal['within_1_grade_accuracy']*100:.2f}%")
    print(f"Errors > 1 Grade: {test_ordinal['errors_greater_than_1_grade']*100:.2f}%")
    print(f"Saved artifacts to {PROJECT_ROOT / 'artifacts'}\n")


if __name__ == "__main__":
    main()
