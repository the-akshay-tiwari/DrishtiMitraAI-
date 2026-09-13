"""Evaluate the trained DrishtiMitra model on the held-out test set and output the Efficiency Matrix."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.train import CLASS_NAMES, FundusDataset, build_model, load_samples, make_transforms


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = PROJECT_ROOT / "artifacts" / "drishtimitra_efficientnet_b0.pt"

    if not checkpoint_path.is_file():
        print(f"Error: Model checkpoint not found at {checkpoint_path}. Run training first!")
        sys.exit(1)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    _, _, test_samples = load_samples("auto")
    _, eval_transform = make_transforms()
    test_loader = DataLoader(FundusDataset(test_samples, eval_transform), batch_size=1, shuffle=False)

    targets: list[int] = []
    predictions: list[int] = []
    latencies: list[float] = []

    with torch.inference_mode():
        for images, labels in test_loader:
            t0 = time.perf_counter()
            logits = model(images.to(device))
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)
            predictions.append(logits.argmax(dim=1).cpu().item())
            targets.append(labels.item())

    t_arr = np.array(targets)
    p_arr = np.array(predictions)

    cm = confusion_matrix(t_arr, p_arr, labels=list(range(5)))
    accuracy = float(accuracy_score(t_arr, p_arr))
    bal_acc = float(balanced_accuracy_score(t_arr, p_arr))
    macro_f1 = float(f1_score(t_arr, p_arr, average="macro", zero_division=0))
    qwk = float(cohen_kappa_score(t_arr, p_arr, weights="quadratic"))

    precision, recall, f1, support = precision_recall_fscore_support(t_arr, p_arr, labels=list(range(5)), zero_division=0)

    # Binary Referral Triage (Grade 0,1 -> Non-Referable; Grade 2,3,4 -> Referable)
    bin_targets = (t_arr >= 2).astype(int)
    bin_preds = (p_arr >= 2).astype(int)
    bin_cm = confusion_matrix(bin_targets, bin_preds)
    tn, fp, fn, tp = bin_cm.ravel()
    ref_sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    ref_specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    ref_ppv = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    ref_npv = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    model_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)

    report = {
        "total_test_samples": len(test_samples),
        "accuracy": round(accuracy, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "quadratic_weighted_kappa": round(qwk, 4),
        "avg_latency_ms": round(float(np.mean(latencies)), 2),
        "model_params_m": round(total_params, 2),
        "checkpoint_size_mb": round(model_size_mb, 2),
        "confusion_matrix_5x5": cm.tolist(),
        "per_class": {
            CLASS_NAMES[i]: {
                "precision": round(float(precision[i]), 4),
                "recall_sensitivity": round(float(recall[i]), 4),
                "f1_score": round(float(f1[i]), 4),
                "support": int(support[i]),
            } for i in range(5)
        },
        "referral_triage": {
            "sensitivity": round(ref_sensitivity, 4),
            "specificity": round(ref_specificity, 4),
            "ppv": round(ref_ppv, 4),
            "npv": round(ref_npv, 4),
            "matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        },
    }

    print("\n=======================================================")
    print("      DRISHTIMITRA MODEL EFFICIENCY MATRIX REPORT      ")
    print("=======================================================")
    print(f"Total Test Samples          : {report['total_test_samples']}")
    print(f"Quadratic Weighted Kappa QWK : {report['quadratic_weighted_kappa']}")
    print(f"Accuracy                    : {report['accuracy'] * 100:.2f}%")
    print(f"Balanced Accuracy           : {report['balanced_accuracy'] * 100:.2f}%")
    print(f"Macro F1-Score              : {report['macro_f1'] * 100:.2f}%")
    print(f"Average CPU Latency         : {report['avg_latency_ms']} ms/image")
    print(f"Model Parameters            : {report['model_params_m']} M")
    print(f"Checkpoint Size             : {report['checkpoint_size_mb']} MB")

    print("\n--- 5x5 CONFUSION MATRIX ---")
    print("Labels: [Grade 0, Grade 1, Grade 2, Grade 3, Grade 4]")
    print(cm)

    print("\n--- REFERRAL TRIAGE (Grade >= 2) ---")
    print(f"Sensitivity (Recall)        : {report['referral_triage']['sensitivity'] * 100:.2f}%")
    print(f"Specificity                 : {report['referral_triage']['specificity'] * 100:.2f}%")
    print(f"Positive Predictive Value   : {report['referral_triage']['ppv'] * 100:.2f}%")

    eval_out = PROJECT_ROOT / "artifacts" / "evaluation_report.json"
    eval_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved detailed JSON report to {eval_out}\n")


if __name__ == "__main__":
    main()
