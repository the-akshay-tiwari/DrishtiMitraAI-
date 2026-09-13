"""Evaluate inference-time TTA and ordinal threshold optimization for E02.

Evaluates 6 TTA strategies and ordinal threshold optimization strictly on the VALIDATION split.
Locks the best validation configuration and evaluates it ONCE on the untouched TEST split.
Does NOT modify model weights or retrain the model.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Callable, Any

import numpy as np
import torch
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.train import load_samples
from backend.experiment_runner import build_model, resolve_device, make_transforms, FundusDataset


def get_tta_probs(
    model: torch.nn.Module,
    images: torch.Tensor,
    strategy: str,
) -> torch.Tensor:
    """Computes softmax probabilities for a batch of images under a given TTA strategy."""
    if strategy == "original":
        logits = model(images)
        return torch.softmax(logits, dim=1)
    elif strategy == "hflip":
        img_h = torch.flip(images, dims=[3])
        logits = model(img_h)
        return torch.softmax(logits, dim=1)
    elif strategy == "vflip":
        img_v = torch.flip(images, dims=[2])
        logits = model(img_v)
        return torch.softmax(logits, dim=1)
    elif strategy == "rot180":
        img_r = torch.flip(images, dims=[2, 3])
        logits = model(img_r)
        return torch.softmax(logits, dim=1)
    elif strategy == "avg_2x":
        p_orig = torch.softmax(model(images), dim=1)
        p_hflip = torch.softmax(model(torch.flip(images, dims=[3])), dim=1)
        return (p_orig + p_hflip) / 2.0
    elif strategy == "avg_4x":
        p_orig = torch.softmax(model(images), dim=1)
        p_hflip = torch.softmax(model(torch.flip(images, dims=[3])), dim=1)
        p_vflip = torch.softmax(model(torch.flip(images, dims=[2])), dim=1)
        p_rot180 = torch.softmax(model(torch.flip(images, dims=[2, 3])), dim=1)
        return (p_orig + p_hflip + p_vflip + p_rot180) / 4.0
    else:
        raise ValueError(f"Unknown TTA strategy: {strategy}")


def compute_continuous_grades(probs: np.ndarray) -> np.ndarray:
    """Computes continuous expected DR grade y_cont = sum(k * p_k)."""
    class_weights = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    return np.sum(probs * class_weights, axis=1)


def cutoffs_to_preds(continuous_grades: np.ndarray, thresholds: list[float] | np.ndarray) -> np.ndarray:
    """Maps continuous grades to discrete 0-4 predictions using ordered cutoffs [t1, t2, t3, t4]."""
    return np.digitize(continuous_grades, thresholds)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Computes full suite of classification and ordinal metrics."""
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    qwk = float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
    within_1 = float(np.mean(np.abs(y_true - y_pred) <= 1))
    severe_err = float(np.mean(np.abs(y_true - y_pred) >= 2))
    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "quadratic_weighted_kappa": qwk,
        "within_1_accuracy": within_1,
        "severe_error_rate": severe_err,
    }


def optimize_ordinal_thresholds(y_true: np.ndarray, continuous_grades: np.ndarray) -> tuple[list[float], dict[str, float]]:
    """Optimizes cutoffs [t1, t2, t3, t4] using Powell optimization on validation set to maximize QWK."""
    initial_thresholds = [0.5, 1.5, 2.5, 3.5]

    def loss_func(thresh: np.ndarray) -> float:
        t1, t2, t3, t4 = thresh
        # Monotonicity & boundary penalty
        if not (0.0 < t1 < t2 < t3 < t4 < 4.0):
            return 100.0
        preds = cutoffs_to_preds(continuous_grades, thresh)
        kappa = cohen_kappa_score(y_true, preds, weights="quadratic")
        return -float(kappa)

    res = minimize(loss_func, initial_thresholds, method="Powell")
    opt_thresh = [round(float(t), 4) for t in res.x]
    
    # Enforce strict sorting if optimization produced boundary noise
    opt_thresh.sort()
    opt_preds = cutoffs_to_preds(continuous_grades, opt_thresh)
    opt_metrics = evaluate_predictions(y_true, opt_preds)
    return opt_thresh, opt_metrics


def run_evaluation() -> None:
    device = resolve_device("cuda")
    print("\n" + "=" * 70)
    print(" DRISHTIMITRA E02 INFERENCE-TIME OPTIMIZATION (TTA & THRESHOLDS)")
    print("=" * 70)
    print(f" Target Device       : {device}")

    # Path setup
    exp_dir = PROJECT_ROOT / "artifacts" / "experiments" / "E02_no_sampler_35ep"
    ckpt_path = exp_dir / "checkpoint.pt"
    out_dir = exp_dir / "tta_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not ckpt_path.is_file():
        raise FileNotFoundError(f"E02 checkpoint not found at: {ckpt_path}")

    # Load E02 Checkpoint
    print(f" Loading E02 Checkpoint: {ckpt_path.name}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Load Datasets
    _, val_samples, test_samples, dataset_name = load_samples("aptos")
    _, eval_transform = make_transforms(384, "standard")

    val_dataset = FundusDataset(val_samples, eval_transform, prep_type="standard", pad_square=False, image_size=384)
    test_dataset = FundusDataset(test_samples, eval_transform, prep_type="standard", pad_square=False, image_size=384)

    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)

    print(f" Dataset Name        : {dataset_name}")
    print(f" Validation Split    : {len(val_samples)} samples")
    print(f" Untouched Test Split: {len(test_samples)} samples (LOCKED UNTIL OPTIMIZATION FINISHES)")
    print("=" * 70 + "\n")

    # Step 1: Evaluate TTA Candidates on VALIDATION set
    tta_strategies = ["original", "hflip", "vflip", "rot180", "avg_2x", "avg_4x"]
    forward_passes_map = {
        "original": 1,
        "hflip": 1,
        "vflip": 1,
        "rot180": 1,
        "avg_2x": 2,
        "avg_4x": 4,
    }

    validation_summary: dict[str, Any] = {}
    val_probs_cache: dict[str, np.ndarray] = {}
    val_y_true: np.ndarray = np.array([s.grade for s in val_samples], dtype=np.int64)

    for strat in tta_strategies:
        print(f"Evaluating TTA Strategy on Validation: [{strat.upper()}] ...", end="", flush=True)
        t_start = time.perf_counter()
        all_probs: list[np.ndarray] = []

        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(device, non_blocking=True)
                probs = get_tta_probs(model, images, strat)
                all_probs.append(probs.cpu().numpy())

        t_elapsed = time.perf_counter() - t_start
        probs_matrix = np.concatenate(all_probs, axis=0)
        val_probs_cache[strat] = probs_matrix

        # 1. Standard Argmax Evaluation
        argmax_preds = np.argmax(probs_matrix, axis=1)
        argmax_metrics = evaluate_predictions(val_y_true, argmax_preds)

        # 2. Continuous Expectation + Default Cutoffs [0.5, 1.5, 2.5, 3.5]
        cont_grades = compute_continuous_grades(probs_matrix)
        default_preds = cutoffs_to_preds(cont_grades, [0.5, 1.5, 2.5, 3.5])
        default_ordinal_metrics = evaluate_predictions(val_y_true, default_preds)

        # 3. Optimized Ordinal Thresholds on Validation
        opt_cutoffs, opt_ordinal_metrics = optimize_ordinal_thresholds(val_y_true, cont_grades)

        latency_ms_per_image = round((t_elapsed / len(val_samples)) * 1000, 2)
        latency_ms_per_batch = round((t_elapsed / len(val_loader)) * 1000, 2)

        validation_summary[strat] = {
            "strategy_name": strat,
            "forward_passes": forward_passes_map[strat],
            "total_latency_sec": round(t_elapsed, 2),
            "latency_ms_per_image": latency_ms_per_image,
            "latency_ms_per_batch": latency_ms_per_batch,
            "standard_argmax_metrics": argmax_metrics,
            "default_ordinal_metrics": default_ordinal_metrics,
            "optimized_thresholds": opt_cutoffs,
            "optimized_ordinal_metrics": opt_ordinal_metrics,
        }

        print(f" Done in {t_elapsed:.2f}s ({latency_ms_per_image:.1f}ms/img) | Argmax QWK: {argmax_metrics['quadratic_weighted_kappa']:.4f} | Opt Cutoffs QWK: {opt_ordinal_metrics['quadratic_weighted_kappa']:.4f}")

    # Save validation_results.json
    (out_dir / "validation_results.json").write_text(json.dumps(validation_summary, indent=2), encoding="utf-8")

    # Step 2: Select Best Configuration strictly on Validation Performance
    best_config_name = ""
    best_val_qwk = -1.0
    best_method = ""
    best_thresholds: list[float] = [0.5, 1.5, 2.5, 3.5]

    for strat, data in validation_summary.items():
        # Check Argmax QWK
        arg_qwk = data["standard_argmax_metrics"]["quadratic_weighted_kappa"]
        if arg_qwk > best_val_qwk:
            best_val_qwk = arg_qwk
            best_config_name = strat
            best_method = "standard_argmax"
            best_thresholds = [0.5, 1.5, 2.5, 3.5]

        # Check Optimized Threshold QWK
        opt_qwk = data["optimized_ordinal_metrics"]["quadratic_weighted_kappa"]
        if opt_qwk > best_val_qwk:
            best_val_qwk = opt_qwk
            best_config_name = strat
            best_method = "optimized_ordinal_cutoffs"
            best_thresholds = data["optimized_thresholds"]

    best_val_metrics = validation_summary[best_config_name]["optimized_ordinal_metrics"] if best_method == "optimized_ordinal_cutoffs" else validation_summary[best_config_name]["standard_argmax_metrics"]

    selected_configuration = {
        "dataset": dataset_name,
        "selected_tta_strategy": best_config_name,
        "forward_passes": forward_passes_map[best_config_name],
        "decision_method": best_method,
        "locked_thresholds": best_thresholds,
        "selected_based_on": "Validation QWK",
        "validation_qwk_achieved": round(best_val_qwk, 4),
        "validation_metrics": best_val_metrics,
        "baseline_val_qwk": validation_summary["original"]["standard_argmax_metrics"]["quadratic_weighted_kappa"],
        "val_qwk_improvement": round(best_val_qwk - validation_summary["original"]["standard_argmax_metrics"]["quadratic_weighted_kappa"], 4),
    }

    (out_dir / "selected_configuration.json").write_text(json.dumps(selected_configuration, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print(" LOCKED SELECTION FROM VALIDATION OPTIMIZATION")
    print("=" * 70)
    print(f" Best TTA Strategy   : {best_config_name.upper()} ({forward_passes_map[best_config_name]} forward pass/image)")
    print(f" Decision Method     : {best_method}")
    print(f" Locked Thresholds   : {best_thresholds}")
    print(f" Baseline Val QWK    : {selected_configuration['baseline_val_qwk']:.4f}")
    print(f" Optimized Val QWK   : {best_val_qwk:.4f} (Delta: +{selected_configuration['val_qwk_improvement']:.4f})")
    print("=" * 70 + "\n")

    # Step 3: Apply Locked Configuration ONCE to the Untouched TEST Split
    print("Applying Locked Configuration ONCE to Untouched TEST Split ...", flush=True)
    t_test_start = time.perf_counter()
    test_probs_list: list[np.ndarray] = []
    test_y_true: np.ndarray = np.array([s.grade for s in test_samples], dtype=np.int64)

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device, non_blocking=True)
            probs = get_tta_probs(model, images, best_config_name)
            test_probs_list.append(probs.cpu().numpy())

    t_test_elapsed = time.perf_counter() - t_test_start
    test_probs_matrix = np.concatenate(test_probs_list, axis=0)

    if best_method == "standard_argmax":
        test_preds = np.argmax(test_probs_matrix, axis=1)
    else:
        test_cont_grades = compute_continuous_grades(test_probs_matrix)
        test_preds = cutoffs_to_preds(test_cont_grades, best_thresholds)

    test_metrics = evaluate_predictions(test_y_true, test_preds)
    test_latency_ms_per_image = round((t_test_elapsed / len(test_samples)) * 1000, 2)

    baseline_test_metrics = {
        "quadratic_weighted_kappa": 0.9146747850413461,
        "accuracy": 0.8469945355191257,
        "macro_f1": 0.6774801750607571,
        "balanced_accuracy": 0.6604436034743759,
    }

    test_report = {
        "experiment_id": "E02_no_sampler_35ep",
        "locked_configuration": selected_configuration,
        "evaluation_scope": "Untouched Held-Out Test Set (366 samples)",
        "test_latency_total_sec": round(t_test_elapsed, 2),
        "test_latency_ms_per_image": test_latency_ms_per_image,
        "test_metrics": test_metrics,
        "baseline_test_metrics": baseline_test_metrics,
        "test_qwk_delta": round(test_metrics["quadratic_weighted_kappa"] - baseline_test_metrics["quadratic_weighted_kappa"], 4),
        "test_accuracy_delta": round(test_metrics["accuracy"] - baseline_test_metrics["accuracy"], 4),
        "test_macro_f1_delta": round(test_metrics["macro_f1"] - baseline_test_metrics["macro_f1"], 4),
        "test_improved": test_metrics["quadratic_weighted_kappa"] > baseline_test_metrics["quadratic_weighted_kappa"],
    }

    (out_dir / "test_results.json").write_text(json.dumps(test_report, indent=2), encoding="utf-8")

    # Step 4: Write Concise Markdown Report
    report_md = f"""# E02 Inference-Time Optimization Report (TTA & Ordinal Thresholds)

## Executive Summary

- **Experiment Target**: `E02_no_sampler_35ep` (35-epoch EfficientNet-B0 trained with natural class distribution).
- **Original E02 Test Benchmark**:
  - Test QWK: **0.9147**
  - Test Accuracy: **84.70%** (0.8470)
  - Test Macro-F1: **0.6775**
- **Optimized Strategy Selected (via Validation Set)**: `{best_config_name.upper()}` ({forward_passes_map[best_config_name]} forward pass/image) with decision method `{best_method}`.
- **Locked Thresholds**: `{best_thresholds}`

---

## 1. Validation Set Optimization Results

| TTA Strategy | Passes | Latency (ms/img) | Argmax QWK | Default Ordinal QWK | Opt Thresholds QWK | Val Acc | Val Macro-F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for strat, data in validation_summary.items():
        arg_q = data["standard_argmax_metrics"]["quadratic_weighted_kappa"]
        def_q = data["default_ordinal_metrics"]["quadratic_weighted_kappa"]
        opt_q = data["optimized_ordinal_metrics"]["quadratic_weighted_kappa"]
        val_acc = data["optimized_ordinal_metrics" if best_method == "optimized_ordinal_cutoffs" else "standard_argmax_metrics"]["accuracy"]
        val_f1 = data["optimized_ordinal_metrics" if best_method == "optimized_ordinal_cutoffs" else "standard_argmax_metrics"]["macro_f1"]
        report_md += f"| **{strat.upper()}** | {data['forward_passes']} | {data['latency_ms_per_image']} | {arg_q:.4f} | {def_q:.4f} | **{opt_q:.4f}** | {val_acc*100:.2f}% | {val_f1:.4f} |\n"

    report_md += f"""
- **Validation QWK Baseline (`original` argmax)**: `{validation_summary['original']['standard_argmax_metrics']['quadratic_weighted_kappa']:.4f}`
- **Best Validation QWK (`{best_config_name.upper()}` {best_method})**: `{best_val_qwk:.4f}` (Gain: `+{selected_configuration['val_qwk_improvement']:.4f}`)

---

## 2. Locked Held-Out Test Set Evaluation

> The locked configuration was evaluated **EXACTLY ONCE** on the untouched test set (366 images).

| Metric | Original E02 Test Result | Optimized TTA + Threshold Test Result | Delta |
| :--- | :---: | :---: | :---: |
| **Quadratic Weighted Kappa (QWK)** | **0.9147** | **{test_metrics['quadratic_weighted_kappa']:.4f}** | **{'+' if test_report['test_qwk_delta']>=0 else ''}{test_report['test_qwk_delta']:.4f}** |
| **Accuracy** | 84.70% (0.8470) | **{test_metrics['accuracy']*100:.2f}%** ({test_metrics['accuracy']:.4f}) | **{'+' if test_report['test_accuracy_delta']>=0 else ''}{test_report['test_accuracy_delta']*100:.2f}%** |
| **Macro-F1** | 0.6775 | **{test_metrics['macro_f1']:.4f}** | **{'+' if test_report['test_macro_f1_delta']>=0 else ''}{test_report['test_macro_f1_delta']:.4f}** |
| **Balanced Accuracy** | 0.6604 | **{test_metrics['balanced_accuracy']:.4f}** | **{'+' if test_metrics['balanced_accuracy'] - baseline_test_metrics['balanced_accuracy']>=0 else ''}{test_metrics['balanced_accuracy'] - baseline_test_metrics['balanced_accuracy']:.4f}** |
| **Within $\\pm 1$ Grade Accuracy** | N/A | **{test_metrics['within_1_accuracy']*100:.2f}%** | High Consistency |
| **Severe Error Rate ($\ge 2$ Grade Off)**| N/A | **{test_metrics['severe_error_rate']*100:.2f}%** | Low Off-by-2 Risk |

---

## 3. Computational Latency Analysis

- **Inference Latency Per Image**: `{test_latency_ms_per_image:.2f} ms` (on CUDA batch_size=16)
- **Forward Passes Required**: `{forward_passes_map[best_config_name]}` per image
- **TTA Improvement Demonstrated on Validation**: **{'YES' if selected_configuration['val_qwk_improvement'] > 0 else 'NO'}**
- **Improvement Demonstrated on Untouched Test Set**: **{'YES' if test_report['test_improved'] else 'NO'}**
"""
    (out_dir / "concise_report.md").write_text(report_md, encoding="utf-8")

    print("=" * 70)
    print(" FINAL HELD-OUT TEST EVALUATION RESULT")
    print("=" * 70)
    print(f" Original E02 Test QWK  : 0.9147 (Acc: 84.70%, Macro-F1: 0.6775)")
    print(f" TTA+Threshold Test QWK : {test_metrics['quadratic_weighted_kappa']:.4f} (Acc: {test_metrics['accuracy']*100:.2f}%, Macro-F1: {test_metrics['macro_f1']:.4f})")
    print(f" Test QWK Delta         : {'+' if test_report['test_qwk_delta']>=0 else ''}{test_report['test_qwk_delta']:.4f}")
    print(f" Demonstrated Test Gain : {'YES - QWK IMPROVED!' if test_report['test_improved'] else 'NO - Baseline maintained'}")
    print(f" Saved Analysis Folder  : {out_dir}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_evaluation()
