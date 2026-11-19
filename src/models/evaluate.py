from typing import Dict, Any, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)
from sklearn.calibration import calibration_curve


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.50,
) -> Dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    try:
        auc_score = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc_score = 0.50

    metrics = {
        "accuracy": float(round(accuracy_score(y_true, y_pred), 4)),
        "precision": float(round(precision_score(y_true, y_pred, zero_division=0), 4)),
        "recall": float(round(recall_score(y_true, y_pred, zero_division=0), 4)),
        "f1": float(round(f1_score(y_true, y_pred, zero_division=0), 4)),
        "roc_auc": float(round(auc_score, 4)),
        "brier_score": float(round(brier_score_loss(y_true, y_prob), 4)),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "threshold": float(round(threshold, 4)),
    }
    return metrics


def sweep_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    start: float = 0.10,
    stop: float = 0.90,
    num: int = 17,
) -> pd.DataFrame:
    threshold_values = np.linspace(start, stop, num)
    records = []

    for th in threshold_values:
        y_pred = (y_prob >= th).astype(int)
        rec = {
            "threshold": round(float(th), 3),
            "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        }
        records.append(rec)

    return pd.DataFrame(records)


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> float:
    sweep_df = sweep_thresholds(y_true, y_prob, start=0.15, stop=0.85, num=71)
    best_row = sweep_df.sort_values(by="f1", ascending=False).iloc[0]
    return float(best_row["threshold"])


def plot_roc_curve_eval(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: Path,
    model_name: str = "Best Model",
) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"{model_name} (AUC = {auc_val:.3f})")
    ax.plot([0, 1], [0, 1], color="#7f7f7f", lw=1.5, linestyle="--", label="Random Chance (0.50)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=10)
    ax.set_title("Receiver Operating Characteristic (ROC)", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_precision_recall_curve_eval(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: Path,
    model_name: str = "Best Model",
) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    baseline = np.mean(y_true)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color="#2ca02c", lw=2, label=f"{model_name}")
    ax.axhline(baseline, color="#d62728", linestyle="--", label=f"Baseline Prevalence ({baseline:.2f})")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall", fontsize=10)
    ax.set_ylabel("Precision", fontsize=10)
    ax.set_title("Precision-Recall Curve", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_calibration_curve_eval(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: Path,
    n_bins: int = 10,
) -> None:
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    brier = brier_score_loss(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(prob_pred, prob_true, "s-", color="#9467bd", label=f"Model (Brier = {brier:.4f})")
    ax.plot([0, 1], [0, 1], "--", color="#7f7f7f", label="Perfect Calibration")
    ax.set_xlabel("Mean Predicted Probability", fontsize=10)
    ax.set_ylabel("Fraction of Positives", fontsize=10)
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlim([-0.05, 1.05])
    ax.set_title("Probability Calibration Curve", fontsize=12, fontweight="bold")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_confusion_matrix_eval(
    cm_dict: Dict[str, int],
    output_path: Path,
    title: str = "Test Confusion Matrix",
) -> None:
    cm_array = np.array([
        [cm_dict["tn"], cm_dict["fp"]],
        [cm_dict["fn"], cm_dict["tp"]]
    ])

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm_array,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=ax,
        xticklabels=["Predicted Attend", "Predicted Miss"],
        yticklabels=["Actual Attend", "Actual Miss"],
    )
    ax.set_title(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
