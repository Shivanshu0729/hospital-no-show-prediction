from typing import Dict, Any
from pathlib import Path
import json
import numpy as np
import pandas as pd
from src.utils.config import ERROR_ANALYSIS_PATH


def analyze_prediction_errors(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    test_df: pd.DataFrame,
    threshold: float = 0.50,
    cost_false_positive: float = 5.0,
    cost_false_negative: float = 65.0,
    output_path: Path = ERROR_ANALYSIS_PATH,
) -> Dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)

    is_tp = (y_true == 1) & (y_pred == 1)
    is_tn = (y_true == 0) & (y_pred == 0)
    is_fp = (y_true == 0) & (y_pred == 1)
    is_fn = (y_true == 1) & (y_pred == 0)

    tp_count = int(np.sum(is_tp))
    tn_count = int(np.sum(is_tn))
    fp_count = int(np.sum(is_fp))
    fn_count = int(np.sum(is_fn))
    total_count = len(y_true)

    fp_cost_total = fp_count * cost_false_positive
    fn_cost_total = fn_count * cost_false_negative
    total_error_cost = fp_cost_total + fn_cost_total

    fp_df = test_df[is_fp]
    fn_df = test_df[is_fn]

    fp_mean_lead_time = float(round(fp_df["lead_time_days"].mean(), 2)) if len(fp_df) > 0 else 0.0
    fn_mean_lead_time = float(round(fn_df["lead_time_days"].mean(), 2)) if len(fn_df) > 0 else 0.0

    analysis_results = {
        "threshold": float(round(threshold, 4)),
        "total_test_samples": total_count,
        "counts": {
            "true_positives": tp_count,
            "true_negatives": tn_count,
            "false_positives": fp_count,
            "false_negatives": fn_count,
        },
        "rates": {
            "false_positive_rate": float(round(fp_count / max(1, (tn_count + fp_count)), 4)),
            "false_negative_rate": float(round(fn_count / max(1, (tp_count + fn_count)), 4)),
        },
        "economic_analysis": {
            "unit_cost_false_positive_usd": cost_false_positive,
            "unit_cost_false_negative_usd": cost_false_negative,
            "total_false_positive_cost_usd": float(round(fp_cost_total, 2)),
            "total_false_negative_cost_usd": float(round(fn_cost_total, 2)),
            "total_estimated_error_cost_usd": float(round(total_error_cost, 2)),
        },
        "characteristics": {
            "false_positives_mean_lead_time": fp_mean_lead_time,
            "false_negatives_mean_lead_time": fn_mean_lead_time,
        },
        "operational_consequences": {
            "false_positive_impact": "Unnecessary proactive reminders, administrative outreach overhead, mild patient reminder fatigue.",
            "false_negative_impact": "Unfilled clinical slots, lost specialist productivity, delayed patient care, unrecovered hospital overhead.",
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(analysis_results, f, indent=2)

    return analysis_results
