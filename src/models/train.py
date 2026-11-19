from typing import Dict, Any, Tuple
from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.utils.config import (
    PROCESSED_DATA_PATH,
    MODEL_PATH,
    METADATA_PATH,
    COMPARISON_METRICS_PATH,
    TEST_METRICS_PATH,
    FIGURES_DIR,
    RANDOM_SEED,
    DEFAULT_THRESHOLD,
    ALL_FEATURE_COLUMNS,
    TARGET_COLUMN,
)
from src.data.preprocessing import (
    create_full_pipeline,
    chronological_split,
    extract_features_and_target,
)
from src.models.evaluate import (
    compute_metrics,
    find_optimal_threshold,
    sweep_thresholds,
    plot_roc_curve_eval,
    plot_precision_recall_curve_eval,
    plot_calibration_curve_eval,
    plot_confusion_matrix_eval,
)
from src.models.explainability import (
    get_global_feature_importance,
    plot_global_feature_importance,
)
from src.analysis.error_analysis import analyze_prediction_errors


def train_and_evaluate_models(
    df: pd.DataFrame,
) -> Dict[str, Any]:
    train_df, val_df, test_df = chronological_split(df, train_ratio=0.70, val_ratio=0.15)

    X_train, y_train = extract_features_and_target(train_df)
    X_val, y_val = extract_features_and_target(val_df)
    X_test, y_test = extract_features_and_target(test_df)

    y_train_arr = y_train.to_numpy()
    y_val_arr = y_val.to_numpy()
    y_test_arr = y_test.to_numpy()

    neg_count = int(np.sum(y_train_arr == 0))
    pos_count = int(np.sum(y_train_arr == 1))
    scale_pos_weight = neg_count / max(1, pos_count)

    candidate_classifiers = {
        "Baseline (Dummy Majority)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_SEED,
        ),
        "Random Forest": RandomForestClassifier(
            class_weight="balanced",
            n_estimators=100,
            max_depth=8,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=RANDOM_SEED,
            n_estimators=100,
            max_depth=5,
            learning_rate=0.08,
        ),
    }

    trained_pipelines = {}
    val_metrics_records = []
    val_probs_dict = {}

    for name, clf in candidate_classifiers.items():
        pipeline = create_full_pipeline(clf)
        pipeline.fit(X_train, y_train)
        trained_pipelines[name] = pipeline

        if hasattr(pipeline, "predict_proba"):
            val_probs = pipeline.predict_proba(X_val)[:, 1]
        else:
            val_preds = pipeline.predict(X_val)
            val_probs = val_preds.astype(float)

        val_probs_dict[name] = val_probs
        metrics = compute_metrics(y_val_arr, val_probs, threshold=DEFAULT_THRESHOLD)
        metrics["model_name"] = name
        val_metrics_records.append(metrics)

    comparison_df = pd.DataFrame(val_metrics_records)
    cols_order = [
        "model_name",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "brier_score",
        "accuracy",
        "threshold",
    ]
    comparison_df = comparison_df[cols_order].sort_values(by="f1", ascending=False)
    COMPARISON_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(COMPARISON_METRICS_PATH, index=False)

    best_candidate_row = comparison_df.iloc[0]
    best_model_name = str(best_candidate_row["model_name"])
    best_pipeline = trained_pipelines[best_model_name]
    best_val_probs = val_probs_dict[best_model_name]

    optimal_threshold = find_optimal_threshold(y_val_arr, best_val_probs)

    test_probs = best_pipeline.predict_proba(X_test)[:, 1]

    test_metrics_default = compute_metrics(y_test_arr, test_probs, threshold=DEFAULT_THRESHOLD)
    test_metrics_optimal = compute_metrics(y_test_arr, test_probs, threshold=optimal_threshold)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_roc_curve_eval(y_test_arr, test_probs, FIGURES_DIR / "roc_curve.png", model_name=best_model_name)
    plot_precision_recall_curve_eval(y_test_arr, test_probs, FIGURES_DIR / "precision_recall_curve.png", model_name=best_model_name)
    plot_calibration_curve_eval(y_test_arr, test_probs, FIGURES_DIR / "calibration_curve.png")
    plot_confusion_matrix_eval(test_metrics_default["confusion_matrix"], FIGURES_DIR / "confusion_matrix_test.png")

    importance_df = get_global_feature_importance(best_pipeline)
    plot_global_feature_importance(importance_df, FIGURES_DIR / "feature_importance.png")
    importance_df.to_csv(FIGURES_DIR.parent / "feature_importance.csv", index=False)

    error_analysis_results = analyze_prediction_errors(
        y_true=y_test_arr,
        y_prob=test_probs,
        test_df=test_df,
        threshold=DEFAULT_THRESHOLD,
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, MODEL_PATH)

    val_threshold_sweep = sweep_thresholds(y_val_arr, best_val_probs)

    final_test_summary = {
        "selected_model": best_model_name,
        "default_threshold": DEFAULT_THRESHOLD,
        "optimal_threshold": float(round(optimal_threshold, 4)),
        "metrics_at_default_threshold": test_metrics_default,
        "metrics_at_optimal_threshold": test_metrics_optimal,
        "validation_metrics": best_candidate_row.to_dict(),
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "test_positive_prevalence": float(round(np.mean(y_test_arr), 4)),
    }

    with open(TEST_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(final_test_summary, f, indent=2)

    model_metadata = {
        "model_name": best_model_name,
        "features": ALL_FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "training_samples": len(train_df),
        "validation_samples": len(val_df),
        "test_samples": len(test_df),
        "default_threshold": DEFAULT_THRESHOLD,
        "optimal_threshold": float(round(optimal_threshold, 4)),
        "test_metrics_default": test_metrics_default,
        "test_metrics_optimal": test_metrics_optimal,
        "top_features": importance_df.head(10).to_dict(orient="records"),
        "error_summary": {
            "false_positives": error_analysis_results["counts"]["false_positives"],
            "false_negatives": error_analysis_results["counts"]["false_negatives"],
            "total_estimated_error_cost_usd": error_analysis_results["economic_analysis"]["total_estimated_error_cost_usd"],
        },
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(model_metadata, f, indent=2)

    return {
        "best_pipeline": best_pipeline,
        "best_model_name": best_model_name,
        "test_metrics": final_test_summary,
        "comparison_df": comparison_df,
        "threshold_sweep_df": val_threshold_sweep,
    }


if __name__ == "__main__":
    if PROCESSED_DATA_PATH.exists():
        data = pd.read_csv(PROCESSED_DATA_PATH)
        train_and_evaluate_models(data)
