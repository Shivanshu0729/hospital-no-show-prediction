from typing import Dict, Any, List
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from src.utils.config import FIGURES_DIR


def extract_feature_names(pipeline: Pipeline) -> List[str]:
    preprocessor = pipeline.named_steps["preprocessor"]
    raw_names = preprocessor.get_feature_names_out()
    cleaned_names = []
    for name in raw_names:
        clean = name.replace("num__", "").replace("cat__", "").replace("bin__", "")
        cleaned_names.append(clean)
    return cleaned_names


def get_global_feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    feature_names = extract_feature_names(pipeline)
    classifier = pipeline.named_steps["classifier"]

    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        importances = np.abs(classifier.coef_[0])
    else:
        importances = np.ones(len(feature_names)) / len(feature_names)

    df_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    })
    df_imp = df_imp.sort_values(by="importance", ascending=False).reset_index(drop=True)
    return df_imp


def plot_global_feature_importance(
    importance_df: pd.DataFrame,
    output_path: Path,
    top_n: int = 12,
) -> None:
    top_df = importance_df.head(top_n).sort_values(by="importance", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(
        x="importance",
        y="feature",
        data=top_df,
        ax=ax,
        palette="viridis",
        hue="feature",
        legend=False,
    )
    ax.set_title("Global Feature Importance (Top Features)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Relative Importance Score", fontsize=10)
    ax.set_ylabel("Feature Name", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def explain_single_prediction(
    pipeline: Pipeline,
    input_data: pd.DataFrame,
    training_data_sample: pd.DataFrame,
) -> Dict[str, Any]:
    feature_adder = pipeline.named_steps["feature_adder"]
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    feature_names = extract_feature_names(pipeline)

    sample_engineered = feature_adder.transform(training_data_sample)
    input_engineered = feature_adder.transform(input_data)

    sample_trans = preprocessor.transform(sample_engineered)
    input_trans = preprocessor.transform(input_engineered)

    baseline_means = np.mean(sample_trans, axis=0)
    input_vector = input_trans[0]

    if hasattr(classifier, "feature_importances_"):
        weights = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        weights = classifier.coef_[0]
    else:
        weights = np.ones(len(feature_names)) / len(feature_names)

    diff = input_vector - baseline_means
    contributions = diff * weights

    contribution_records = []
    for name, cont in zip(feature_names, contributions):
        raw_val = None
        for col in input_data.columns:
            if col in name:
                raw_val = str(input_data[col].iloc[0])
                break

        contribution_records.append({
            "feature": name,
            "impact": float(round(cont, 4)),
            "direction": "Increases Risk" if cont > 0 else "Decreases Risk",
            "reference_value": raw_val,
        })

    sorted_records = sorted(
        contribution_records, key=lambda x: abs(x["impact"]), reverse=True
    )

    top_drivers = sorted_records[:5]

    explanation = {
        "top_drivers": top_drivers,
        "lead_time_days": int(input_data["lead_time_days"].iloc[0]),
        "previous_appointments": int(input_data["previous_appointments"].iloc[0]),
        "previous_no_shows": int(input_data["previous_no_shows"].iloc[0]),
        "reminder_preference": str(input_data["reminder_preference"].iloc[0]),
        "booking_channel": str(input_data["booking_channel"].iloc[0]),
    }
    return explanation
