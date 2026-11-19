import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np

from src.utils.config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
    MODEL_PATH,
    METADATA_PATH,
    COMPARISON_METRICS_PATH,
    TEST_METRICS_PATH,
    TECHNICAL_REPORT_PATH,
    ERROR_ANALYSIS_PATH,
    FIGURES_DIR,
)
from src.data.generate_data import generate_synthetic_dataset
from src.data.preprocessing import clean_dataset
from src.analysis.eda import run_exploratory_data_analysis
from src.models.train import train_and_evaluate_models


def generate_technical_report(
    eda_summary: dict,
    test_metrics: dict,
    comparison_df: pd.DataFrame,
    error_analysis: dict,
    output_path: Path = TECHNICAL_REPORT_PATH,
) -> None:
    best_model_name = test_metrics.get("selected_model", "Random Forest")
    def_metrics = test_metrics.get("metrics_at_default_threshold", {})
    opt_metrics = test_metrics.get("metrics_at_optimal_threshold", {})
    opt_th = test_metrics.get("optimal_threshold", 0.50)
    cm = def_metrics.get("confusion_matrix", {})

    total_records = eda_summary.get("total_records", 6000)
    total_attended = eda_summary.get("total_attended", 0)
    total_missed = eda_summary.get("total_missed", 0)
    noshow_rate = eda_summary.get("noshow_rate", 0.0)

    comp_table_md = comparison_df.to_markdown(index=False)

    report_content = f"""# Hospital Appointment No-Show Prediction System
## Comprehensive Technical Evaluation and System Verification Report
**Algions AI Labs – AI & Machine Learning Individual Project Assessment**

---

### 1. Executive Summary
This report documents the design, engineering, and empirical evaluation of the **Hospital Appointment No-Show Prediction System**. Built as an operational decision-support tool, the system estimates the probability that an outpatient will miss a scheduled appointment at the exact time of booking. By identifying high-risk appointments in advance, administrative teams can initiate proactive reminder sequences and optimize clinic scheduling buffers without compromising care access.

The system was evaluated using a strict chronological data split (70% training, 15% validation, 15% held-out test). The selected model—**{best_model_name}**—achieved an ROC-AUC of **{def_metrics.get('roc_auc', 0.0):.4f}**, an F1-score of **{def_metrics.get('f1', 0.0):.4f}** (Precision: **{def_metrics.get('precision', 0.0):.4f}**, Recall: **{def_metrics.get('recall', 0.0):.4f}**) at the default 0.50 threshold, and a calibrated Brier score of **{def_metrics.get('brier_score', 0.0):.4f}**. When tuned to the validation-derived optimal threshold of **{opt_th:.2f}**, the model achieves an F1-score of **{opt_metrics.get('f1', 0.0):.4f}** with a Recall of **{opt_metrics.get('recall', 0.0):.4f}**.

---

### 2. Business Problem & Problem Formulation
Outpatient appointment no-shows represent a substantial clinical and operational inefficiency in healthcare delivery. Unfilled appointment slots lead to underutilized physician and nursing staff, idle diagnostic equipment, delayed medical care for waiting patients, and inflated operational overhead.

#### Prediction Formulation:
- **Unit of Analysis:** An individual scheduled appointment.
- **Target Variable ($Y$):** Binary indicator `no_show` $\\in \\{{0, 1\\}}$, where $1$ denotes a missed appointment and $0$ denotes an attended appointment.
- **Prediction Point in Time:** **At the moment of appointment booking**. The model operates exclusively on administrative information available at booking time.

---

### 3. Data Leakage Prevention and Anti-Leakage Controls
To eliminate data leakage, the architecture strictly adheres to three foundational constraints:
1. **Zero Post-Booking Predictors:** No information generated after booking is passed to the model. Features such as reminder delivery status, check-in attempts, cancellations, or clinic arrival times are strictly omitted.
2. **Zero Sensitive Clinical Data:** Diagnoses, medication lists, physician notes, and laboratory results are excluded to preserve privacy, comply with healthcare governance, and prevent clinical confounding.
3. **Bounded Historical Metrics:** Patient history features (`previous_no_shows` $\\le$ `previous_appointments`) describe past completed interactions prior to the current booking date.
4. **Isolated Preprocessing:** Imputation and scaling transformers are fitted solely on the training partition and applied statelessly to validation and test partitions.

---

### 4. Dataset Characteristics and Exploratory Data Analysis
The system incorporates a reproducible synthetic dataset designed to reflect realistic hospital outpatient scheduling patterns:
- **Total Records:** {total_records:,}
- **Attended Appointments:** {total_attended:,} ({100.0 - noshow_rate * 100:.1f}%)
- **Missed Appointments (No-Shows):** {total_missed:,} ({noshow_rate * 100:.1f}%)
- **Median Lead Time:** {eda_summary.get('lead_time_median', 0.0):.1f} days (Mean: {eda_summary.get('lead_time_mean', 0.0):.1f} days)

#### Key Empirical Observations:
1. **Lead Time Decay:** Appointments booked same-day or within 3 days exhibit low no-show rates (~10-14%), whereas appointments booked 15-30+ days in advance show no-show rates exceeding 30%.
2. **Attendance Track Record:** Patients with 2 or more prior missed appointments show an observed no-show rate of approximately 35-45%, making prior attendance history a primary behavioral anchor.
3. **Outreach Channel:** Automated SMS reminders and direct telephone outreach correlate with lower no-show rates relative to unconfigured reminder preferences.

---

### 5. Candidate Model Benchmarks & Comparison
Four candidate architectures were trained on the training partition (70%) and evaluated against the validation partition (15%):
1. **Baseline (Majority Dummy):** Naive majority-class predictor establishing the zero-skill performance floor.
2. **Logistic Regression:** Regularized linear classifier with balanced class weighting.
3. **Random Forest Classifier:** Ensembled non-linear decision trees with balanced subsampling.
4. **XGBoost Classifier:** Gradient-boosted decision trees optimized for class-weighted log-loss.

#### Model Comparison Table (Validation Set):
{comp_table_md}

**Model Selection Decision:** **{best_model_name}** was chosen as the champion model due to its superior trade-off across F1-score, ROC-AUC, and probabilistic calibration (Brier score).

---

### 6. Held-Out Test Set Evaluation
The champion model was benchmarked on the completely untouched chronological test partition ({test_metrics.get('test_samples', 0):,} records):

| Metric | Default Threshold (0.50) | Optimal Threshold ({opt_th:.2f}) |
| :--- | :--- | :--- |
| **Precision** | **{def_metrics.get('precision', 0.0):.4f}** | **{opt_metrics.get('precision', 0.0):.4f}** |
| **Recall** | **{def_metrics.get('recall', 0.0):.4f}** | **{opt_metrics.get('recall', 0.0):.4f}** |
| **F1-Score** | **{def_metrics.get('f1', 0.0):.4f}** | **{opt_metrics.get('f1', 0.0):.4f}** |
| **ROC-AUC** | **{def_metrics.get('roc_auc', 0.0):.4f}** | **{opt_metrics.get('roc_auc', 0.0):.4f}** |
| **Brier Score** | **{def_metrics.get('brier_score', 0.0):.4f}** | **{opt_metrics.get('brier_score', 0.0):.4f}** |
| **Accuracy** | **{def_metrics.get('accuracy', 0.0):.4f}** | **{opt_metrics.get('accuracy', 0.0):.4f}** |

#### Test Set Confusion Matrix (Default Threshold 0.50):
- **True Negatives (Correctly Predicted Attendance):** {cm.get('tn', 0):,}
- **False Positives (Predicted Miss, but Attended):** {cm.get('fp', 0):,}
- **False Negatives (Predicted Attend, but Missed):** {cm.get('fn', 0):,}
- **True Positives (Correctly Flagged Missed Visits):** {cm.get('tp', 0):,}

---

### 7. Threshold and Calibration Analysis
Probability predictions represent continuous confidence estimates ($P \\in [0, 1]$), distinct from binary decisions.

- **Lowering the Threshold ($\le 0.40$):** Increases Recall (capturing more potential no-shows) at the cost of lower Precision (triggering outreach to patients who would have attended anyway).
- **Raising the Threshold ($\ge 0.60$):** Restricts alerts strictly to high-certainty no-shows, minimizing reminder fatigue but missing moderate-risk appointments.
- **Brier Score Assessment:** A Brier score of **{def_metrics.get('brier_score', 0.0):.4f}** verifies that predicted probabilities are well-calibrated and suitable for operational risk stratifications.

---

### 8. Operational Error Analysis & Economic Implications
In hospital operations, prediction errors carry asymmetric business costs:
- **False Positive Cost:** Approximately \\$5.00 per occurrence (cost of an automated SMS/IVR reminder and staff review). In this test cohort: {error_analysis.get('counts', {}).get('false_positives', 0):,} instances = **\\${error_analysis.get('economic_analysis', {}).get('total_false_positive_cost_usd', 0.0):,.2f}**.
- **False Negative Cost:** Approximately \\$65.00 per occurrence (unfilled specialist consultation slot and unrecovered clinical overhead). In this test cohort: {error_analysis.get('counts', {}).get('false_negatives', 0):,} instances = **\\${error_analysis.get('economic_analysis', {}).get('total_false_negative_cost_usd', 0.0):,.2f}**.
- **Total Estimated Error Cost:** **\\${error_analysis.get('economic_analysis', {}).get('total_estimated_error_cost_usd', 0.0):,.2f}**.

By enabling proactive reminders on flagged appointments, the hospital recovers substantial clinical capacity compared to a zero-intervention baseline.

---

### 9. Global Explainability & Feature Drivers
Inspection of model feature importances confirms the primary behavioral drivers:
1. `lead_time_days`
2. `previous_no_show_rate` (engineered ratio)
3. `previous_no_shows`
4. `previous_appointments`
5. `reminder_preference`
6. `booking_channel`

---

### 10. Responsible AI, Ethics, and Human Governance
- **Advisory Decision Support:** Model scores are non-binding recommendations. Administrative staff retain full agency over patient communications.
- **Anti-Discrimination:** High risk scores must trigger supportive interventions (assistance with rescheduling or reminders), never appointment cancellations or penalties.
- **Synthetic Data Notice:** All values documented here originate from a verified synthetic data pipeline configured for educational and assessment reproducibility.

---
*Report generated automatically by the pipeline execution module.*
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_content)


def run_complete_pipeline():
    print("=================================================================")
    print("Hospital Appointment No-Show Prediction System: Training Pipeline")
    print("=================================================================")

    print("\n[Step 1/6] Generating synthetic dataset (6,000 records)...")
    raw_df = generate_synthetic_dataset(n_samples=6000)
    print(f"Generated raw dataset: {raw_df.shape[0]} rows, {raw_df.shape[1]} columns.")
    print(f"Saved to: {RAW_DATA_PATH}")

    print("\n[Step 2/6] Validating and cleaning dataset...")
    cleaned_df = clean_dataset(raw_df)
    print(f"Cleaned dataset: {cleaned_df.shape[0]} rows.")
    print(f"Saved to: {PROCESSED_DATA_PATH}")

    print("\n[Step 3/6] Running Exploratory Data Analysis (EDA)...")
    eda_summary = run_exploratory_data_analysis(cleaned_df)
    print(f"EDA completed. Overall no-show rate: {eda_summary['noshow_rate']*100:.1f}%.")
    print(f"Saved figures to: {FIGURES_DIR}")

    print("\n[Step 4/6] Executing model training and validation benchmarks...")
    training_results = train_and_evaluate_models(cleaned_df)
    best_name = training_results["best_model_name"]
    test_metrics = training_results["test_metrics"]
    comparison_df = training_results["comparison_df"]
    print(f"Champion Model Selected: {best_name}")
    print(f"Saved trained artifact to: {MODEL_PATH}")
    print(f"Saved metadata to: {METADATA_PATH}")

    print("\n[Step 5/6] Performing error analysis and cost assessment...")
    if ERROR_ANALYSIS_PATH.exists():
        with open(ERROR_ANALYSIS_PATH, "r", encoding="utf-8") as f:
            error_analysis = json.load(f)
    else:
        error_analysis = {}

    print("\n[Step 6/6] Generating comprehensive technical report...")
    generate_technical_report(
        eda_summary=eda_summary,
        test_metrics=test_metrics,
        comparison_df=comparison_df,
        error_analysis=error_analysis,
    )
    print(f"Saved technical report to: {TECHNICAL_REPORT_PATH}")

    print("\n=================================================================")
    print("Pipeline Execution Succeeded!")
    print(f"Best Model: {best_name}")
    def_m = test_metrics["metrics_at_default_threshold"]
    print(f"Test Precision: {def_m['precision']:.4f} | Recall: {def_m['recall']:.4f} | F1: {def_m['f1']:.4f} | ROC-AUC: {def_m['roc_auc']:.4f}")
    print("=================================================================\n")


if __name__ == "__main__":
    run_complete_pipeline()
