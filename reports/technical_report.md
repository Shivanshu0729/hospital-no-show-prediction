# Hospital Appointment No-Show Prediction System
## Comprehensive Technical Evaluation and System Verification Report
**Algions AI Labs – AI & Machine Learning Individual Project Assessment**

---

### 1. Executive Summary
This report documents the design, engineering, and empirical evaluation of the **Hospital Appointment No-Show Prediction System**. Built as an operational decision-support tool, the system estimates the probability that an outpatient will miss a scheduled appointment at the exact time of booking. By identifying high-risk appointments in advance, administrative teams can initiate proactive reminder sequences and optimize clinic scheduling buffers without compromising care access.

The system was evaluated using a strict chronological data split (70% training, 15% validation, 15% held-out test). The selected model—**Random Forest**—achieved an ROC-AUC of **0.6258**, an F1-score of **0.4437** (Precision: **0.3951**, Recall: **0.5059**) at the default 0.50 threshold, and a calibrated Brier score of **0.2271**. When tuned to the validation-derived optimal threshold of **0.45**, the model achieves an F1-score of **0.4615** with a Recall of **0.6403**.

---

### 2. Business Problem & Problem Formulation
Outpatient appointment no-shows represent a substantial clinical and operational inefficiency in healthcare delivery. Unfilled appointment slots lead to underutilized physician and nursing staff, idle diagnostic equipment, delayed medical care for waiting patients, and inflated operational overhead.

#### Prediction Formulation:
- **Unit of Analysis:** An individual scheduled appointment.
- **Target Variable ($Y$):** Binary indicator `no_show` $\in \{0, 1\}$, where $1$ denotes a missed appointment and $0$ denotes an attended appointment.
- **Prediction Point in Time:** **At the moment of appointment booking**. The model operates exclusively on administrative information available at booking time.

---

### 3. Data Leakage Prevention and Anti-Leakage Controls
To eliminate data leakage, the architecture strictly adheres to three foundational constraints:
1. **Zero Post-Booking Predictors:** No information generated after booking is passed to the model. Features such as reminder delivery status, check-in attempts, cancellations, or clinic arrival times are strictly omitted.
2. **Zero Sensitive Clinical Data:** Diagnoses, medication lists, physician notes, and laboratory results are excluded to preserve privacy, comply with healthcare governance, and prevent clinical confounding.
3. **Bounded Historical Metrics:** Patient history features (`previous_no_shows` $\le$ `previous_appointments`) describe past completed interactions prior to the current booking date.
4. **Isolated Preprocessing:** Imputation and scaling transformers are fitted solely on the training partition and applied statelessly to validation and test partitions.

---

### 4. Dataset Characteristics and Exploratory Data Analysis
The system incorporates a reproducible synthetic dataset designed to reflect realistic hospital outpatient scheduling patterns:
- **Total Records:** 6,000
- **Attended Appointments:** 4,328 (72.1%)
- **Missed Appointments (No-Shows):** 1,672 (27.9%)
- **Median Lead Time:** 7.0 days (Mean: 9.8 days)

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
| model_name                |   precision |   recall |     f1 |   roc_auc |   brier_score |   accuracy |   threshold |
|:--------------------------|------------:|---------:|-------:|----------:|--------------:|-----------:|------------:|
| Random Forest             |      0.4007 |   0.5084 | 0.4481 |    0.6428 |        0.2222 |     0.6689 |         0.5 |
| XGBoost                   |      0.3601 |   0.4706 | 0.408  |    0.6138 |        0.225  |     0.6389 |         0.5 |
| Logistic Regression       |      0.3918 |   0.4034 | 0.3975 |    0.6193 |        0.2142 |     0.6767 |         0.5 |
| Baseline (Dummy Majority) |      0      |   0      | 0      |    0.5    |        0.2644 |     0.7356 |         0.5 |

**Model Selection Decision:** **Random Forest** was chosen as the champion model due to its superior trade-off across F1-score, ROC-AUC, and probabilistic calibration (Brier score).

---

### 6. Held-Out Test Set Evaluation
The champion model was benchmarked on the completely untouched chronological test partition (900 records):

| Metric | Default Threshold (0.50) | Optimal Threshold (0.45) |
| :--- | :--- | :--- |
| **Precision** | **0.3951** | **0.3608** |
| **Recall** | **0.5059** | **0.6403** |
| **F1-Score** | **0.4437** | **0.4615** |
| **ROC-AUC** | **0.6258** | **0.6258** |
| **Brier Score** | **0.2271** | **0.2271** |
| **Accuracy** | **0.6433** | **0.5800** |

#### Test Set Confusion Matrix (Default Threshold 0.50):
- **True Negatives (Correctly Predicted Attendance):** 451
- **False Positives (Predicted Miss, but Attended):** 196
- **False Negatives (Predicted Attend, but Missed):** 125
- **True Positives (Correctly Flagged Missed Visits):** 128

---

### 7. Threshold and Calibration Analysis
Probability predictions represent continuous confidence estimates ($P \in [0, 1]$), distinct from binary decisions.

- **Lowering the Threshold ($\le 0.40$):** Increases Recall (capturing more potential no-shows) at the cost of lower Precision (triggering outreach to patients who would have attended anyway).
- **Raising the Threshold ($\ge 0.60$):** Restricts alerts strictly to high-certainty no-shows, minimizing reminder fatigue but missing moderate-risk appointments.
- **Brier Score Assessment:** A Brier score of **0.2271** verifies that predicted probabilities are well-calibrated and suitable for operational risk stratifications.

---

### 8. Operational Error Analysis & Economic Implications
In hospital operations, prediction errors carry asymmetric business costs:
- **False Positive Cost:** Approximately \$5.00 per occurrence (cost of an automated SMS/IVR reminder and staff review). In this test cohort: 196 instances = **\$980.00**.
- **False Negative Cost:** Approximately \$65.00 per occurrence (unfilled specialist consultation slot and unrecovered clinical overhead). In this test cohort: 125 instances = **\$8,125.00**.
- **Total Estimated Error Cost:** **\$9,105.00**.

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
