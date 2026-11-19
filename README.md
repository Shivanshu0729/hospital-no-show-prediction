# Hospital Appointment No-Show Prediction System
### Algions AI Labs – AI & Machine Learning Individual Project Assessment

An end-to-end, production-structured machine learning application that predicts the probability that an outpatient will miss a scheduled hospital appointment at the time of booking. Built for clinical operations and administrative scheduling teams to proactively identify high-risk appointments and optimize reminder workflows without compromising patient care.

---

## 1. Project Overview & Business Problem
Missed outpatient appointments ("no-shows") cost healthcare systems billions of dollars annually, lead to idle physician capacity and diagnostic equipment, increase scheduling buffers, and delay care delivery for other waiting patients.

The **Hospital Appointment No-Show Prediction System** acts as a decision-support tool. By estimating no-show probabilities when an appointment is booked, hospital administrative teams can target proactive outreach (such as interactive SMS reminders, phone confirmation calls, or transportation check-ins) to high-risk appointments.

### Strict Governance & Clinical Boundaries
- **Advisory Support Only:** The system generates decision-support estimates. It never makes automated clinical or denial decisions.
- **No Diagnostic or Treatment Prediction:** Does not predict diseases, medical outcomes, or triage urgency.
- **Zero Sensitive Clinical Data:** Excludes diagnoses, notes, treatments, and protected health information.
- **No Care Deprioritization:** Predictions must never be used to penalize patients, refuse appointments, or cancel bookings.

---

## 2. System Architecture & Tech Stack

```text
hospital-no-show-prediction/
│
├── data/
│   ├── raw/
│   │   └── hospital_appointments.csv
│   └── processed/
│       └── hospital_appointments_cleaned.csv
│
├── models/
│   ├── best_model.joblib
│   └── model_metadata.json
│
├── notebooks/
│   └── exploration_and_demo.ipynb
│
├── reports/
│   ├── figures/
│   ├── model_comparison.csv
│   ├── test_metrics.json
│   ├── error_analysis.json
│   └── technical_report.md
│
├── src/
│   ├── data/
│   │   ├── generate_data.py
│   │   └── preprocessing.py
│   ├── analysis/
│   │   ├── eda.py
│   │   └── error_analysis.py
│   ├── models/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── explainability.py
│   ├── api/
│   │   ├── main.py
│   │   └── schemas.py
│   └── utils/
│       └── config.py
│
├── dashboard/
│   └── app.py
│
├── tests/
│   └── test_pipeline.py
│
├── scripts/
│   └── run_pipeline.py
│
├── .gitignore
├── .env.example
├── requirements.txt
├── README.md
└── LICENSE
```

### Technology Stack
- **Language & Runtime:** Python 3.11+
- **Data & Modeling:** Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn, Joblib
- **Backend API:** FastAPI, Uvicorn, Pydantic, HTTPX
- **Interactive UI:** Streamlit, Plotly
- **Quality Assurance:** Pytest

---

## 3. Dataset & Data Leakage Prevention

### Synthetic Dataset Characteristics
A fully reproducible dataset containing 6,000 appointment records generated with fixed seed `42` (`data/raw/hospital_appointments.csv`):
- `appointment_id`: Unique identifier (e.g., `APT-100001`).
- `booking_date`: Date the appointment was scheduled.
- `appointment_date`: Scheduled date of the appointment (`appointment_date >= booking_date`).
- `lead_time_days`: Difference in days between booking and appointment (0 to 60 days).
- `appointment_weekday`: Monday through Sunday.
- `appointment_month`: Month of appointment (January to December).
- `appointment_hour`: Consultation hour (8:00 to 17:00).
- `booking_channel`: `Online`, `Phone`, `In-Person`, `Referral`.
- `reminder_preference`: `SMS`, `Email`, `Phone Call`, `None`.
- `previous_appointments`: Count of prior clinic visits.
- `previous_no_shows`: Prior missed appointments (`previous_no_shows <= previous_appointments`).
- `appointment_type`: `General Consultation`, `Follow-up`, `Specialist`, `Diagnostic`, `Preventive Care`.
- `is_weekend`: 1 if Saturday or Sunday, else 0.
- `no_show`: Target variable (1 = Missed, 0 = Attended). Realistically imbalanced (~22-25%).

### Anti-Leakage Protocol
Predictions occur at the **exact moment an appointment is scheduled**. The pipeline strictly omits:
- Actual attendance outcomes.
- Post-booking reminder confirmations or delivery logs.
- Same-day cancellations or clinical check-in events.
- Medical diagnoses and treatment outcomes.

---

## 4. Machine Learning Pipeline & Validation Strategy

### Chronological Splitting
To simulate real-world prospective deployment and prevent temporal lookahead leakage, records are sorted chronologically by date and partitioned into:
- **Training Set (70%):** Past appointment horizon used to fit transformers and models.
- **Validation Set (15%):** Intermediate horizon used to evaluate candidate models, perform threshold sweeps, and assess probability calibration.
- **Held-Out Test Set (15%):** Prospective future horizon untouched until final evaluation.

### Candidate Models Evaluated
1. **Majority Dummy Baseline:** Zero-skill baseline predicting the majority class.
2. **Logistic Regression:** Linear decision boundary with balanced class weighting.
3. **Random Forest Classifier:** Ensembled non-linear decision trees with balanced subsampling.
4. **XGBoost Classifier:** Gradient-boosted decision trees optimizing class-weighted log-loss.

### Evaluation Metrics
- **Precision, Recall, F1-Score:** Selected for class-imbalanced evaluation (prioritizing recall to catch no-shows while maintaining precision to minimize outreach costs).
- **ROC-AUC:** Threshold-independent ranking capability.
- **Brier Score:** Mean squared difference between predicted probabilities and actual outcomes, quantifying probability calibration.
- **Confusion Matrix:** Breakdown of True Positives, False Positives, False Negatives, and True Negatives.

### Threshold & Calibration Analysis
- Continuous probabilities are converted to binary actions via an adjustable decision threshold (default: 0.50).
- Validation sweeps identify the optimal F1 threshold.
- Lower thresholds increase Recall (identifying more no-shows at the expense of higher reminder outreach costs).
- Higher thresholds increase Precision (saving outreach costs but allowing more missed appointments).

---

## 5. FastAPI Backend Service

The REST API exposes the trained inference pipeline independently of the frontend UI:
- `GET /`: API welcome and endpoint index.
- `GET /health`: Model artifact availability and system health.
- `GET /model-info`: Active model metadata, features, and evaluation metrics.
- `POST /predict`: Real-time probability prediction, risk tier (`Low`, `Moderate`, `High`), and local feature explainability for single appointments.
- `POST /predict/batch`: High-throughput batch inference for scheduling rosters.
- `GET /metrics`: Test set evaluation benchmarks.

Interactive Swagger documentation is available at `http://127.0.0.1:8000/docs`.

---

## 6. Streamlit Frontend Dashboard

The **Hospital Appointment Intelligence** web dashboard (`dashboard/app.py`) provides:
1. **Overview Dashboard:** High-level KPIs, attendance proportions, time trends, weekday distributions, and lead-time analysis.
2. **Predict Appointment:** Single-appointment prediction form with dynamic probability gauge, risk level badges, configurable threshold slider, and batch CSV upload and download.
3. **Model Performance:** Model benchmark tables, ROC curves, PR curves, calibration curves, confusion matrix, and threshold trade-off curve.
4. **Data Explorer:** Tabular inspection, multi-attribute filtering, search by appointment ID, and CSV dataset export.
5. **Model Explainability:** Global feature importance visualizations and empirical risk factor breakdown.
6. **About & Responsible AI:** Detailed documentation of operational context, ethical governance, and data leakage safeguards.

---

## 7. Setup & Execution Guide (Windows PowerShell)

### Step 1: Open Terminal and Navigate to Project
```powershell
cd "c:\Users\shiva\ML projects\hospital-no-show-prediction"
```

### Step 2: Create and Activate Virtual Environment
```powershell
python -m venv .venv
.venv\Scripts\activate
```

### Step 3: Install Required Dependencies
```powershell
pip install -r requirements.txt
```

### Step 4: Run the Complete Training Pipeline
Executes data generation, cleaning, EDA visualization generation, model training, validation benchmarking, champion selection, threshold tuning, test set evaluation, and technical report creation:
```powershell
python -m scripts.run_pipeline
```

### Step 5: Run the Pytest Test Suite
Executes the comprehensive 12-test validation suite:
```powershell
pytest -q
```

### Step 6: Launch the FastAPI Backend Server
In your primary terminal:
```powershell
uvicorn src.api.main:app --reload --port 8000
```
Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### Step 7: Launch the Streamlit Frontend Dashboard
In a second terminal:
```powershell
cd "c:\Users\shiva\ML projects\hospital-no-show-prediction"
.venv\Scripts\activate
streamlit run dashboard/app.py
```
Dashboard URL: [http://localhost:8501](http://localhost:8501)

---

## 8. Responsible AI & Operational Ethics
- **Decision Support Only:** Intended to guide supportive interventions (e.g., proactive reminder calls, SMS notifications, and scheduling buffers).
- **No Penalization:** High no-show probability must never lead to punitive cancellation or refusal of patient care.
- **Human-in-the-Loop:** Clinical coordinators review flagged appointments to ensure patient-centered communication.
- **Synthetic Data Disclaimer:** The dataset provided in this repository is synthetic and generated for demonstration and assessment purposes; it must not be cited as real-world patient outcomes.
