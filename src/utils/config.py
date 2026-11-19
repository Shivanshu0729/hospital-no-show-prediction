from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RAW_DATA_PATH = DATA_RAW_DIR / "hospital_appointments.csv"
PROCESSED_DATA_PATH = DATA_PROCESSED_DIR / "hospital_appointments_cleaned.csv"
MODEL_PATH = MODELS_DIR / "best_model.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
COMPARISON_METRICS_PATH = REPORTS_DIR / "model_comparison.csv"
TEST_METRICS_PATH = REPORTS_DIR / "test_metrics.json"
TECHNICAL_REPORT_PATH = REPORTS_DIR / "technical_report.md"
ERROR_ANALYSIS_PATH = REPORTS_DIR / "error_analysis.json"

RANDOM_SEED = 42
DEFAULT_THRESHOLD = 0.50
RISK_THRESHOLD_LOW = 0.35
RISK_THRESHOLD_HIGH = 0.65

NUMERICAL_FEATURES = [
    "lead_time_days",
    "appointment_hour",
    "previous_appointments",
    "previous_no_shows",
]

CATEGORICAL_FEATURES = [
    "appointment_weekday",
    "appointment_month",
    "booking_channel",
    "reminder_preference",
    "appointment_type",
]

BINARY_FEATURES = [
    "is_weekend",
]

ALL_FEATURE_COLUMNS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES
TARGET_COLUMN = "no_show"
ID_COLUMNS = ["appointment_id", "booking_date", "appointment_date"]

BOOKING_CHANNELS = ["Online", "Phone", "In-Person", "Referral"]
REMINDER_PREFERENCES = ["SMS", "Email", "Phone Call", "None"]
APPOINTMENT_TYPES = [
    "General Consultation",
    "Follow-up",
    "Specialist",
    "Diagnostic",
    "Preventive Care",
]
WEEKDAYS = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]
MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
