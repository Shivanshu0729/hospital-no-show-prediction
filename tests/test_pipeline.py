import pytest
import pandas as pd
import numpy as np
import joblib
from fastapi.testclient import TestClient

from src.utils.config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
    MODEL_PATH,
    ALL_FEATURE_COLUMNS,
    TARGET_COLUMN,
    ID_COLUMNS,
)
from src.data.generate_data import generate_synthetic_dataset
from src.data.preprocessing import (
    clean_dataset,
    create_full_pipeline,
    extract_features_and_target,
)
from src.api.main import app

client = TestClient(app)


def test_dataset_generation(tmp_path):
    test_csv = tmp_path / "test_raw.csv"
    df = generate_synthetic_dataset(n_samples=5000, random_seed=42, output_path=test_csv)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5000
    assert test_csv.exists()


def test_dataset_schema():
    expected_columns = ID_COLUMNS + ALL_FEATURE_COLUMNS + [TARGET_COLUMN]
    df = generate_synthetic_dataset(n_samples=50, random_seed=42, output_path=RAW_DATA_PATH)
    for col in expected_columns:
        assert col in df.columns


def test_preprocessing(tmp_path):
    raw_df = pd.DataFrame({
        "appointment_id": ["A1", "A1", "A2", "A3"],
        "booking_date": ["2024-01-01", "2024-01-01", "2024-05-10", "2024-02-01"],
        "appointment_date": ["2024-01-05", "2024-01-05", "2024-05-01", "2024-02-10"],
        "lead_time_days": [4, 4, -9, 9],
        "appointment_weekday": ["Friday", "Friday", "Wednesday", "Saturday"],
        "appointment_month": ["January", "January", "May", "February"],
        "appointment_hour": [10, 10, 11, 14],
        "booking_channel": ["Online", "Online", "Phone", np.nan],
        "reminder_preference": ["SMS", "SMS", np.nan, "Email"],
        "previous_appointments": [3, 3, 5, np.nan],
        "previous_no_shows": [1, 1, 6, 0],
        "appointment_type": ["Consultation", "Consultation", "Specialist", "Follow-up"],
        "is_weekend": [0, 0, 0, 1],
        "no_show": [0, 0, 1, 0],
    })
    cleaned = clean_dataset(raw_df)
    assert len(cleaned) <= 2
    assert "A1" in cleaned["appointment_id"].values
    assert cleaned["lead_time_days"].min() >= 0


def test_feature_validation():
    df = generate_synthetic_dataset(n_samples=100, random_seed=42, output_path=RAW_DATA_PATH)
    cleaned = clean_dataset(df)
    assert (cleaned["previous_no_shows"] <= cleaned["previous_appointments"]).all()
    assert (cleaned["lead_time_days"] >= 0).all()
    assert set(cleaned["no_show"].unique()).issubset({0, 1})


def test_model_artifact_loading():
    if not MODEL_PATH.exists():
        pytest.skip("Model artifact not yet created. Run pipeline first.")
    model = joblib.load(MODEL_PATH)
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")


def test_prediction_output():
    if not MODEL_PATH.exists():
        pytest.skip("Model artifact not yet created. Run pipeline first.")
    model = joblib.load(MODEL_PATH)
    sample_input = pd.DataFrame([{
        "lead_time_days": 10,
        "appointment_weekday": "Monday",
        "appointment_month": "October",
        "appointment_hour": 10,
        "booking_channel": "Online",
        "reminder_preference": "SMS",
        "previous_appointments": 3,
        "previous_no_shows": 1,
        "appointment_type": "General Consultation",
        "is_weekend": 0,
    }])
    preds = model.predict(sample_input)
    assert preds.shape == (1,)
    assert preds[0] in [0, 1]


def test_probability_range():
    if not MODEL_PATH.exists():
        pytest.skip("Model artifact not yet created. Run pipeline first.")
    model = joblib.load(MODEL_PATH)
    sample_input = pd.DataFrame([{
        "lead_time_days": 25,
        "appointment_weekday": "Friday",
        "appointment_month": "November",
        "appointment_hour": 16,
        "booking_channel": "Phone",
        "reminder_preference": "None",
        "previous_appointments": 5,
        "previous_no_shows": 3,
        "appointment_type": "Specialist",
        "is_weekend": 0,
    }])
    probs = model.predict_proba(sample_input)
    assert probs.shape == (1, 2)
    p0, p1 = probs[0]
    assert 0.0 <= p0 <= 1.0
    assert 0.0 <= p1 <= 1.0
    assert np.isclose(p0 + p1, 1.0, atol=1e-5)


def test_api_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data


def test_api_prediction_endpoint():
    payload = {
        "lead_time_days": 12,
        "appointment_weekday": "Tuesday",
        "appointment_month": "April",
        "appointment_hour": 11,
        "booking_channel": "Online",
        "reminder_preference": "SMS",
        "previous_appointments": 4,
        "previous_no_shows": 1,
        "appointment_type": "Follow-up",
        "is_weekend": 0,
        "threshold": 0.50,
    }
    response = client.post("/predict", json=payload)
    if response.status_code == 503:
        pytest.skip("Model artifact not loaded into API.")
    assert response.status_code == 200
    data = response.json()
    assert "no_show_probability" in data
    assert "predicted_no_show" in data
    assert "risk_level" in data
    assert data["risk_level"] in ["Low", "Moderate", "High"]
    assert 0.0 <= data["no_show_probability"] <= 1.0


def test_batch_prediction():
    payload = {
        "appointments": [
            {
                "lead_time_days": 2,
                "appointment_weekday": "Monday",
                "appointment_month": "May",
                "appointment_hour": 9,
                "booking_channel": "Online",
                "reminder_preference": "SMS",
                "previous_appointments": 1,
                "previous_no_shows": 0,
                "appointment_type": "General Consultation",
                "is_weekend": 0,
            },
            {
                "lead_time_days": 30,
                "appointment_weekday": "Saturday",
                "appointment_month": "December",
                "appointment_hour": 16,
                "booking_channel": "Phone",
                "reminder_preference": "None",
                "previous_appointments": 4,
                "previous_no_shows": 3,
                "appointment_type": "Specialist",
                "is_weekend": 1,
            },
        ],
        "threshold": 0.50,
    }
    response = client.post("/predict/batch", json=payload)
    if response.status_code == 503:
        pytest.skip("Model artifact not loaded into API.")
    assert response.status_code == 200
    data = response.json()
    assert data["total_processed"] == 2
    assert len(data["predictions"]) == 2


def test_invalid_input():
    invalid_payload = {
        "lead_time_days": 10,
        "appointment_weekday": "Monday",
        "appointment_month": "October",
        "appointment_hour": 10,
        "booking_channel": "Online",
        "reminder_preference": "SMS",
        "previous_appointments": 2,
        "previous_no_shows": 5,
        "appointment_type": "General Consultation",
        "is_weekend": 0,
        "threshold": 0.50,
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422


def test_reproducibility(tmp_path):
    path1 = tmp_path / "rep1.csv"
    path2 = tmp_path / "rep2.csv"
    df1 = generate_synthetic_dataset(n_samples=200, random_seed=42, output_path=path1)
    df2 = generate_synthetic_dataset(n_samples=200, random_seed=42, output_path=path2)
    pd.testing.assert_frame_equal(df1, df2)
