import json
from pathlib import Path
from typing import Dict, Any, List
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from src.utils.config import (
    MODEL_PATH,
    METADATA_PATH,
    TEST_METRICS_PATH,
    PROCESSED_DATA_PATH,
    RISK_THRESHOLD_LOW,
    RISK_THRESHOLD_HIGH,
    DEFAULT_THRESHOLD,
)
from src.api.schemas import (
    AppointmentInput,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
)
from src.models.explainability import explain_single_prediction

app = FastAPI(
    title="Hospital Appointment No-Show Prediction API",
    description="Decision-support inference API for hospital appointment attendance risk assessment.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_cached_model = None
_cached_metadata = None
_cached_train_sample = None


def get_model():
    global _cached_model
    if _cached_model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model artifact not found at {MODEL_PATH}. Run training pipeline first.",
            )
        try:
            _cached_model = joblib.load(MODEL_PATH)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to load model artifact: {str(e)}",
            )
    return _cached_model


def get_metadata() -> Dict[str, Any]:
    global _cached_metadata
    if _cached_metadata is None:
        if METADATA_PATH.exists():
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                _cached_metadata = json.load(f)
        else:
            _cached_metadata = {}
    return _cached_metadata


def get_train_sample() -> pd.DataFrame:
    global _cached_train_sample
    if _cached_train_sample is None:
        if PROCESSED_DATA_PATH.exists():
            df = pd.read_csv(PROCESSED_DATA_PATH)
            _cached_train_sample = df.head(300)
        else:
            _cached_train_sample = pd.DataFrame()
    return _cached_train_sample


def determine_risk_level(prob: float) -> str:
    if prob < RISK_THRESHOLD_LOW:
        return "Low"
    elif prob < RISK_THRESHOLD_HIGH:
        return "Moderate"
    else:
        return "High"


@app.get("/", tags=["General"])
def root():
    return {
        "message": "Welcome to Hospital Appointment No-Show Prediction System API",
        "documentation": "/docs",
        "health_check": "/health",
        "model_info": "/model-info",
        "version": "1.0.0",
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check():
    model_exists = MODEL_PATH.exists()
    return HealthResponse(
        status="healthy" if model_exists else "degraded",
        model_loaded=model_exists,
        model_path=str(MODEL_PATH),
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Monitoring"])
def model_info():
    meta = get_metadata()
    if not meta:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model metadata is not available. Please execute training pipeline.",
        )
    return ModelInfoResponse(
        model_name=meta.get("model_name", "Unknown"),
        features=meta.get("features", []),
        default_threshold=meta.get("default_threshold", 0.50),
        optimal_threshold=meta.get("optimal_threshold", 0.50),
        training_samples=meta.get("training_samples", 0),
        test_metrics=meta.get("test_metrics_default", {}),
    )


@app.get("/metrics", tags=["Monitoring"])
def get_metrics():
    if not TEST_METRICS_PATH.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Test evaluation metrics not found.",
        )
    with open(TEST_METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    return metrics


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(appointment: AppointmentInput):
    model = get_model()
    meta = get_metadata()
    train_sample = get_train_sample()

    input_dict = appointment.model_dump()
    threshold = input_dict.pop("threshold", DEFAULT_THRESHOLD)
    if threshold is None:
        threshold = DEFAULT_THRESHOLD

    input_df = pd.DataFrame([input_dict])

    try:
        prob = float(model.predict_proba(input_df)[0, 1])
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}",
        )

    predicted_class = bool(prob >= threshold)
    risk_level = determine_risk_level(prob)

    explanation = {}
    if not train_sample.empty:
        try:
            explanation = explain_single_prediction(model, input_df, train_sample)
        except Exception:
            explanation = {"error": "Explanation unavailable for input"}

    return PredictionResponse(
        no_show_probability=round(prob, 4),
        predicted_no_show=predicted_class,
        risk_level=risk_level,
        threshold=round(threshold, 4),
        model_name=meta.get("model_name", "Ensemble Pipeline"),
        key_factors=explanation,
        disclaimer="Decision-support estimate requiring human review. Not a clinical diagnosis or automated care restriction.",
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_batch(batch_req: BatchPredictionRequest):
    model = get_model()
    meta = get_metadata()
    train_sample = get_train_sample()

    records = [item.model_dump() for item in batch_req.appointments]
    if not records:
        return BatchPredictionResponse(total_processed=0, predictions=[])

    threshold = batch_req.threshold if batch_req.threshold is not None else DEFAULT_THRESHOLD
    df_batch = pd.DataFrame(records)
    if "threshold" in df_batch.columns:
        df_batch = df_batch.drop(columns=["threshold"])

    try:
        probs = model.predict_proba(df_batch)[:, 1]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference failed: {str(e)}",
        )

    results = []
    for idx, prob in enumerate(probs):
        prob_float = float(prob)
        predicted_class = bool(prob_float >= threshold)
        risk_level = determine_risk_level(prob_float)
        single_row = df_batch.iloc[[idx]]

        explanation = {}
        if not train_sample.empty:
            try:
                explanation = explain_single_prediction(model, single_row, train_sample)
            except Exception:
                explanation = {}

        pred_resp = PredictionResponse(
            no_show_probability=round(prob_float, 4),
            predicted_no_show=predicted_class,
            risk_level=risk_level,
            threshold=round(threshold, 4),
            model_name=meta.get("model_name", "Ensemble Pipeline"),
            key_factors=explanation,
            disclaimer="Decision-support estimate requiring human review. Not a clinical diagnosis or automated care restriction.",
        )
        results.append(pred_resp)

    return BatchPredictionResponse(
        total_processed=len(results),
        predictions=results,
    )
