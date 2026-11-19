from typing import Tuple, List, Dict, Any
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from src.utils.config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
    DATA_PROCESSED_DIR,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    BINARY_FEATURES,
    TARGET_COLUMN,
)

ENGINEERED_NUMERICAL_FEATURES = NUMERICAL_FEATURES + ["previous_no_show_rate"]


def add_engineered_features(X: pd.DataFrame) -> pd.DataFrame:
    df_out = X.copy()
    if "previous_appointments" in df_out.columns and "previous_no_shows" in df_out.columns:
        prev_appts = pd.to_numeric(df_out["previous_appointments"], errors="coerce").fillna(0)
        prev_no_shows = pd.to_numeric(df_out["previous_no_shows"], errors="coerce").fillna(0)
        prev_no_shows = np.clip(prev_no_shows, 0, prev_appts)
        df_out["previous_no_show_rate"] = prev_no_shows / (prev_appts + 1.0)
    else:
        df_out["previous_no_show_rate"] = 0.0
    return df_out


def create_preprocessor() -> ColumnTransformer:
    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    binary_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, ENGINEERED_NUMERICAL_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
            ("bin", binary_transformer, BINARY_FEATURES),
        ]
    )

    return preprocessor


def create_full_pipeline(classifier: Any) -> Pipeline:
    feature_adder = FunctionTransformer(add_engineered_features, validate=False)
    preprocessor = create_preprocessor()

    full_pipeline = Pipeline(
        steps=[
            ("feature_adder", feature_adder),
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )
    return full_pipeline


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()

    cleaned = cleaned.drop_duplicates(subset=["appointment_id"]).reset_index(drop=True)

    if "booking_date" in cleaned.columns and "appointment_date" in cleaned.columns:
        booking_dt = pd.to_datetime(cleaned["booking_date"], errors="coerce")
        appt_dt = pd.to_datetime(cleaned["appointment_date"], errors="coerce")
        valid_dates = appt_dt >= booking_dt
        cleaned = cleaned[valid_dates].reset_index(drop=True)

    if "lead_time_days" in cleaned.columns:
        cleaned["lead_time_days"] = pd.to_numeric(cleaned["lead_time_days"], errors="coerce").fillna(0).clip(lower=0)

    if "previous_appointments" in cleaned.columns and "previous_no_shows" in cleaned.columns:
        cleaned["previous_appointments"] = pd.to_numeric(cleaned["previous_appointments"], errors="coerce").fillna(0).clip(lower=0)
        cleaned["previous_no_shows"] = pd.to_numeric(cleaned["previous_no_shows"], errors="coerce").fillna(0).clip(lower=0)
        cleaned["previous_no_shows"] = np.minimum(cleaned["previous_no_shows"], cleaned["previous_appointments"])

    if "booking_channel" in cleaned.columns:
        cleaned["booking_channel"] = cleaned["booking_channel"].fillna("Online")

    if "reminder_preference" in cleaned.columns:
        cleaned["reminder_preference"] = cleaned["reminder_preference"].fillna("None")

    if TARGET_COLUMN in cleaned.columns:
        cleaned = cleaned[cleaned[TARGET_COLUMN].isin([0, 1])].reset_index(drop=True)

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(PROCESSED_DATA_PATH, index=False)
    return cleaned


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_sorted = df.sort_values(by=["booking_date", "appointment_date"]).reset_index(drop=True)

    total_rows = len(df_sorted)
    train_end = int(total_rows * train_ratio)
    val_end = int(total_rows * (train_ratio + val_ratio))

    train_df = df_sorted.iloc[:train_end].copy().reset_index(drop=True)
    val_df = df_sorted.iloc[train_end:val_end].copy().reset_index(drop=True)
    test_df = df_sorted.iloc[val_end:].copy().reset_index(drop=True)

    return train_df, val_df, test_df


def extract_features_and_target(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    feature_columns = [
        col for col in (NUMERICAL_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES)
        if col in df.columns
    ]
    X = df[feature_columns].copy()
    y = df[TARGET_COLUMN].astype(int) if TARGET_COLUMN in df.columns else None
    return X, y
