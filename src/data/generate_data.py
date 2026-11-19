import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from src.utils.config import (
    RAW_DATA_PATH,
    DATA_RAW_DIR,
    RANDOM_SEED,
    BOOKING_CHANNELS,
    REMINDER_PREFERENCES,
    APPOINTMENT_TYPES,
    WEEKDAYS,
    MONTHS,
)


def generate_synthetic_dataset(
    n_samples: int = 6000,
    random_seed: int = RANDOM_SEED,
    output_path: Path = RAW_DATA_PATH,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)

    start_date = datetime(2024, 1, 1)
    booking_offsets = rng.integers(0, 330, size=n_samples)
    booking_dates = [start_date + timedelta(days=int(offset)) for offset in booking_offsets]

    lead_times = np.round(rng.exponential(scale=10.0, size=n_samples)).astype(int)
    lead_times = np.clip(lead_times, 0, 60)

    appointment_dates = [
        b_date + timedelta(days=int(lt))
        for b_date, lt in zip(booking_dates, lead_times)
    ]

    appointment_weekdays = [WEEKDAYS[d.weekday()] for d in appointment_dates]
    appointment_months = [MONTHS[d.month - 1] for d in appointment_dates]
    is_weekends = [1 if d.weekday() >= 5 else 0 for d in appointment_dates]

    hours_pool = np.array([8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
    hour_probs = np.array([0.08, 0.12, 0.14, 0.12, 0.08, 0.08, 0.13, 0.12, 0.08, 0.05])
    hour_probs = hour_probs / hour_probs.sum()
    appointment_hours = rng.choice(hours_pool, size=n_samples, p=hour_probs)

    channel_probs = [0.42, 0.33, 0.15, 0.10]
    booking_channels = rng.choice(BOOKING_CHANNELS, size=n_samples, p=channel_probs)

    reminder_probs = [0.50, 0.25, 0.15, 0.10]
    reminder_preferences = rng.choice(REMINDER_PREFERENCES, size=n_samples, p=reminder_probs)

    type_probs = [0.35, 0.25, 0.20, 0.12, 0.08]
    appointment_types = rng.choice(APPOINTMENT_TYPES, size=n_samples, p=type_probs)

    previous_appointments = rng.poisson(lam=2.5, size=n_samples)
    previous_appointments = np.clip(previous_appointments, 0, 20)

    previous_no_shows = np.zeros(n_samples, dtype=int)
    for idx, prev_count in enumerate(previous_appointments):
        if prev_count > 0:
            hist_rate = rng.beta(1.5, 5.0)
            previous_no_shows[idx] = rng.binomial(prev_count, hist_rate)

    base_logit = -1.45
    lead_time_effect = 0.028 * lead_times
    prev_ratio = previous_no_shows / (previous_appointments + 1.0)
    history_effect = 1.95 * prev_ratio

    channel_effects = {
        "Online": 0.0,
        "Phone": 0.15,
        "In-Person": -0.22,
        "Referral": -0.28,
    }
    reminder_effects = {
        "SMS": -0.32,
        "Email": -0.18,
        "Phone Call": -0.36,
        "None": 0.48,
    }
    type_effects = {
        "General Consultation": 0.0,
        "Follow-up": -0.20,
        "Specialist": 0.10,
        "Diagnostic": 0.18,
        "Preventive Care": -0.25,
    }

    logits = (
        base_logit
        + lead_time_effect
        + history_effect
        + np.array([channel_effects[c] for c in booking_channels])
        + np.array([reminder_effects[r] for r in reminder_preferences])
        + np.array([type_effects[t] for t in appointment_types])
        + 0.28 * np.array(is_weekends)
        + rng.normal(0, 0.45, size=n_samples)
    )

    probs = 1.0 / (1.0 + np.exp(-logits))
    no_shows = (rng.random(size=n_samples) < probs).astype(int)

    appointment_ids = [f"APT-{100000 + i}" for i in range(n_samples)]

    df = pd.DataFrame(
        {
            "appointment_id": appointment_ids,
            "booking_date": [d.strftime("%Y-%m-%d") for d in booking_dates],
            "appointment_date": [d.strftime("%Y-%m-%d") for d in appointment_dates],
            "lead_time_days": lead_times,
            "appointment_weekday": appointment_weekdays,
            "appointment_month": appointment_months,
            "appointment_hour": appointment_hours,
            "booking_channel": booking_channels,
            "reminder_preference": reminder_preferences,
            "previous_appointments": previous_appointments,
            "previous_no_shows": previous_no_shows,
            "appointment_type": appointment_types,
            "is_weekend": is_weekends,
            "no_show": no_shows,
        }
    )

    df = df.sort_values(by=["booking_date", "appointment_date"]).reset_index(drop=True)

    missing_channel_mask = rng.random(size=n_samples) < 0.015
    missing_reminder_mask = rng.random(size=n_samples) < 0.02
    missing_prev_appts_mask = rng.random(size=n_samples) < 0.01

    df.loc[missing_channel_mask, "booking_channel"] = np.nan
    df.loc[missing_reminder_mask, "reminder_preference"] = np.nan
    df.loc[missing_prev_appts_mask, "previous_appointments"] = np.nan

    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    generate_synthetic_dataset()
