from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, model_validator


class AppointmentInput(BaseModel):
    lead_time_days: int = Field(..., ge=0, le=365, description="Number of days between booking and appointment")
    appointment_weekday: str = Field(..., description="Day of the week (e.g., Monday)")
    appointment_month: str = Field(..., description="Month of the appointment (e.g., March)")
    appointment_hour: int = Field(..., ge=6, le=22, description="Hour of the appointment in 24h format")
    booking_channel: str = Field(..., description="Channel used to book: Online, Phone, In-Person, Referral")
    reminder_preference: str = Field(..., description="Preferred reminder channel: SMS, Email, Phone Call, None")
    previous_appointments: int = Field(..., ge=0, le=100, description="Total number of prior appointments")
    previous_no_shows: int = Field(..., ge=0, le=100, description="Number of prior missed appointments")
    appointment_type: str = Field(..., description="Type of consultation: General Consultation, Follow-up, Specialist, Diagnostic, Preventive Care")
    is_weekend: int = Field(..., ge=0, le=1, description="1 if appointment is on Saturday or Sunday, else 0")
    threshold: Optional[float] = Field(default=0.50, ge=0.01, le=0.99, description="Decision threshold for binary classification")

    @model_validator(mode="after")
    def validate_no_shows_not_exceed_appointments(self):
        if self.previous_no_shows > self.previous_appointments:
            raise ValueError("previous_no_shows cannot be greater than previous_appointments")
        return self


class PredictionResponse(BaseModel):
    no_show_probability: float
    predicted_no_show: bool
    risk_level: str
    threshold: float
    model_name: str
    key_factors: Dict[str, Any]
    disclaimer: str = "Decision-support estimate requiring human review. Not a clinical diagnosis or automated care restriction."


class BatchPredictionRequest(BaseModel):
    appointments: List[AppointmentInput]
    threshold: Optional[float] = Field(default=0.50, ge=0.01, le=0.99)


class BatchPredictionResponse(BaseModel):
    total_processed: int
    predictions: List[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str


class ModelInfoResponse(BaseModel):
    model_name: str
    features: List[str]
    default_threshold: float
    optimal_threshold: float
    training_samples: int
    test_metrics: Dict[str, Any]
