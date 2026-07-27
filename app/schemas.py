from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import List, Dict, Optional, Literal, Any
import uuid
import math
from datetime import datetime, timezone

class PredictionRequest(BaseModel):
    features: List[float] = Field(
        ...,
        description="A list of floating point numbers matching the model features"
    )
    request_id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Optional[Dict[str, str]] = Field(default=None, description="Optional metadata")

    @field_validator('features')
    @classmethod
    def validate_features(cls, v: List[float]) -> List[float]:
        for val in v:
            if math.isnan(val) or math.isinf(val):
                raise ValueError("Features cannot contain NaN or Infinity values.")
            if not (-1_000_000 <= val <= 1_000_000):
                raise ValueError("Feature values must be between -1,000,000 and 1,000,000.")
        return v

    @field_validator('metadata')
    @classmethod
    def validate_metadata(cls, v: Optional[Dict[str, str]]) -> Optional[Dict[str, str]]:
        if v is None:
            return v
        if len(v) > 5:
            raise ValueError("Metadata can have a maximum of 5 keys.")
        for k, val in v.items():
            if len(k) > 64:
                raise ValueError(f"Metadata key '{k}' exceeds 64 characters.")
            if len(val) > 64:
                raise ValueError(f"Metadata value for key '{k}' exceeds 64 characters.")
        return v
    
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "features": [0.1, -1.5, 3.2, 0.0, 42.1, -9.9, 1.1, 2.2, 3.3, 4.4],
                    "request_id": "req-12345",
                    "metadata": {"client": "web_app", "user_segment": "premium"}
                },
                {
                    "features": [0.0]*10,
                    "metadata": {"source": "api"}
                }
            ]
        }
    )

class PredictionResponse(BaseModel):
    prediction: Literal[0, 1]
    probability: float = Field(..., ge=0.0, le=1.0)
    model_version: str
    latency_ms: float
    request_id: str

class FeedbackRequest(BaseModel):
    prediction_id: str
    ground_truth: Literal[0, 1]
    timestamp: Optional[datetime] = Field(default_factory=lambda: datetime.now(timezone.utc))

class HealthResponse(BaseModel):
    status: Literal['healthy', 'degraded']
    model_name: str
    model_version: str
    uptime_seconds: float
    last_prediction_at: Optional[datetime] = None

class ErrorResponse(BaseModel):
    error_code: str
    message: str
    detail: Optional[Any] = None
