import pytest
from pydantic import ValidationError
from app.schemas import PredictionRequest, FeedbackRequest
import math

def test_valid_prediction_request():
    req = PredictionRequest(features=[0.0] * 10)
    assert len(req.features) == 10
    assert req.request_id is not None

def test_wrong_feature_count():
    with pytest.raises(ValidationError):
        PredictionRequest(features=[0.0] * 9)
    with pytest.raises(ValidationError):
        PredictionRequest(features=[0.0] * 11)

def test_nan_value_rejected():
    with pytest.raises(ValidationError):
        PredictionRequest(features=[float('nan')] + [0.0] * 9)

def test_inf_value_rejected():
    with pytest.raises(ValidationError):
        PredictionRequest(features=[float('inf')] + [0.0] * 9)
    with pytest.raises(ValidationError):
        PredictionRequest(features=[float('-inf')] + [0.0] * 9)

def test_metadata_too_many_keys():
    metadata = {f"key_{i}": "value" for i in range(6)}
    with pytest.raises(ValidationError):
        PredictionRequest(features=[0.0] * 10, metadata=metadata)

def test_feedback_invalid_ground_truth():
    with pytest.raises(ValidationError):
        FeedbackRequest(prediction_id="uuid-1234", ground_truth=2)
