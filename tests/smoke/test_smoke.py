import requests
import pytest
import os

def test_health_check():
    """Verify that the API is up and running."""
    base_url = os.environ.get("BASE_URL", "http://localhost:8000")
    response = requests.get(f"{base_url}/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_prediction_endpoint():
    """Verify that the prediction endpoint is responsive."""
    base_url = os.environ.get("BASE_URL", "http://localhost:8000")
    payload = {"features": [0.1] * 10}
    response = requests.post(f"{base_url}/v1/predict", json=payload)
    assert response.status_code == 200
    assert "prediction" in response.json()
