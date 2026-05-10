import pytest
from fastapi.testclient import TestClient
from app.main import app
import hashlib
import time

client = TestClient(app)

# Dummy test key
TEST_KEY_PLAIN = "test_api_key_123"
TEST_KEY_HASH = hashlib.sha256(TEST_KEY_PLAIN.encode()).hexdigest()

@pytest.fixture(autouse=True)
def setup_api_keys():
    # Inject test key hash before each test
    app.state.valid_api_keys = {TEST_KEY_HASH}
    app.state.api_admin_key_hash = ""
    app.state.startup_time = time.time()
    app.state.model = None
    app.state.model_info = None
    app.state.last_prediction_at = None
    
    yield
    app.state.valid_api_keys = set()

def test_missing_api_key():
    # predict endpoint requires API key
    payload = {"features": [0.0] * 10}
    response = client.post("/v1/predict", json=payload)
    assert response.status_code == 401
    assert response.json() == {"error": "Missing API key"}

def test_invalid_api_key():
    payload = {"features": [0.0] * 10}
    response = client.post("/v1/predict", json=payload, headers={"X-API-Key": "wrong_key"})
    assert response.status_code == 403
    assert response.json() == {"error": "Invalid API key"}

def test_rate_limit():
    import uuid
    dynamic_test_key = "test_" + str(uuid.uuid4())
    dynamic_hash = hashlib.sha256(dynamic_test_key.encode()).hexdigest()
    app.state.valid_api_keys.add(dynamic_hash)
    
    payload = {"features": [0.0] * 10}
    headers = {"X-API-Key": dynamic_test_key}
    
    # Send 100 requests
    for _ in range(100):
        response = client.post("/v1/predict", json=payload, headers=headers)
        assert response.status_code != 429

    # The 101st request should be rate-limited
    response = client.post("/v1/predict", json=payload, headers=headers)
    assert response.status_code == 429
    assert response.json() == {"error": "Too Many Requests"}
    assert "Retry-After" in response.headers

def test_security_headers():
    response = client.get("/v1/health")  # Health doesn't require API key
    assert response.headers.get("X-Content-Type-Options") == "nosniff"
    assert response.headers.get("X-Frame-Options") == "DENY"
    assert response.headers.get("Strict-Transport-Security") == "max-age=31536000; includeSubDomains"
    assert response.headers.get("X-XSS-Protection") == "1; mode=block"
