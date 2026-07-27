import os
import time
import requests
import subprocess
import uuid
import hashlib

def setup_mock_model():
    import mlflow
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    
    os.environ.pop("MLFLOW_TRACKING_URI", None)
    os.environ.pop("AWS_ACCESS_KEY_ID", None)
    os.environ.pop("AWS_SECRET_ACCESS_KEY", None)
    os.environ.pop("MLFLOW_S3_ENDPOINT_URL", None)
    mlflow.set_tracking_uri("sqlite:///test_mlflow.db")
    os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///test_mlflow.db"
    mlflow.set_experiment("test_fastapi_exp")
    
    model = LogisticRegression()
    X = np.random.rand(10, 10)
    y = np.array([0, 1]*5)
    model.fit(X, y)
    
    with mlflow.start_run():
        mlflow.sklearn.log_model(model, "model", registered_model_name="anti_gravity_v1")
        
    client = mlflow.tracking.MlflowClient()
    versions = client.get_latest_versions("anti_gravity_v1")
    for v in versions:
        client.transition_model_version_stage("anti_gravity_v1", v.version, "Production")

def test():
    import sys
    setup_mock_model()
    
    # Setup API Key for testing
    test_key = "test-api-key-123"
    test_key_hash = hashlib.sha256(test_key.encode()).hexdigest()
    os.environ["API_KEYS"] = test_key_hash
    headers = {"X-API-Key": test_key}

    print("Starting uvicorn server...")
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--port", "8000"], env=os.environ)
    
    # Try to catch the 503 before MLflow model loads
    ready_503_seen = False
    for _ in range(50):
        try:
            resp = requests.get("http://localhost:8000/v1/ready")
            if resp.status_code == 503:
                ready_503_seen = True
            elif resp.status_code == 200:
                print("App fully ready!")
                break
        except requests.exceptions.ConnectionError:
            time.sleep(0.1)
            
    # Wait for full start if needed
    time.sleep(1)
    
    try:
        print("Testing predict...")
        # 10 features list
        features = [0.5, 1.2, 0.1, -0.5, 0.0, 1.0, 2.0, -1.0, 0.5, -0.2]
        resp = requests.post("http://localhost:8000/v1/predict", json={"features": features}, headers=headers)
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}. Response: {resp.text}"
        data = resp.json()
        assert "prediction" in data
        assert "probability" in data
        assert 0.0 <= data["probability"] <= 1.0
        
        print("Testing wrong length predict...")
        resp_wrong = requests.post("http://localhost:8000/v1/predict", json={"features": [1.0] * 5}, headers=headers)
        assert resp_wrong.status_code == 200
        
        print("Testing health...")
        resp_h = requests.get("http://localhost:8000/v1/health")
        assert resp_h.status_code == 200
        h_data = resp_h.json()
        for k in ["status", "model_name", "model_version", "uptime_seconds", "last_prediction_at"]:
            assert k in h_data
            
        print("Testing feedback...")
        if os.path.exists("feedback_log.jsonl"):
            os.remove("feedback_log.jsonl")
        resp_f = requests.post("http://localhost:8000/v1/feedback", json={"prediction_id": "test-uuid-1234", "ground_truth": 1}, headers=headers)
        assert resp_f.status_code == 200
        with open("feedback_log.jsonl") as f:
            lines = f.readlines()
        assert len(lines) == 1
        
        print("Testing X-Request-ID...")
        assert "x-request-id" in resp.headers or "X-Request-ID" in resp.headers
        req_id = resp.headers.get("x-request-id", resp.headers.get("X-Request-ID"))
        uuid.UUID(req_id) # should not raise error
        
        print("Testing Swagger /docs...")
        resp_docs = requests.get("http://localhost:8000/docs")
        assert resp_docs.status_code == 200
        
        print("ALL FASTAPI CHECKS PASSED")
        if ready_503_seen:
            print("Successfully verified 503 while model was loading!")
            
    finally:
        proc.terminate()
        proc.wait()

if __name__ == "__main__":
    test()
