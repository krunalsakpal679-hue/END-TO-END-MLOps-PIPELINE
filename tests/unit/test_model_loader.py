import pytest
import time
import threading
import random
import os
import numpy as np

from app.model_loader import ModelLoader

@pytest.fixture(scope="module")
def mock_model_uri():
    import mlflow
    from sklearn.linear_model import LogisticRegression
    
    os.environ.pop("MLFLOW_TRACKING_URI", None)
    mlflow.set_tracking_uri("./test_mlruns")
    mlflow.set_experiment("model_loader_exp")
    
    model = LogisticRegression()
    X = np.random.rand(10, 10)
    y = np.array([0, 1]*5)
    model.fit(X, y)
    
    with mlflow.start_run():
        mlflow.sklearn.log_model(model, "model", registered_model_name="anti_gravity_v1_loader")
        
    client = mlflow.tracking.MlflowClient()
    versions = client.get_latest_versions("anti_gravity_v1_loader")
    for v in versions:
        client.transition_model_version_stage("anti_gravity_v1_loader", v.version, "Production")
    
    return "models:/anti_gravity_v1_loader/Production"

def test_predict_before_load():
    loader = ModelLoader()
    with pytest.raises(RuntimeError):
        loader.predict([0.0]*10)

def test_load_and_warmup(mock_model_uri):
    loader = ModelLoader()
    loader.load(mock_model_uri)
    assert loader._warmup_latency_ms >= 0.0
    assert isinstance(loader._warmup_latency_ms, float)
    
def test_predict_results_and_validation(mock_model_uri):
    loader = ModelLoader()
    loader.load(mock_model_uri)
    
    prediction, prob = loader.predict([0.5]*10)
    assert isinstance(prediction, int)
    assert prediction in [0, 1]
    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0
    
    with pytest.raises(ValueError):
        loader.predict([0.5]*9)
        
    with pytest.raises(ValueError):
        loader.predict([0.5]*11)

def test_health_and_info(mock_model_uri):
    loader = ModelLoader()
    loader.load(mock_model_uri)
    
    assert loader.health_check() is True
    
    info = loader.get_model_info()
    assert set(info.keys()) == {"model_uri", "model_version", "loaded_at", "warmup_latency_ms"}
    assert info["model_uri"] == mock_model_uri
    
def test_concurrency_and_reload(mock_model_uri):
    loader = ModelLoader()
    loader.load(mock_model_uri)
    
    results = []
    exceptions = []
    
    def worker():
        for _ in range(50):
            try:
                res = loader.predict([random.random() for _ in range(10)])
                results.append(res)
            except Exception as e:
                exceptions.append(e)
            time.sleep(0.001)
            
    threads = [threading.Thread(target=worker) for _ in range(10)]
    
    for t in threads:
        t.start()
        
    # Reload while threads are running
    time.sleep(0.01)
    loader.reload(mock_model_uri)
    
    for t in threads:
        t.join()
        
    assert len(exceptions) == 0, f"Exceptions occurred: {exceptions}"
    assert len(results) == 500
