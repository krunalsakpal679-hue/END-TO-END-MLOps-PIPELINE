import time
import logging
import threading
from datetime import datetime, timezone
import random
import re

import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

logger = logging.getLogger("model_loader")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(handler)

class ModelLoader:
    def __init__(self):
        self._model = None
        self._model_uri = None
        self._model_version = "unknown"
        self._loaded_at = None
        self._warmup_latency_ms = 0.0
        
        # Thread safety lock protecting the model object
        self._lock = threading.RLock()

    def _fetch_model_version(self, model_uri: str) -> str:
        """Helper to extract model version securely from MLflow."""
        version_str = "unknown"
        match = re.match(r"^models:/([^/]+)/(.+)$", model_uri)
        if match:
            name, stage_or_version = match.groups()
            try:
                if stage_or_version.isdigit():
                    version_str = stage_or_version
                else:
                    client = MlflowClient()
                    versions = client.get_latest_versions(name, stages=[stage_or_version])
                    if versions:
                        version_str = str(versions[0].version)
            except Exception as e:
                logger.warning(f"Could not fetch version for {model_uri}: {e}")
        return version_str

    def load(self, model_uri: str):
        """Loads the model, logs telemetry, and triggers warmup."""
        logger.info(f"Loading model from {model_uri}...")
        model = mlflow.pyfunc.load_model(model_uri)
        
        with self._lock:
            self._model = model
            self._model_uri = model_uri
            self._model_version = self._fetch_model_version(model_uri)
            self._loaded_at = datetime.now(timezone.utc)
            
        self._warmup_latency_ms = self.warmup()

    def reload(self, model_uri: str):
        """Loads a NEW version into a temp variable and atomically swaps it securely."""
        logger.info(f"Preparing to reload model from {model_uri}...")
        new_model = mlflow.pyfunc.load_model(model_uri)
        new_version = self._fetch_model_version(model_uri)
        new_loaded_at = datetime.now(timezone.utc)
        
        with self._lock:
            old_version = self._model_version
            # Atomically swap the pointer ensuring in-flight predictions are not corrupted
            self._model = new_model
            self._model_uri = model_uri
            self._model_version = new_version
            self._loaded_at = new_loaded_at
            
        logger.info(f"Model reloaded from {old_version} to {new_version}")
        
        self._warmup_latency_ms = self.warmup()

    def predict(self, features: list[float]) -> tuple[int, float]:
        """Validates input, translates to native array format, and returns predictions securely."""
        if not isinstance(features, list) or len(features) != 10:
            raise ValueError("features must be a list of exactly 10 floats")
            
        # Standardize features format explicitly to shape (1, 10)
        X = np.array(features, dtype=np.float64).reshape(1, 10)
        
        # Scikit-Learn models wrapped via mlflow pyfunc usually expect DataFrame mapping natively
        # But we pass the array explicitly ensuring robust typing.
        col_names = [f"feat_{i:02d}" for i in range(1, 11)]
        df = pd.DataFrame(X, columns=col_names)

        with self._lock:
            if self._model is None:
                raise RuntimeError("Model is not loaded")
                
            # Pyfunc strictly exposes .predict(), we must unwrap to access native predict_proba()
            # fallback to generic predict if wrapped object heavily abstracted.
            model_impl = getattr(self._model, "_model_impl", self._model)
            if hasattr(model_impl, "predict_proba"):
                probs = model_impl.predict_proba(df)[0]
                probability = float(probs[1])
                prediction = int(np.argmax(probs))
            else:
                # Absolute fallback boundary
                prediction = int(self._model.predict(df)[0])
                probability = 1.0 if prediction == 1 else 0.0
                
        return prediction, probability

    def warmup(self, n_calls: int = 10) -> float:
        """Executes sequential dummy predictions verifying memory load state and baseline latencies."""
        latencies = []
        for _ in range(n_calls):
            # Generate dummy payload securely inside valid feature space [-1M, +1M]
            dummy_features = [random.uniform(-10.0, 10.0) for _ in range(10)]
            
            start = time.time()
            self.predict(dummy_features)
            end = time.time()
            
            latencies.append((end - start) * 1000)
            
        avg_latency = float(np.mean(latencies))
        logger.info(f"Warmup complete. Average latency: {avg_latency:.2f}ms")
        return avg_latency

    def health_check(self) -> bool:
        """One-shot test prediction validating system viability and SLA compliance."""
        if self._model is None:
            return False
            
        dummy_features = [0.0] * 10
        start = time.time()
        try:
            self.predict(dummy_features)
            elapsed_ms = (time.time() - start) * 1000
            return elapsed_ms < 100.0
        except Exception as e:
            logger.error(f"Health check failed during prediction: {e}")
            return False

    def get_model_info(self) -> dict:
        """Returns internal state telemetry of the active execution pointer."""
        loaded_str = self._loaded_at.isoformat() if self._loaded_at else None
        return {
            "model_uri": self._model_uri,
            "model_version": self._model_version,
            "loaded_at": loaded_str,
            "warmup_latency_ms": round(self._warmup_latency_ms, 2)
        }
