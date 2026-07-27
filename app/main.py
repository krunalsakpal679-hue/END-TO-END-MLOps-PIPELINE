import os
import time
import json
import uuid
import logging
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import Callable

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
import pandas as pd
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient

# Import Pydantic models from the schemas module
from app.schemas import PredictionRequest, PredictionResponse, FeedbackRequest

from app.metrics import (
    prediction_requests_total,
    prediction_latency_seconds,
    model_load_duration_seconds,
    active_model_info,
    feedback_received_total,
    feedback_accuracy_percentage
)
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response
from pydantic import BaseModel
import hashlib
import io
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from app.middleware import APIKeyMiddleware, RateLimitMiddleware, InputSanitizationMiddleware, AuditLogMiddleware

# ---------------------------------------------------------------------------
# Structured JSON Logging Setup
# ---------------------------------------------------------------------------
logger = logging.getLogger("api_logger")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(handler)

def fetch_model_info(model_name: str, stage: str):
    """Retrieve model registry tags securely using MlflowClient."""
    try:
        client = MlflowClient()
        versions = client.get_latest_versions(model_name, stages=[stage])
        if not versions:
            return None
        version = versions[0]
        run = client.get_run(version.run_id)
        return {
            "model_name": model_name,
            "version": str(version.version),
            "stage": stage,
            "training_date": run.data.tags.get("training_date", "Unknown")
        }
    except Exception as e:
        logger.error(json.dumps({"event": "fetch_mlflow_tags_failed", "error": str(e)}))
        return None

# ---------------------------------------------------------------------------
# Lifespan Context Manager (Modern Startup/Shutdown Handling)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Set default MLflow tracking database fallback (Upgrade B)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri or tracking_uri.startswith("./") or "localhost" not in tracking_uri and not tracking_uri.startswith("sqlite") and not tracking_uri.startswith("postgres"):
        tracking_uri = "sqlite:///mlflow.db"
        os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    mlflow.set_tracking_uri(tracking_uri)

    api_keys_env = os.getenv("API_KEYS", "")
    if not api_keys_env:
        # Default to SHA-256 hash of "my-secret-key" for seamless local development/demo
        api_keys_env = "1311f8fc80a7ea28d78dd7723f09c44c1754cd35160ca8e7133ae3d7f636a19a"
    app.state.valid_api_keys = set(api_keys_env.split(","))
    app.state.valid_api_keys = {k.strip() for k in app.state.valid_api_keys if k.strip()}
    app.state.api_admin_key_hash = os.getenv("API_ADMIN_KEY_HASH", "").strip()
    
    from collections import OrderedDict
    app.state.prediction_cache = OrderedDict()
    app.state.startup_time = time.time()
    app.state.last_prediction_at = None
    app.state.model = None
    app.state.model_info = None
    app.state.model_metadata = {
        "source_file": "Default Dataset",
        "algorithm": "Gradient Boosting",
        "last_trained": "System Startup",
        "feature_names": [f"Feature {i}" for i in range(1, 11)]
    }

    model_uri = "models:/anti_gravity_v1/Production"
    try:
        logger.info(json.dumps({"event": "startup_model_load_started", "uri": model_uri}))
        
        load_start = time.time()
        # Load directly using mlflow.sklearn to perfectly map to model.predict_proba()
        # Fallback to standard pyfunc wrapper if not strictly an sklearn payload
        try:
            model = mlflow.sklearn.load_model(model_uri)
        except Exception:
            model = mlflow.pyfunc.load_model(model_uri)
        load_time = time.time() - load_start
        model_load_duration_seconds.set(load_time)
            
        app.state.model = model
        
        info = fetch_model_info("anti_gravity_v1", "Production")
        if info:
            app.state.model_info = info
            active_model_info.info({
                "name": info["model_name"],
                "version": info["version"],
                "stage": info["stage"]
            })
        else:
            app.state.model_info = {
                "model_name": "anti_gravity_v1",
                "version": "unknown",
                "stage": "Production",
                "training_date": "unknown"
            }
            active_model_info.info({
                "name": "anti_gravity_v1",
                "version": "unknown",
                "stage": "Production"
            })
            
        logger.info(json.dumps({"event": "startup_model_loaded_successfully", "version": app.state.model_info["version"]}))
    except Exception as e:
        active_model_info.info({
            "name": "anti_gravity_v1",
            "version": "unknown",
            "stage": "Production"
        })
        # Crucial Requirement: Log CRITICAL error and continue without crashing server
        logger.critical(json.dumps({"event": "startup_model_load_failed", "error": str(e)}))
        
    yield
    logger.info(json.dumps({"event": "shutdown"}))

# ---------------------------------------------------------------------------
# FastAPI Application Instantiation
# ---------------------------------------------------------------------------
app = FastAPI(title="Anti-Gravity Classifier API v1", lifespan=lifespan)

# ---------------------------------------------------------------------------
# CORS & Middlewares
# ---------------------------------------------------------------------------
allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "")
if allowed_origins_env.strip():
    origins = [orig.strip() for orig in allowed_origins_env.split(",") if orig.strip()]
else:
    origins = ["*"]

app.add_middleware(InputSanitizationMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(APIKeyMiddleware)
app.add_middleware(AuditLogMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve Static Files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/", tags=["Frontend"])
async def read_index():
    """Serves the modern predictive dashboard."""
    return FileResponse("app/static/index.html")

@app.middleware("http")
async def struct_logger_and_request_id(request: Request, call_next: Callable):
    req_id = str(uuid.uuid4())
    request.state.request_id = req_id
    
    start_time = time.time()
    response = await call_next(request)
    latency_ms = (time.time() - start_time) * 1000
    
    response.headers["X-Request-ID"] = req_id
    
    log_payload = {
        "method": request.method,
        "path": request.url.path,
        "status_code": response.status_code,
        "latency_ms": round(latency_ms, 2),
        "request_id": req_id
    }
    logger.info(json.dumps(log_payload))
    
    return response

# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.post("/v1/predict", response_model=PredictionResponse)
async def predict_endpoint(payload: PredictionRequest, request: Request):
    """High performance inference endpoint. Yields classification class alongside confidence probability."""
    if app.state.model is None:
        raise HTTPException(status_code=503, detail="Model unavailable/not loaded")
        
    start_ts = time.time()
    
    # Use the feature names stored during training
    col_names = request.app.state.model_metadata.get("feature_names", [f"feat_{i}" for i in range(len(payload.features))])
    
    # Ensure feature count matches payload
    if len(payload.features) < len(col_names):
        # Pad with zeros if payload is short
        payload.features.extend([0.0] * (len(col_names) - len(payload.features)))
    
    df = pd.DataFrame([payload.features[:len(col_names)]], columns=col_names)
    
    try:
        model = app.state.model
        
        # Determine predict_proba dynamically if wrapped or pure sklearn
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(df)[0]
            prediction = int(np.argmax(probs))
            probability = float(probs[prediction])  # Return the confidence of the prediction
        else:
            # Fallback if the underlying pipeline hides predict_proba
            prediction = int(model.predict(df)[0])
            probability = 1.0

        latency = (time.time() - start_ts) * 1000
        app.state.last_prediction_at = time.time()
        
        ver = str(app.state.model_info["version"]) if app.state.model_info else "unknown"
        
        prediction_latency_seconds.labels(model_version=ver).observe(latency / 1000.0)
        prediction_requests_total.labels(model_version=ver, prediction_class=str(prediction), status="success").inc()
        
        # Cache the prediction outcome for feedback tracing (Upgrade C)
        req_id = request.state.request_id
        request.app.state.prediction_cache[req_id] = prediction
        if len(request.app.state.prediction_cache) > 10000:
            request.app.state.prediction_cache.popitem(last=False)
            
        return PredictionResponse(
            prediction=prediction,
            probability=probability,
            model_version=ver,
            latency_ms=round(latency, 2),
            request_id=req_id
        )
    except Exception as e:
        ver = str(app.state.model_info["version"]) if (hasattr(app.state, "model_info") and app.state.model_info) else "unknown"
        prediction_requests_total.labels(model_version=ver, prediction_class="unknown", status="error").inc()
        logger.error(json.dumps({"event": "prediction_execution_failed", "error": str(e), "request_id": request.state.request_id}))
        raise HTTPException(status_code=500, detail="Inference execution failed")

@app.get("/v1/health")
async def health_check():
    """Liveness probe. Will always return 200 OK as long as the webserver is breathing."""
    uptime = time.time() - app.state.startup_time
    status_str = "healthy" if app.state.model is not None else "degraded"
    
    m_name = app.state.model_info["model_name"] if app.state.model_info else "unknown"
    m_ver = app.state.model_info["version"] if app.state.model_info else "unknown"
    
    last_pred_iso = None
    if app.state.last_prediction_at:
        last_pred_iso = datetime.fromtimestamp(app.state.last_prediction_at, tz=timezone.utc).isoformat()
        
    return {
        "status": status_str,
        "model_name": m_name,
        "model_version": m_ver,
        "uptime_seconds": round(uptime, 2),
        "last_prediction_at": last_pred_iso
    }

@app.get("/v1/ready")
async def readiness_probe():
    """Readiness probe. Checks if ML weights have been fully materialized into memory."""
    if app.state.model is not None:
        return {"status": "ready"}
    else:
        return JSONResponse(
            status_code=503, 
            content={"status": "not ready", "reason": "Model is not loaded yet"}
        )

@app.post("/v1/feedback")
async def submit_feedback(payload: FeedbackRequest, request: Request):
    """Logs ground-truth feedback against previous predictions dynamically."""
    prediction_id = payload.prediction_id
    ground_truth = payload.ground_truth
    
    # Check if prediction is cached in memory to resolve correctness (Upgrade C)
    prediction = request.app.state.prediction_cache.get(prediction_id)
    is_correct = None
    if prediction is not None:
        is_correct = (prediction == ground_truth)
        
    log_line = {
        "prediction_id": prediction_id,
        "ground_truth": ground_truth,
        "prediction": prediction,
        "is_correct": is_correct,
        "timestamp": time.time()
    }
    
    try:
        with open("feedback_log.jsonl", "a") as f:
            f.write(json.dumps(log_line) + "\n")
        feedback_received_total.labels(ground_truth=str(ground_truth)).inc()
        
        # Calculate rolling feedback accuracy metrics dynamically over the last 100 submissions
        try:
            if os.path.exists("feedback_log.jsonl"):
                with open("feedback_log.jsonl", "r") as f:
                    lines = f.readlines()
                recent_feedbacks = []
                for line in reversed(lines):
                    try:
                        data = json.loads(line.strip())
                        if data.get("is_correct") is not None:
                            recent_feedbacks.append(data["is_correct"])
                            if len(recent_feedbacks) >= 100:
                                break
                    except Exception:
                        pass
                if recent_feedbacks:
                    accuracy = sum(1 for x in recent_feedbacks if x) / len(recent_feedbacks)
                    feedback_accuracy_percentage.set(accuracy)
        except Exception as e:
            logger.error(f"Failed to calculate rolling feedback accuracy: {e}")
            
        return {"status": "received"}
    except Exception as e:
        logger.error(json.dumps({"event": "feedback_write_failed", "error": str(e)}))
        raise HTTPException(status_code=500, detail="Feedback storage error")

@app.get("/v1/info")
async def model_metadata():
    """Returns static model origin and build metadata fetched from MLflow directly."""
    if app.state.model_info:
        return app.state.model_info
    return {"status": "Information unavailable"}

@app.get("/v1/metrics")
async def metrics_endpoint():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

class RotateKeyRequest(BaseModel):
    new_key_hash: str

@app.post("/v1/admin/rotate-key")
async def rotate_key(request: Request, payload: RotateKeyRequest):
    admin_key = request.headers.get("X-Admin-Key", "")
    if not admin_key:
        raise HTTPException(status_code=401, detail="Missing admin key")
    
    admin_hash = hashlib.sha256(admin_key.encode()).hexdigest()
    if admin_hash != request.app.state.api_admin_key_hash and admin_key != request.app.state.api_admin_key_hash:
        raise HTTPException(status_code=403, detail="Invalid admin key")
        
    request.app.state.valid_api_keys.add(payload.new_key_hash)
    return {"status": "key added"}

@app.post("/v1/train", tags=["Training"])
async def train_model(
    request: Request,
    algorithm: str = "random_forest",
    test_size: float = 0.2
):
    """Interactive training endpoint: Upload CSV and train a model."""
    # Check for CSV file in request
    form = await request.form()
    file = form.get("file")
    if not file:
        raise HTTPException(status_code=400, detail="No CSV file provided")
    
    try:
        # Load the CSV with fallback encoding for special characters
        contents = await file.read()
        try:
            df = pd.read_csv(io.BytesIO(contents), encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(io.BytesIO(contents), encoding='latin-1')
        
        # --- SMART DATA PREPROCESSING ---
        # 1. Drop obvious ID/Name columns that don't help prediction
        to_drop = [col for col in df.columns if any(word in col.lower() for word in ['id', 'name', 'email', 'phone', 'address'])]
        df = df.drop(columns=to_drop)
        
        # 2. Fill missing values
        df = df.fillna(df.mode().iloc[0]) 

        # 3. Identify Target and Features BEFORE Encoding
        # By default assume the last column is the target, but check for common label names
        target_col = df.columns[-1]
        for col in df.columns:
            if col.lower() in ['target', 'label', 'class', 'v1']:
                target_col = col
                break
                
        y = df[target_col]
        X = df.drop(columns=[target_col])
        
        # Factorize the target if it's text (e.g. 'ham'/'spam' -> 0/1)
        if y.dtype == 'object':
            y, _ = pd.factorize(y)
            
        # 4. Handle Categorical Data in Features (One-Hot Encoding)
        # This safely creates dummy variables for text features without corrupting the target
        X = pd.get_dummies(X)
        
        feature_names = X.columns.tolist()
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Select algorithm
        if algorithm == "random_forest":
            clf = RandomForestClassifier(n_estimators=100)
        elif algorithm == "logistic_regression":
            clf = LogisticRegression(max_iter=1000)
        elif algorithm == "gradient_boosting":
            clf = GradientBoostingClassifier()
        else:
            raise HTTPException(status_code=400, detail="Unsupported algorithm")
            
        # Unified feature preprocessing pipeline to eliminate training-serving skew
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer
        
        model = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('classifier', clf)
        ])
            
        model.fit(X_train, y_train)
        
        # Evaluate
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average='weighted')
        
        # Log to MLflow
        with mlflow.start_run(run_name=f"manual_train_{algorithm}"):
            mlflow.log_param("algorithm", algorithm)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_score", f1)
            # Passing pip_requirements skips the very slow environment inference
            mlflow.sklearn.log_model(
                model, 
                "model", 
                registered_model_name="anti_gravity_v1",
                pip_requirements=["scikit-learn", "pandas", "numpy"]
            )
            
        # Update active model in memory immediately
        request.app.state.model = model
        request.app.state.model_info = {
            "model_name": "anti_gravity_v1",
            "version": "latest-trained",
            "stage": "Development",
            "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
            
        # Update model metadata for the UI
        request.app.state.model_metadata = {
            "source_file": file.filename,
            "algorithm": algorithm.replace('_', ' ').title(),
            "last_trained": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "feature_names": feature_names
        }
            
        return {
            "status": "success",
            "algorithm": algorithm,
            "metrics": {
                "accuracy": round(acc, 4),
                "f1_score": round(f1, 4)
            },
            "message": "Model trained and registered in MLflow"
        }
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.get("/v1/model-info", tags=["Infrastructure"])
async def get_model_info(request: Request):
    """Returns metadata about the active model being used."""
    return request.app.state.model_metadata
