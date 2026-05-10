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
from fastapi.responses import JSONResponse
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
    feedback_received_total
)
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response
from pydantic import BaseModel
import hashlib
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
    app.state.valid_api_keys = set(os.getenv("API_KEYS", "").split(",")) if os.getenv("API_KEYS") else set()
    app.state.valid_api_keys = {k.strip() for k in app.state.valid_api_keys if k.strip()}
    app.state.api_admin_key_hash = os.getenv("API_ADMIN_KEY_HASH", "").strip()
    
    app.state.startup_time = time.time()
    app.state.last_prediction_at = None
    app.state.model = None
    app.state.model_info = None

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
    
    # Convert payload accurately into a Pandas DataFrame expected by Scikit-Learn
    col_names = [f"feat_{i:02d}" for i in range(1, 11)]
    df = pd.DataFrame([payload.features], columns=col_names)
    
    try:
        model = app.state.model
        
        # Determine predict_proba dynamically if wrapped or pure sklearn
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(df)[0]
            probability = float(probs[1])  # Assumes standard binary classification setup
            prediction = int(np.argmax(probs))
        else:
            # Fallback if the underlying pipeline hides predict_proba
            prediction = int(model.predict(df)[0])
            probability = 1.0 if prediction == 1 else 0.0

        latency = (time.time() - start_ts) * 1000
        app.state.last_prediction_at = time.time()
        
        ver = str(app.state.model_info["version"]) if app.state.model_info else "unknown"
        
        prediction_latency_seconds.labels(model_version=ver).observe(latency / 1000.0)
        prediction_requests_total.labels(model_version=ver, prediction_class=str(prediction), status="success").inc()
        
        return PredictionResponse(
            prediction=prediction,
            probability=probability,
            model_version=ver,
            latency_ms=round(latency, 2),
            request_id=request.state.request_id
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
async def submit_feedback(payload: FeedbackRequest):
    """Logs ground-truth feedback against previous predictions dynamically."""
    log_line = {
        "prediction_id": payload.prediction_id,
        "ground_truth": payload.ground_truth,
        "timestamp": time.time()
    }
    
    try:
        with open("feedback_log.jsonl", "a") as f:
            f.write(json.dumps(log_line) + "\n")
        feedback_received_total.labels(ground_truth=str(payload.ground_truth)).inc()
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
