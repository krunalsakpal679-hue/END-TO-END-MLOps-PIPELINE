@echo off
echo Starting Anti-Gravity MLOps Project...
set API_KEYS=1311f8fc80a7ea28d78dd7723f09c44c1754cd35160ca8e7133ae3d7f636a19a
echo API Key is set to: my-secret-key
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
