import time
import json
import math
import hashlib
from datetime import datetime, timezone
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

class APIKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Exclude health and metrics endpoints
        if request.method == "OPTIONS" or request.url.path in ["/v1/health", "/v1/ready", "/v1/metrics", "/docs", "/openapi.json", "/v1/admin/rotate-key"]:
            return await call_next(request)
            
        api_key = request.headers.get("X-API-Key")
        if not api_key:
            return JSONResponse(status_code=401, content={"error": "Missing API key"})
            
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        valid_keys = getattr(request.app.state, "valid_api_keys", set())
        
        if key_hash not in valid_keys:
            return JSONResponse(status_code=403, content={"error": "Invalid API key"})
            
        request.state.api_key_hash_prefix = key_hash[:8]
        return await call_next(request)

class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self.request_counts = {}
        
    async def dispatch(self, request: Request, call_next):
        if request.method == "OPTIONS" or request.url.path in ["/v1/health", "/v1/ready", "/v1/metrics", "/docs", "/openapi.json", "/v1/admin/rotate-key"]:
            return await call_next(request)
            
        key_hash_prefix = getattr(request.state, "api_key_hash_prefix", None)
        if key_hash_prefix:
            now = time.time()
            timestamps = self.request_counts.get(key_hash_prefix, [])
            timestamps = [t for t in timestamps if now - t < 60]
            if len(timestamps) >= 100:
                return JSONResponse(status_code=429, content={"error": "Too Many Requests"}, headers={"Retry-After": "60"})
            timestamps.append(now)
            self.request_counts[key_hash_prefix] = timestamps
            
        return await call_next(request)

class InputSanitizationMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path == "/v1/predict" and request.method == "POST":
            body = await request.body()
            try:
                payload = json.loads(body)
                features = payload.get("features", [])
                for idx, val in enumerate(features):
                    if not isinstance(val, (int, float)):
                        continue
                    if math.isnan(val) or math.isinf(val) or val < -1000000 or val > 1000000:
                        return JSONResponse(status_code=422, content={"error": f"Invalid feature value at index {idx}: {val}"})
            except Exception:
                pass
            
            async def receive():
                return {"type": "http.request", "body": body}
            request._receive = receive
            
        return await call_next(request)

class AuditLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_ts = time.time()
        response = await call_next(request)
        latency_ms = (time.time() - start_ts) * 1000
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["X-XSS-Protection"] = "1; mode=block"

        prefix = getattr(request.state, "api_key_hash_prefix", "none")
        log_line = {
            "timestamp": datetime.fromtimestamp(start_ts, tz=timezone.utc).isoformat(),
            "api_key_hash_prefix": prefix,
            "endpoint": request.url.path,
            "status_code": response.status_code,
            "latency_ms": round(latency_ms, 2)
        }
        try:
            with open("audit.jsonl", "a") as f:
                f.write(json.dumps(log_line) + "\n")
        except Exception:
            pass
            
        return response
