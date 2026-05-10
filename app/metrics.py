from prometheus_client import Counter, Histogram, Gauge, Info

prediction_requests_total = Counter(
    "prediction_requests_total",
    "Counts total number of predictions served",
    ["model_version", "prediction_class", "status"]
)

prediction_latency_seconds = Histogram(
    "prediction_latency_seconds",
    "Measures how long each prediction takes",
    ["model_version"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1.0]
)

model_load_duration_seconds = Gauge(
    "model_load_duration_seconds",
    "Records how long the last model load took (updated each time the model is reloaded)"
)

# Info automatically appends '_info' to the metric name, so 'active_model' becomes 'active_model_info'
active_model_info = Info(
    "active_model",
    "Records metadata about the currently loaded model"
)

feedback_received_total = Counter(
    "feedback_received_total",
    "Counts feedback submissions",
    ["ground_truth"]
)

drift_psi_score = Gauge(
    "drift_psi_score",
    "Stores the latest PSI drift score per feature",
    ["feature_name"]
)
