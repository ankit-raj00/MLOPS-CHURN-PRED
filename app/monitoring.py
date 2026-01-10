from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# --- Custom Business Metrics ---

# 1. Total Predictions (Counter)
# Labels: prediction_class (Churn/NotChurn), model_version (e.g., v1)
churn_prediction_total = Counter(
    "churn_prediction_total",
    "Total number of churn predictions made",
    ["prediction_class", "model_version"]
)

# 2. Prediction Latency (Histogram)
# Tracks how long the model takes to infer (excluding network time)
# Buckets optimized for ML inference (10ms to 2s)
prediction_latency_seconds = Histogram(
    "prediction_latency_seconds",
    "Time taken for model inference in seconds",
    buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
)

# 3. Model Confidence/Probability (Histogram)
# Tracks data drift (e.g. if model starts becoming unsure < 0.6)
churn_probability_histogram = Histogram(
    "churn_prediction_probability",
    "Distribution of churn probability scores (0.0 to 1.0)",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

# --- Metric Exposure Logic ---
def get_metrics():
    """Returns all metrics in Prometheus text format"""
    return generate_latest(), CONTENT_TYPE_LATEST
