from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import Response
from fastapi.responses import JSONResponse

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Summary,
    generate_latest
)

import pandas as pd
import joblib
import os
import time
import socket
import boto3

from botocore.client import Config


# ---------------------------------------------------
# FastAPI App
# ---------------------------------------------------

app = FastAPI(
    title="SmartPredict MLOps API",
    version="2.0.0"
)


# ---------------------------------------------------
# Application Metadata
# ---------------------------------------------------

APP_NAME = "smartpredict-mlops"

HOSTNAME = socket.gethostname()


# ---------------------------------------------------
# Paths
# ---------------------------------------------------

MODEL_DIR = "app/models"

MODEL_PATH = f"{MODEL_DIR}/churn_model.pkl"

os.makedirs(MODEL_DIR, exist_ok=True)


# ---------------------------------------------------
# MinIO Configuration
# ---------------------------------------------------

MINIO_ENDPOINT = os.getenv(
    "MINIO_ENDPOINT",
    "http://minio-service:9000"
)

MINIO_ACCESS_KEY = "minioadmin"

MINIO_SECRET_KEY = "minioadmin"

MINIO_BUCKET = "models"


# ---------------------------------------------------
# MinIO Client
# ---------------------------------------------------

s3_client = boto3.client(
    "s3",
    endpoint_url=MINIO_ENDPOINT,
    aws_access_key_id=MINIO_ACCESS_KEY,
    aws_secret_access_key=MINIO_SECRET_KEY,
    config=Config(signature_version="s3v4"),
    region_name="us-east-1"
)


# ---------------------------------------------------
# Download Model From MinIO
# ---------------------------------------------------

model = None

try:

    print("Downloading model from MinIO...")

    s3_client.download_file(
        MINIO_BUCKET,
        "churn_model.pkl",
        MODEL_PATH
    )

    print("Loading model...")

    model = joblib.load(MODEL_PATH)

    print("Model loaded successfully")

except Exception as e:

    print(f"Model loading failed: {e}")


# ---------------------------------------------------
# Prometheus Metrics
# ---------------------------------------------------

# Total Requests
REQUEST_COUNT = Counter(
    "prediction_requests_total",
    "Total prediction requests"
)

# Success Requests
PREDICTION_SUCCESS = Counter(
    "prediction_success_total",
    "Successful prediction requests"
)

# Failed Requests
PREDICTION_FAILURE = Counter(
    "prediction_failure_total",
    "Failed prediction requests"
)

# High Risk Predictions
HIGH_RISK_PREDICTIONS = Counter(
    "high_risk_predictions_total",
    "High churn risk predictions"
)

# Low Risk Predictions
LOW_RISK_PREDICTIONS = Counter(
    "low_risk_predictions_total",
    "Low churn risk predictions"
)

# API Latency
PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction API latency"
)

# Request Processing Time
REQUEST_PROCESSING_TIME = Summary(
    "request_processing_seconds",
    "Request processing time"
)

# Current Churn Probability
CHURN_PROBABILITY = Gauge(
    "prediction_churn_probability",
    "Current churn probability"
)

# Current Prediction Result
PREDICTION_RESULT = Gauge(
    "prediction_result",
    "Current prediction result"
)

# Current Model Status
MODEL_STATUS = Gauge(
    "model_loaded_status",
    "Model loaded status"
)

# API Health
API_HEALTH = Gauge(
    "api_health_status",
    "API health status"
)

# Active Requests
ACTIVE_REQUESTS = Gauge(
    "active_prediction_requests",
    "Currently active prediction requests"
)

# Last Prediction Latency
LAST_PREDICTION_LATENCY = Gauge(
    "last_prediction_latency_seconds",
    "Last prediction latency"
)

# Last Churn Probability
LAST_CHURN_PROBABILITY = Gauge(
    "last_churn_probability",
    "Last churn probability"
)

# Set Initial States
if model is not None:
    MODEL_STATUS.set(1)
else:
    MODEL_STATUS.set(0)

API_HEALTH.set(1)


# ---------------------------------------------------
# Request Schema
# ---------------------------------------------------

class CustomerData(BaseModel):

    gender: int

    SeniorCitizen: int

    Partner: int

    Dependents: int

    tenure: int

    PhoneService: int

    MonthlyCharges: float

    TotalCharges: float


# ---------------------------------------------------
# Root Endpoint
# ---------------------------------------------------

@app.get("/")
def home():

    return {
        "application": APP_NAME,
        "status": "running",
        "hostname": HOSTNAME
    }


# ---------------------------------------------------
# Health Check
# ---------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "hostname": HOSTNAME
    }


# ---------------------------------------------------
# Prediction Endpoint
# ---------------------------------------------------

@app.post("/predict")
@REQUEST_PROCESSING_TIME.time()
def predict(data: CustomerData):

    global model

    REQUEST_COUNT.inc()

    ACTIVE_REQUESTS.inc()

    start_time = time.time()

    try:

        # ---------------------------------------------------
        # Model Validation
        # ---------------------------------------------------

        if model is None:

            raise Exception("Model not loaded")


        # ---------------------------------------------------
        # Simulated Failure Condition
        # ---------------------------------------------------

        if data.MonthlyCharges < 0:

            raise Exception("Invalid MonthlyCharges value")


        # ---------------------------------------------------
        # Convert To DataFrame
        # ---------------------------------------------------

        input_df = pd.DataFrame([data.dict()])


        # ---------------------------------------------------
        # Prediction
        # ---------------------------------------------------

        prediction = model.predict(input_df)[0]

        probability = model.predict_proba(input_df)[0][1]


        # ---------------------------------------------------
        # Risk Classification
        # ---------------------------------------------------

        if probability >= 0.7:

            risk_level = "HIGH"

            HIGH_RISK_PREDICTIONS.inc()

        elif probability >= 0.4:

            risk_level = "MEDIUM"

        else:

            risk_level = "LOW"

            LOW_RISK_PREDICTIONS.inc()


        # ---------------------------------------------------
        # Latency
        # ---------------------------------------------------

        latency = time.time() - start_time


        # ---------------------------------------------------
        # Metrics
        # ---------------------------------------------------

        PREDICTION_SUCCESS.inc()

        PREDICTION_LATENCY.observe(latency)

        CHURN_PROBABILITY.set(float(probability))

        PREDICTION_RESULT.set(int(prediction))

        LAST_PREDICTION_LATENCY.set(latency)

        LAST_CHURN_PROBABILITY.set(float(probability))


        # ---------------------------------------------------
        # Response
        # ---------------------------------------------------

        return {
            "prediction": int(prediction),
            "churn_probability": round(float(probability), 4),
            "risk_level": risk_level,
            "latency_seconds": round(latency, 4),
            "hostname": HOSTNAME
        }


    except Exception as e:

        PREDICTION_FAILURE.inc()

        return JSONResponse(
            status_code=500,
            content={
                "error": str(e)
            }
        )

    finally:

        ACTIVE_REQUESTS.dec()


# ---------------------------------------------------
# Metrics Endpoint
# ---------------------------------------------------

@app.get("/metrics")
def metrics():

    return Response(
        generate_latest(),
        media_type="text/plain"
    )
