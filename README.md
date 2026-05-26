# SmartPredict MLOps Platform Documentation

## Project Overview

SmartPredict is a production-style MLOps platform built using:

* FastAPI
* MLflow
* MinIO
* Prometheus
* Grafana
* Docker Compose
* Scikit-learn

The platform performs real-time customer churn prediction while implementing:

* model lifecycle management
* object storage
* observability
* metrics collection
* inference monitoring
* operational dashboards

---

# Final Architecture

```text
                    ┌────────────────────┐
                    │   User Requests    │
                    └─────────┬──────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │      FastAPI        │
                   │  Inference Service  │
                   └─────────┬───────────┘
                             │
            ┌────────────────┼────────────────┐
            │                │                │
            ▼                ▼                ▼
   ┌────────────────┐ ┌──────────────┐ ┌───────────────┐
   │   Prometheus   │ │    MinIO     │ │    MLflow     │
   │ Metrics Engine │ │ Object Store │ │ Experiment Log│
   └────────────────┘ └──────────────┘ └───────────────┘
            │
            ▼
    ┌────────────────┐
    │    Grafana     │
    │ Visualization  │
    └────────────────┘
```

---

# Technology Stack

| Component              | Purpose                       |
| ---------------------- | ----------------------------- |
| FastAPI                | Real-time inference API       |
| MLflow                 | Experiment tracking           |
| MinIO                  | S3-compatible object storage  |
| Prometheus             | Metrics collection            |
| Grafana                | Visualization dashboards      |
| Docker Compose         | Multi-container orchestration |
| Scikit-learn           | Machine learning model        |
| RandomForestClassifier | Churn prediction algorithm    |

---

# Project Structure

```text
smartpredict-mlops/
│
├── app/
│   ├── main.py
│   └── models/
│       └── churn_model.pkl
│
├── training/
│   └── train.py
│
├── data/
│   └── raw/
│       └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── monitoring/
│   └── prometheus/
│       └── prometheus.yml
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── traffic-generator.sh
```

---

# Dataset

Dataset Used:

Telco Customer Churn Dataset

Source:
[https://www.kaggle.com/datasets/blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

# Dataset Download

## Install Kaggle CLI

```bash
pip install kaggle
```

## Configure Kaggle API

```bash
mkdir -p ~/.kaggle
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

## Download Dataset

```bash
mkdir -p data/raw

kaggle datasets download \
-d blastchar/telco-customer-churn \
-p data/raw
```

## Extract Dataset

```bash
cd data/raw
unzip telco-customer-churn.zip
```

---

# Docker Compose Services

## Services Running

| Service       | Port |
| ------------- | ---- |
| FastAPI       | 8000 |
| MLflow        | 5000 |
| Prometheus    | 9090 |
| Grafana       | 3000 |
| MinIO API     | 9000 |
| MinIO Console | 9001 |

---

# Docker Compose Startup

## Build Stack

```bash
docker compose build --no-cache
```

## Start Stack

```bash
docker compose up -d
```

## Verify Containers

```bash
docker ps
```

---

# Model Training Workflow

## Training Process

1. Load dataset
2. Clean dataset
3. Encode categorical features
4. Split train/test data
5. Train RandomForest model
6. Evaluate metrics
7. Save model locally
8. Upload model to MinIO
9. Log metadata to MLflow

---

# Training Command

```bash
docker compose exec fastapi python training/train.py
```

---

# MinIO Integration

## MinIO Console

```text
http://YOUR_VM_IP:9001
```

## Default Credentials

```text
Username: minioadmin
Password: minioadmin
```

## Bucket Created

```text
models
```

## Uploaded Artifact

```text
churn_model.pkl
```

---

# FastAPI Endpoints

## Swagger UI

```text
http://YOUR_VM_IP:8000/docs
```

## Root Endpoint

```text
GET /
```

## Health Check

```text
GET /health
```

## Prediction Endpoint

```text
POST /predict
```

## Metrics Endpoint

```text
GET /metrics
```

---

# Prediction Payload

```json
{
  "gender": 1,
  "SeniorCitizen": 0,
  "Partner": 1,
  "Dependents": 0,
  "tenure": 24,
  "PhoneService": 1,
  "MonthlyCharges": 95,
  "TotalCharges": 2300
}
```

---

# Example Prediction Response

```json
{
  "prediction": 0,
  "churn_probability": 0.32,
  "risk_level": "LOW",
  "latency_seconds": 0.011,
  "hostname": "smartpredict-api"
}
```

---

# Prometheus Metrics Implemented

## API Metrics

* prediction_requests_total
* prediction_success_total
* prediction_failure_total
* active_prediction_requests

## ML Metrics

* high_risk_predictions_total
* low_risk_predictions_total
* prediction_churn_probability
* prediction_result

## Performance Metrics

* prediction_latency_seconds
* request_processing_seconds
* last_prediction_latency_seconds

## Health Metrics

* model_loaded_status
* api_health_status

---

# Grafana Dashboard

## Dashboard Sections

### Traffic Metrics

* Total Traffic
* Request Rate
* Active Requests

### Prediction Metrics

* Successful Predictions
* Failed Predictions
* Failure Rate

### ML Metrics

* High Risk Customers
* Low Risk Customers
* Churn Probability

### Performance Metrics

* API Latency
* P95 Latency
* Request Processing Time

### Health Metrics

* API Health
* Model Status

---

# Important Grafana Queries

## Total Requests

```promql
prediction_requests_total
```

## Successful Predictions

```promql
prediction_success_total
```

## Failed Predictions

```promql
prediction_failure_total
```

## Request Rate

```promql
rate(prediction_requests_total[1m])
```

## Average Latency

```promql
rate(prediction_latency_seconds_sum[1m])
/
rate(prediction_latency_seconds_count[1m])
```

## Failure Rate Percentage

```promql
(
prediction_failure_total
/
prediction_requests_total
) * 100
```

## High Risk Customers

```promql
high_risk_predictions_total
```

## Low Risk Customers

```promql
low_risk_predictions_total
```

## Active Requests

```promql
active_prediction_requests
```

## Current Churn Probability

```promql
last_churn_probability
```

## Model Status

```promql
model_loaded_status
```

---

# Traffic Generation Scripts

## Low Risk Traffic

```bash
for i in {1..50}
do

curl -s -X POST http://localhost:8000/predict \
-H "Content-Type: application/json" \
-d '{
  "gender":0,
  "SeniorCitizen":0,
  "Partner":1,
  "Dependents":1,
  "tenure":72,
  "PhoneService":1,
  "MonthlyCharges":30,
  "TotalCharges":5000
}' > /dev/null

done
```

---

## High Risk Traffic

```bash
for i in {1..50}
do

curl -s -X POST http://localhost:8000/predict \
-H "Content-Type: application/json" \
-d '{
  "gender":1,
  "SeniorCitizen":1,
  "Partner":0,
  "Dependents":0,
  "tenure":1,
  "PhoneService":1,
  "MonthlyCharges":120,
  "TotalCharges":120
}' > /dev/null

done
```

---

## Failure Traffic

```bash
for i in {1..100}
do

curl -s -X POST http://localhost:8000/predict \
-H "Content-Type: application/json" \
-d '{
  "gender":1,
  "SeniorCitizen":0,
  "Partner":1,
  "Dependents":0,
  "tenure":24,
  "PhoneService":1,
  "MonthlyCharges":-95,
  "TotalCharges":2300
}' > /dev/null

done
```

---

# Key MLOps Concepts Demonstrated

## Infrastructure

* Containerization
* Multi-service orchestration
* Object storage
* Monitoring stack

## ML Lifecycle

* Model training
* Experiment tracking
* Artifact storage
* Model loading
* Inference serving

## Observability

* API metrics
* Latency monitoring
* Failure monitoring
* ML telemetry
* Operational dashboards

## Reliability Engineering

* Failure simulation
* Request monitoring
* Performance analysis
* Health checks

---

# Real-World Engineering Problems Solved

## Training-Serving Skew

Resolved feature mismatch between:

* training pipeline
* inference pipeline

---

## Artifact Lifecycle Management

Implemented:

* model upload
* model retrieval
* object storage lifecycle

using MinIO.

---

## MLflow Version Compatibility

Handled MLflow client/server API mismatch.

---

## Inference Observability

Implemented:

* success metrics
* failure metrics
* latency histograms
* operational monitoring

---

# Resume Summary

Designed and implemented a production-style MLOps platform for real-time customer churn prediction using FastAPI, MLflow, MinIO, Prometheus, Grafana, Docker, and Scikit-learn. Built complete ML lifecycle workflows including model training, experiment tracking, object storage integration, real-time inference APIs, operational monitoring, latency tracking, and observability dashboards.

---

# Future Improvements

## Planned Enhancements

* Kubernetes deployment
* Helm charts
* GitHub Actions CI/CD
* PostgreSQL backend for MLflow
* Node Exporter
* cAdvisor
* Drift monitoring using Evidently AI
* Automated retraining pipeline
* Model versioning
* Canary deployments
* Horizontal scaling
* Alerting rules

---

# Final Outcome

This project demonstrates:

* MLOps Engineering
* DevOps + AI Integration
* ML Platform Engineering
* Observability Engineering
* Production Inference Systems
* Containerized AI Infrastructure

The platform simulates realistic enterprise ML operations with complete monitoring and operational telemetry.

