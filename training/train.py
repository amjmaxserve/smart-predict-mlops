import os

# Suppress Git warnings inside containers
os.environ["GIT_PYTHON_REFRESH"] = "quiet"

import joblib
import boto3
import mlflow
import mlflow.sklearn
import pandas as pd

from botocore.client import Config

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.preprocessing import LabelEncoder


# ---------------------------------------------------
# Paths
# ---------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATASET_PATH = os.path.join(
    BASE_DIR,
    "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "app/models"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "churn_model.pkl"
)

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
# Create Bucket If Not Exists
# ---------------------------------------------------

existing_buckets = s3_client.list_buckets()

bucket_names = [bucket['Name'] for bucket in existing_buckets['Buckets']]

if MINIO_BUCKET not in bucket_names:

    print(f"Creating bucket: {MINIO_BUCKET}")

    s3_client.create_bucket(Bucket=MINIO_BUCKET)

else:

    print(f"Bucket already exists: {MINIO_BUCKET}")


# ---------------------------------------------------
# Load Dataset
# ---------------------------------------------------

print("Loading dataset...")

df = pd.read_csv(DATASET_PATH)

print(f"Dataset Shape: {df.shape}")


# ---------------------------------------------------
# Data Cleaning
# ---------------------------------------------------

# Drop customerID
df.drop("customerID", axis=1, inplace=True)

# Convert TotalCharges
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Fill missing values
df.fillna(0, inplace=True)


# ---------------------------------------------------
# Encode Categorical Columns
# ---------------------------------------------------

print("Encoding categorical features...")

label_encoders = {}

for col in df.select_dtypes(include=["object"]).columns:

    le = LabelEncoder()

    df[col] = le.fit_transform(df[col])

    label_encoders[col] = le


# ---------------------------------------------------
# Features & Target
# ---------------------------------------------------

FEATURE_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MonthlyCharges",
    "TotalCharges"
]

X = df[FEATURE_COLUMNS]

#X = df.drop("Churn", axis=1)

y = df["Churn"]


# ---------------------------------------------------
# Train Test Split
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ---------------------------------------------------
# MLflow Configuration
# ---------------------------------------------------

print("Connecting to MLflow...")

mlflow.set_tracking_uri("http://mlflow:5000")

mlflow.set_experiment("customer-churn")


# ---------------------------------------------------
# Model Training
# ---------------------------------------------------

with mlflow.start_run():

    print("Training RandomForest model...")

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    print("Model training completed")


    # ---------------------------------------------------
    # Predictions
    # ---------------------------------------------------

    preds = model.predict(X_test)


    # ---------------------------------------------------
    # Metrics
    # ---------------------------------------------------

    accuracy = accuracy_score(y_test, preds)

    precision = precision_score(y_test, preds)

    recall = recall_score(y_test, preds)

    f1 = f1_score(y_test, preds)


    # ---------------------------------------------------
    # MLflow Logging
    # ---------------------------------------------------

    mlflow.log_param("model", "RandomForest")

    mlflow.log_param("n_estimators", 100)

    mlflow.log_metric("accuracy", accuracy)

    mlflow.log_metric("precision", precision)

    mlflow.log_metric("recall", recall)

    mlflow.log_metric("f1_score", f1)


    # ---------------------------------------------------
    # Save Model Locally
    # ---------------------------------------------------

    joblib.dump(model, MODEL_PATH)

    print(f"Model saved locally: {MODEL_PATH}")


    # ---------------------------------------------------
    # Upload Model to MinIO
    # ---------------------------------------------------

    print("Uploading model to MinIO...")

    s3_client.upload_file(
        MODEL_PATH,
        MINIO_BUCKET,
        "churn_model.pkl"
    )

    print("Model uploaded successfully")


    # ---------------------------------------------------
    # Log Artifact Path to MLflow
    # ---------------------------------------------------

    mlflow.log_param(
        "model_artifact",
        f"s3://{MINIO_BUCKET}/churn_model.pkl"
    )


    # ---------------------------------------------------
    # Final Metrics
    # ---------------------------------------------------

    print("\n========== MODEL METRICS ==========")

    print(f"Accuracy  : {accuracy:.4f}")

    print(f"Precision : {precision:.4f}")

    print(f"Recall    : {recall:.4f}")

    print(f"F1 Score  : {f1:.4f}")

    print("===================================\n")
