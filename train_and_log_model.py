import os
from pathlib import Path

import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


# -------------------------
# DagsHub / MLflow settings
# -------------------------
TRACKING_URI = "https://dagshub.com/Khasim0210/student-risk-classification.mlflow"
mlflow.set_tracking_uri(TRACKING_URI)

# Use env vars for token locally; in Streamlit Cloud use secrets.toml
# export MLFLOW_TRACKING_USERNAME="Khasim0210"
# export MLFLOW_TRACKING_PASSWORD="<YOUR_DAGSHUB_TOKEN>"
if not os.getenv("MLFLOW_TRACKING_USERNAME") or not os.getenv("MLFLOW_TRACKING_PASSWORD"):
    raise RuntimeError(
        "Missing MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD env vars.\n"
        "Set them to your DagsHub username and access token."
    )

mlflow.set_experiment("student-risk-classification")

# -------------------------
# Load dataset
# -------------------------
candidate_paths = [
    "data/student.csv",
    "data/students.csv",
    "student.csv",
    "students.csv",
    "data/student-mat.csv",
    "student-mat.csv",
    "data/student-por.csv",
    "student-por.csv",
]

data_path = None
for p in candidate_paths:
    if Path(p).exists():
        data_path = Path(p)
        break

if data_path is None:
    raise FileNotFoundError(
        "Could not find a dataset CSV. Put your dataset in one of these paths:\n"
        + "\n".join(candidate_paths)
    )

df = pd.read_csv(data_path)

# -------------------------
# Define features / target
# -------------------------
FEATURES = ["age", "studytime", "failures", "absences", "schoolsup", "famsup", "activities"]

missing = [c for c in FEATURES if c not in df.columns]
if missing:
    raise ValueError(f"Dataset is missing required feature columns: {missing}")

# Target detection
if "at_risk" in df.columns:
    target_col = "at_risk"
    y = df[target_col].astype(int)
elif "risk" in df.columns:
    target_col = "risk"
    y = df[target_col].astype(int)
elif "G3" in df.columns:
    target_col = "G3"
    # Example rule: at risk if final grade < 10
    y = (df["G3"] < 10).astype(int)
else:
    raise ValueError("No target found. Add an 'at_risk' or 'risk' column, or include 'G3'.")

X = df[FEATURES].copy()

# -------------------------
# Train/test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None
)

# -------------------------
# Preprocess + Model
# -------------------------
num_features = ["age", "studytime", "failures", "absences"]
cat_features = ["schoolsup", "famsup", "activities"]

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
    ]
)

clf = LogisticRegression(max_iter=2000)

pipeline = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("model", clf),
    ]
)

# -------------------------
# Fit + evaluate
# -------------------------
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, zero_division=0)

# ROC-AUC if possible (needs predict_proba and at least 2 classes)
auc = None
if hasattr(pipeline, "predict_proba") and y_test.nunique() > 1:
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)

# -------------------------
# Log to MLflow (THIS is what your run was missing)
# -------------------------
with mlflow.start_run(run_name="student-risk-model") as run:
    mlflow.log_param("data_path", str(data_path))
    mlflow.log_param("target", target_col)
    mlflow.log_param("model_type", "LogisticRegression")

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1", f1)
    if auc is not None:
        mlflow.log_metric("roc_auc", auc)

    signature = infer_signature(X_train, pipeline.predict(X_train))

    mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path="student_risk_model",  # <-- Streamlit will load this
        signature=signature,
        input_example=X_train.head(3),
    )

    # Write run id to a file so Streamlit can load it automatically
    run_id_file = Path("model_run_id.txt")
    run_id_file.write_text(run.info.run_id, encoding="utf-8")

    # Optional: also log that file as artifact
    mlflow.log_artifact(str(run_id_file), artifact_path="meta")

    print("✅ Logged MLflow model.")
    print("RUN_ID =", run.info.run_id)
