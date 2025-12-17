import os
from pathlib import Path

import streamlit as st
import pandas as pd
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

st.set_page_config(page_title="Student Risk Classification", layout="centered")

# =========================
# CONFIG
# =========================
RUN_ID = "430295b203584572848b7c8881d7e9aa"

# IMPORTANT:
# Put your model file in the repo as ONE of these paths.
# If your filename is different, add it here.
LOCAL_MODEL_CANDIDATES = [
    Path("model.joblib"),            # repo root
    Path("model.pkl"),               # repo root
    Path("models/model.joblib"),     # models/ folder
    Path("models/model.pkl"),        # models/ folder
]

# =========================
# MLflow config from secrets (optional)
# =========================
if "MLFLOW_TRACKING_URI" in st.secrets:
    mlflow.set_tracking_uri(st.secrets["MLFLOW_TRACKING_URI"])
    os.environ["MLFLOW_TRACKING_URI"] = st.secrets["MLFLOW_TRACKING_URI"]

if "MLFLOW_TRACKING_USERNAME" in st.secrets and "MLFLOW_TRACKING_PASSWORD" in st.secrets:
    os.environ["MLFLOW_TRACKING_USERNAME"] = st.secrets["MLFLOW_TRACKING_USERNAME"]
    os.environ["MLFLOW_TRACKING_PASSWORD"] = st.secrets["MLFLOW_TRACKING_PASSWORD"]

st.caption(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")

# =========================
# Helpers
# =========================
def _find_mlflow_model_dir(client: MlflowClient, run_id: str) -> str | None:
    """Find an artifact directory that contains an MLflow model (has an 'MLmodel' file)."""
    def walk(path: str):
        for item in client.list_artifacts(run_id, path):
            if item.is_dir:
                found = walk(item.path)
                if found:
                    return found
            else:
                if item.path.endswith("MLmodel"):
                    return os.path.dirname(item.path)
        return None

    return walk("")


def _load_local_model():
    # Debug: show what files exist in the deployed folder
    with st.expander("🔎 Debug: local model file check"):
        for p in LOCAL_MODEL_CANDIDATES:
            st.write(f"{p} -> {'✅ exists' if p.exists() else '❌ missing'}")
        st.write("Current directory files:", [x.name for x in Path(".").iterdir()])

    model_path = next((p for p in LOCAL_MODEL_CANDIDATES if p.exists()), None)
    if model_path is None:
        st.error(
            "No model found to load.\n\n"
            "✅ Your MLflow run exists, but it contains **no model artifacts**.\n\n"
            "To run this app WITHOUT MLflow logging, upload ONE of these files to your repo:\n"
            "- model.joblib (recommended)\n"
            "- model.pkl\n"
            "- models/model.joblib\n"
            "- models/model.pkl\n\n"
            "Make sure it is a REAL model file created by joblib/pickle, not an empty text file."
        )
        st.stop()

    try:
        import joblib
        model = joblib.load(model_path)
    except Exception:
        import pickle
        with open(model_path, "rb") as f:
            model = pickle.load(f)

    st.success(f"Loaded local model: {model_path}")
    return model


@st.cache_resource
def load_model():
    # Try MLflow first (only works if the run actually has a model artifact)
    try:
        client = MlflowClient()
        run = client.get_run(RUN_ID)
        st.success(f"Connected. Experiment ID: {run.info.experiment_id}")

        model_dir = _find_mlflow_model_dir(client, RUN_ID)
        if model_dir:
            model_uri = f"runs:/{RUN_ID}/{model_dir}"
            st.info(f"Loading MLflow model from: {model_uri}")
            return mlflow.pyfunc.load_model(model_uri)

        st.warning("This run has no MLflow model artifacts. Falling back to local model file...")

    except Exception as e:
        st.warning("MLflow not usable here (tracking/auth/run issue). Falling back to local model file...")
        st.caption(f"MLflow detail: {type(e).__name__}: {e}")

    # Fallback to local file
    return _load_local_model()


model = load_model()

# =========================
# UI
# =========================
st.title("🎓 Student Risk Classification")
st.write("Predict whether a student is **at risk** based on academic and demographic factors.")
st.markdown("---")

with st.sidebar:
    st.header("Student Input Features")
    age = st.number_input("Age", min_value=10, max_value=30, value=18)
    studytime = st.selectbox("Weekly Study Time (1: <2h, 4: >10h)", [1, 2, 3, 4])
    failures = st.selectbox("Past Class Failures", [0, 1, 2, 3], index=0)
    absences = st.number_input("Number of Absences", min_value=0, max_value=100, value=5)

    schoolsup = st.selectbox("School Support", ["yes", "no"], index=1)
    famsup = st.selectbox("Family Support", ["yes", "no"], index=0)
    activities = st.selectbox("Extra Curricular Activities", ["yes", "no"], index=0)

input_data = pd.DataFrame([{
    "age": age,
    "studytime": studytime,
    "failures": failures,
    "absences": absences,
    "schoolsup": schoolsup,
    "famsup": famsup,
    "activities": activities
}])

st.subheader("Input Preview")
st.dataframe(input_data, use_container_width=True)

if st.button("🔍 Predict Risk"):
    try:
        pred = model.predict(input_data)
        prediction = pred[0] if hasattr(pred, "__len__") else pred

        try:
            label = int(prediction)
        except Exception:
            label = 1 if str(prediction).strip().lower() in ("1", "yes", "true", "at risk") else 0

        st.subheader("Prediction Result")
        if label == 1:
            st.error("⚠️ Student is **AT RISK**")
            st.metric("Risk Indicator", "High", "Needs Intervention")
        else:
            st.success("✅ Student is **NOT AT RISK**")
            st.metric("Risk Indicator", "Low", "On Track")

    except Exception as e:
        st.error("Prediction failed. Your model may expect different feature names/types or preprocessing.")
        st.exception(e)
