import os
from pathlib import Path

import streamlit as st
import pandas as pd
import joblib

import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

from dagshub.common.api.repo import RepoAPI  # requires `dagshub` in requirements.txt

st.set_page_config(page_title="Student Risk Classification", layout="centered")

# =========================
# CONFIG
# =========================
RUN_ID = "430295b203584572848b7c8881d7e9aa"
DAGSHUB_REPO = "Khasim0210/student-risk-classification"
MODEL_PATH_IN_REPO = "model.joblib"  # must match the file path in your repo

# Download target (safe writable location on Streamlit Cloud)
DOWNLOADED_MODEL_PATH = Path("/tmp/model.joblib")

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
def is_git_lfs_pointer_bytes(b: bytes) -> bool:
    return b"version https://git-lfs.github.com/spec/v1" in b[:200]

def download_real_model_from_dagshub(dest: Path) -> None:
    user = st.secrets.get("MLFLOW_TRACKING_USERNAME")
    token = st.secrets.get("MLFLOW_TRACKING_PASSWORD")
    if not user or not token:
        st.error("Missing secrets: MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD (use your DagsHub token).")
        st.stop()

    st.info("Downloading the real model from DagsHub (resolving Git LFS pointer)...")
    api = RepoAPI(repo=DAGSHUB_REPO, auth=(user, token))
    data = api.get_file(MODEL_PATH_IN_REPO)  # returns bytes :contentReference[oaicite:0]{index=0}

    # If API still returns an LFS pointer, we can't proceed
    if is_git_lfs_pointer_bytes(data):
        st.error(
            "DagsHub API returned a Git LFS pointer instead of the real model bytes.\n"
            "This means the real LFS object is not accessible to this environment.\n"
            "Fix: re-upload the model without LFS, or store the model in DagsHub Storage/DVC and download that file."
        )
        st.stop()

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    st.success(f"Model downloaded to {dest} ({dest.stat().st_size} bytes)")

def _find_mlflow_model_dir(client: MlflowClient, run_id: str) -> str | None:
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

@st.cache_resource
def load_model():
    # 1) Try MLflow model (your run currently has none; this will fall back)
    try:
        client = MlflowClient()
        run = client.get_run(RUN_ID)
        st.success(f"Connected. Experiment ID: {run.info.experiment_id}")

        model_dir = _find_mlflow_model_dir(client, RUN_ID)
        if model_dir:
            model_uri = f"runs:/{RUN_ID}/{model_dir}"
            st.info(f"Loading MLflow model from: {model_uri}")
            return mlflow.pyfunc.load_model(model_uri)

        st.warning("This run has no MLflow model artifacts. Falling back to local model...")
    except Exception as e:
        st.warning("MLflow not usable. Falling back to local model...")
        st.caption(f"MLflow detail: {type(e).__name__}: {e}")

    # 2) Local file in repo (may be LFS pointer)
    repo_model_path = Path("model.joblib")
    if not repo_model_path.exists():
        st.error("model.joblib not found in the repo folder.")
        st.stop()

    head = repo_model_path.read_bytes()[:200]
    if is_git_lfs_pointer_bytes(head):
        # Download real bytes to /tmp and load from there
        download_real_model_from_dagshub(DOWNLOADED_MODEL_PATH)
        return joblib.load(DOWNLOADED_MODEL_PATH)

    # If not a pointer, try loading directly
    return joblib.load(repo_model_path)

# =========================
# Load model
# =========================
try:
    model = load_model()
except Exception as e:
    st.error("Model loading FAILED.")
    st.exception(e)
    st.stop()

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
        label = int(prediction)

        st.subheader("Prediction Result")
        if label == 1:
            st.error("⚠️ Student is **AT RISK**")
            st.metric("Risk Indicator", "High", "Needs Intervention")
        else:
            st.success("✅ Student is **NOT AT RISK**")
            st.metric("Risk Indicator", "Low", "On Track")
    except Exception as e:
        st.error("Prediction failed. Model may expect different preprocessing/feature types.")
        st.exception(e)
