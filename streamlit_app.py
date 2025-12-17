import os
import streamlit as st
import pandas as pd
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

st.set_page_config(page_title="Student Risk Classification", layout="centered")

# =========================
# MLflow / DagsHub config
# =========================
if "MLFLOW_TRACKING_URI" in st.secrets:
    mlflow.set_tracking_uri(st.secrets["MLFLOW_TRACKING_URI"])
    os.environ["MLFLOW_TRACKING_URI"] = st.secrets["MLFLOW_TRACKING_URI"]

if "MLFLOW_TRACKING_USERNAME" in st.secrets and "MLFLOW_TRACKING_PASSWORD" in st.secrets:
    os.environ["MLFLOW_TRACKING_USERNAME"] = st.secrets["MLFLOW_TRACKING_USERNAME"]
    os.environ["MLFLOW_TRACKING_PASSWORD"] = st.secrets["MLFLOW_TRACKING_PASSWORD"]

st.caption(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")

RUN_ID = "430295b203584572848b7c8881d7e9aa"

def _find_model_dir_in_run(client: MlflowClient, run_id: str) -> str:
    """Find the directory containing an MLflow model (has an 'MLmodel' file)."""
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

    found = walk("")
    if not found:
        raise RuntimeError("No MLflow model found in artifacts (no 'MLmodel' file).")
    return found

@st.cache_resource
def load_model():
    try:
        client = MlflowClient()

        # Fails with "run not found" if tracking URI/auth is wrong
        run = client.get_run(RUN_ID)
        st.success(f"Connected. Experiment ID: {run.info.experiment_id}")

        model_dir = _find_model_dir_in_run(client, RUN_ID)
        model_uri = f"runs:/{RUN_ID}/{model_dir}"

        st.info(f"Detected model path: {model_dir}")
        st.info(f"Loading model from: {model_uri}")

        return mlflow.pyfunc.load_model(model_uri)

    except Exception as e:
        st.error("Model loading FAILED: your tracking URI/auth is not pointing to the server that contains this run.")
        st.exception(e)
        st.stop()

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

if st.button("🔍 Predict Risk"):
    try:
        pred = model.predict(input_data)
        prediction = pred[0] if hasattr(pred, "__len__") else pred

        st.subheader("Prediction Result")
        if int(prediction) == 1:
            st.error("⚠️ Student is **AT RISK**")
            st.metric("Risk Indicator", "High", "Needs Intervention")
        else:
            st.success("✅ Student is **NOT AT RISK**")
            st.metric("Risk Indicator", "Low", "On Track")

    except Exception as e:
        st.error("Prediction failed. Check model input schema / preprocessing.")
        st.exception(e)
