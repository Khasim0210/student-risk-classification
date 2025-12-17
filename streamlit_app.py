import os
import streamlit as st
import pandas as pd
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

st.set_page_config(page_title="Student Risk Classification", layout="centered")

# =========================
# MLflow Configuration
# =========================
# Read from Streamlit Secrets -> environment variables (and set tracking uri explicitly)
if "MLFLOW_TRACKING_URI" in st.secrets:
    os.environ["MLFLOW_TRACKING_URI"] = st.secrets["MLFLOW_TRACKING_URI"]
    mlflow.set_tracking_uri(st.secrets["MLFLOW_TRACKING_URI"])

# Optional auth (DagsHub / basic auth)
if "MLFLOW_TRACKING_USERNAME" in st.secrets and "MLFLOW_TRACKING_PASSWORD" in st.secrets:
    os.environ["MLFLOW_TRACKING_USERNAME"] = st.secrets["MLFLOW_TRACKING_USERNAME"]
    os.environ["MLFLOW_TRACKING_PASSWORD"] = st.secrets["MLFLOW_TRACKING_PASSWORD"]

# Helpful debug (shows whether secrets are actually applied)
st.caption(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")

RUN_ID = "430295b203584572848b7c8881d7e9aa"

def _find_model_dir_in_run(client: MlflowClient, run_id: str) -> str:
    """
    Recursively searches the run's artifacts for an 'MLmodel' file
    and returns the directory containing it (the correct pyfunc model folder).
    """
    def walk(path: str):
        for item in client.list_artifacts(run_id, path):
            if item.is_dir:
                found = walk(item.path)
                if found:
                    return found
            else:
                # MLflow model directories contain an 'MLmodel' file
                if item.path.endswith("MLmodel"):
                    # directory containing MLmodel
                    return os.path.dirname(item.path)
        return None

    found_dir = walk("")
    if not found_dir:
        raise RuntimeError(
            "Could not find an MLflow model artifact (no 'MLmodel' file) under this run's artifacts."
        )
    return found_dir

@st.cache_resource
def load_model():
    st.info(f"Attempting to load model from run: {RUN_ID}")

    try:
        client = MlflowClient()

        # This will throw if run_id is not visible in the current tracking server
        _ = client.get_run(RUN_ID)

        # Auto-detect model folder inside artifacts
        model_dir = _find_model_dir_in_run(client, RUN_ID)
        model_uri = f"runs:/{RUN_ID}/{model_dir}"

        st.info(f"Detected model artifact path: {model_dir}")
        st.info(f"Loading model from: {model_uri}")

        model = mlflow.pyfunc.load_model(model_uri)
        return model

    except Exception as e:
        st.error("Model loading FAILED. This usually means the tracking URI/auth is wrong or the run ID is not in that server.")
        st.exception(e)
        st.stop()

# Load the model only once
model = load_model()

# =========================
# Streamlit UI
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

        # Handle common return types (array-like / scalar)
        prediction = pred[0] if hasattr(pred, "__len__") else pred

        st.subheader("Prediction Result")
        if int(prediction) == 1:
            st.error("⚠️ Student is **AT RISK**")
            st.metric(label="Risk Indicator", value="High", delta="Needs Intervention")
        else:
            st.success("✅ Student is **NOT AT RISK**")
            st.metric(label="Risk Indicator", value="Low", delta="On Track")

    except Exception as e:
        st.error("Prediction failed. Check model compatibility with input data / preprocessing.")
        st.exception(e)
