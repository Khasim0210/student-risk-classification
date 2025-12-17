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


# =========================
# Artifact utilities
# =========================
def list_artifacts_recursive(client: MlflowClient, run_id: str, path: str = ""):
    """Return list of dicts: {path, is_dir} for all artifacts under a run."""
    results = []
    for item in client.list_artifacts(run_id, path):
        results.append({"path": item.path, "is_dir": item.is_dir})
        if item.is_dir:
            results.extend(list_artifacts_recursive(client, run_id, item.path))
    return results


def find_mlflow_model_dirs(artifacts):
    """Find directories that contain an MLmodel file."""
    model_dirs = set()
    for a in artifacts:
        if (not a["is_dir"]) and a["path"].endswith("MLmodel"):
            model_dirs.add(os.path.dirname(a["path"]))
    return sorted(model_dirs)


def find_pickle_candidates(artifacts):
    """Find likely serialized models."""
    exts = (".pkl", ".pickle", ".joblib")
    return [a["path"] for a in artifacts if (not a["is_dir"]) and a["path"].lower().endswith(exts)]


class SimpleModelWrapper:
    """Wraps a non-MLflow-loaded model object to have a .predict(DataFrame) interface."""
    def __init__(self, obj):
        self.obj = obj

    def predict(self, df: pd.DataFrame):
        if hasattr(self.obj, "predict"):
            return self.obj.predict(df)
        raise TypeError("Loaded object has no .predict method.")


@st.cache_resource
def load_model():
    client = MlflowClient()

    # 1) Verify run exists (your earlier error was here)
    run = client.get_run(RUN_ID)
    st.success(f"Connected. Experiment ID: {run.info.experiment_id}")

    # 2) List artifacts and try to find an MLflow model directory
    artifacts = list_artifacts_recursive(client, RUN_ID, "")
    model_dirs = find_mlflow_model_dirs(artifacts)
    pickle_files = find_pickle_candidates(artifacts)

    # Show artifacts so you can see what's actually stored
    with st.expander("📦 Show run artifacts"):
        if not artifacts:
            st.write("No artifacts found in this run.")
        else:
            # Render as a simple table
            st.dataframe(pd.DataFrame(artifacts).sort_values(["is_dir", "path"], ascending=[False, True]),
                         use_container_width=True)

    # 3) Prefer MLflow model load if available
    if model_dirs:
        chosen_dir = model_dirs[0]  # pick first; usually only one
        if len(model_dirs) > 1:
            chosen_dir = st.selectbox("Multiple MLflow models found. Choose one:", model_dirs)

        model_uri = f"runs:/{RUN_ID}/{chosen_dir}"
        st.info(f"Loading MLflow model from: {model_uri}")
        return mlflow.pyfunc.load_model(model_uri)

    # 4) Fallback: load a .pkl / .joblib if present
    if pickle_files:
        chosen_file = pickle_files[0]
        if len(pickle_files) > 1:
            chosen_file = st.selectbox("No MLflow model found. Pick a pickle/joblib artifact to load:", pickle_files)

        st.warning("No MLflow 'MLmodel' file found. Falling back to loading a pickle/joblib artifact.")
        artifact_uri = f"runs:/{RUN_ID}/{chosen_file}"
        local_path = mlflow.artifacts.download_artifacts(artifact_uri=artifact_uri)

        # Try joblib first, then pickle
        obj = None
        try:
            import joblib
            obj = joblib.load(local_path)
        except Exception:
            import pickle
            with open(local_path, "rb") as f:
                obj = pickle.load(f)

        st.info(f"Loaded serialized model from: {chosen_file}")
        return SimpleModelWrapper(obj)

    # 5) If neither exists, the run simply doesn't contain a model artifact
    raise RuntimeError(
        "This run has no MLflow model (no MLmodel file) and no .pkl/.joblib artifacts. "
        "You need to log the model as an artifact in this run."
    )


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
