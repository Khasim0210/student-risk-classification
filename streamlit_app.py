# strealit_app.py  (Streamlit Cloud also works best if you name this file: streamlit_app.py)

import streamlit as st
import pandas as pd
from pathlib import Path

# Optional (only needed if you keep an MLflow-exported model folder in the repo)
try:
    import mlflow
except Exception:
    mlflow = None

# Optional (recommended if you save a .joblib model in the repo)
try:
    import joblib
except Exception:
    joblib = None

import pickle
import sqlite3


# ============================================================
# Model loading (GitHub-only: model lives inside your repo)
# ============================================================
# Put ONE of these in your GitHub repo:
#   A) MLflow exported model folder (recommended if you already used MLflow):
#        ./model/   (contains MLmodel + artifacts)
#      -> created via: mlflow models save -m "runs:/<run_id>/model" -d model
#
#   B) A plain sklearn pipeline pickle/joblib:
#        ./artifacts/model.joblib   (or .pkl)
#
# This app will try A first, then B.


@st.cache_resource
def load_model():
    repo_root = Path(__file__).resolve().parent

    # A) MLflow exported model folder in repo (no tracking server)
    mlflow_model_dir = repo_root / "model"
    if mlflow is not None and mlflow_model_dir.exists() and (mlflow_model_dir / "MLmodel").exists():
        return ("mlflow_pyfunc", mlflow.pyfunc.load_model(str(mlflow_model_dir)))

    # B) Joblib / pickle model file in repo
    candidates = [
        repo_root / "artifacts" / "model.joblib",
        repo_root / "artifacts" / "model.pkl",
        repo_root / "artifacts" / "model.pickle",
        repo_root / "model.joblib",
        repo_root / "model.pkl",
    ]
    for path in candidates:
        if path.exists():
            if path.suffix == ".joblib" and joblib is not None:
                return ("sklearn_like", joblib.load(path))
            # fallback to pickle
            with open(path, "rb") as f:
                return ("sklearn_like", pickle.load(f)

)

    raise FileNotFoundError(
        "Model not found in repo.\n"
        "Add either:\n"
        "  - an MLflow exported model folder at ./model (must contain MLmodel), OR\n"
        "  - a model file at ./artifacts/model.joblib (or .pkl)\n"
    )


def get_expected_columns(model_obj):
    """
    Best-effort: detect which columns the model expects.
    Works for:
      - sklearn estimators / pipelines with feature_names_in_
      - mlflow pyfunc models with a signature (when available)
    """
    # sklearn style
    if hasattr(model_obj, "feature_names_in_"):
        try:
            return list(model_obj.feature_names_in_)
        except Exception:
            pass

    # mlflow pyfunc style (signature)
    if hasattr(model_obj, "metadata"):
        try:
            sig = model_obj.metadata.signature
            if sig is not None and sig.inputs is not None:
                return [col.name for col in sig.inputs.inputs]
        except Exception:
            pass

    return None


# ============================================================
# Optional: local SQLite loader (only if you commit the DB file)
# ============================================================
def load_data_from_sqlite(db_path: Path) -> pd.DataFrame:
    conn = sqlite3.connect(str(db_path))
    query = """
    SELECT
        s.age,
        s.sex,
        s.address,
        f.Medu,
        f.Fedu,
        f.internet,
        a.studytime,
        a.failures,
        a.absences,
        a.subject
    FROM students s
    JOIN family_background f ON s.student_id = f.student_id
    JOIN academics a ON s.student_id = a.student_id
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


# ============================================================
# UI
# ============================================================
st.set_page_config(page_title="Student Risk Classification", layout="centered")

st.title("🎓 Student Risk Classification")
st.write("Predict whether a student is **at risk** based on academic and demographic factors.")
st.markdown("---")

model_type, model = load_model()
expected_cols = get_expected_columns(model)

# Sidebar: show model info + optional batch prediction
with st.sidebar:
    st.header("Model")
    st.write(f"Loaded from repo as: **{model_type}**")
    if expected_cols:
        st.caption("Expected input columns (detected):")
        st.code(", ".join(expected_cols), language="text")
    else:
        st.caption("Could not auto-detect expected columns (will try best-effort).")

    st.markdown("---")
    st.subheader("Batch prediction (optional)")

    # Option 1: CSV upload
    csv_file = st.file_uploader("Upload CSV", type=["csv"])
    if csv_file is not None:
        batch_df = pd.read_csv(csv_file)
        st.write("Preview:")
        st.dataframe(batch_df.head())

        if st.button("Predict on CSV"):
            X = batch_df.copy()
            if expected_cols:
                missing = [c for c in expected_cols if c not in X.columns]
                if missing:
                    st.error(f"CSV is missing required columns: {missing}")
                else:
                    X = X[expected_cols]
            preds = model.predict(X)
            out = batch_df.copy()
            out["prediction"] = preds
            st.success("Done!")
            st.dataframe(out.head(50))

    # Option 2: SQLite (only if DB exists in repo)
    repo_root = Path(__file__).resolve().parent
    db_path = repo_root / "database" / "students.db"
    if db_path.exists():
        if st.button("Predict from local SQLite DB"):
            df_db = load_data_from_sqlite(db_path)
            X = df_db.copy()
            if expected_cols:
                missing = [c for c in expected_cols if c not in X.columns]
                if missing:
                    st.error(f"DB data is missing required columns: {missing}")
                else:
                    X = X[expected_cols]
            preds = model.predict(X)
            df_out = df_db.copy()
            df_out["prediction"] = preds
            st.success("Done!")
            st.dataframe(df_out.head(50))
    else:
        st.caption("SQLite DB not found at ./database/students.db (optional).")

# ============================================================
# Manual input form (covers BOTH of your feature sets)
# ============================================================
st.subheader("Manual input")

# Common fields from both of your scripts (we include all; we’ll only pass what the model needs)
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=10, max_value=30, value=18)
    sex = st.selectbox("Sex", ["F", "M"])
    address = st.selectbox("Address", ["U", "R"])  # U=Urban, R=Rural (common in student datasets)

    Medu = st.selectbox("Mother's Education (Medu)", [0, 1, 2, 3, 4])
    Fedu = st.selectbox("Father's Education (Fedu)", [0, 1, 2, 3, 4])

with col2:
    internet = st.selectbox("Internet", ["yes", "no"])
    studytime = st.selectbox("Weekly Study Time", [1, 2, 3, 4])
    failures = st.selectbox("Past Class Failures", [0, 1, 2, 3])
    absences = st.number_input("Number of Absences", min_value=0, max_value=100, value=5)

    # These are in your Streamlit version
    schoolsup = st.selectbox("School Support", ["yes", "no"])
    famsup = st.selectbox("Family Support", ["yes", "no"])
    activities = st.selectbox("Extra Curricular Activities", ["yes", "no"])

# In your sqlite app you also had subject
subject = st.selectbox("Subject", ["math", "por", "other"])

# Build full input row
input_row = {
    "age": age,
    "sex": sex,
    "address": address,
    "Medu": Medu,
    "Fedu": Fedu,
    "internet": internet,
    "studytime": studytime,
    "failures": failures,
    "absences": absences,
    "schoolsup": schoolsup,
    "famsup": famsup,
    "activities": activities,
    "subject": subject,
}
input_df = pd.DataFrame([input_row])

# If we detected expected columns, strictly match them
X_manual = input_df.copy()
if expected_cols:
    missing = [c for c in expected_cols if c not in X_manual.columns]
    extra = [c for c in X_manual.columns if c not in expected_cols]
    if missing:
        st.warning(f"Model expects columns not provided by UI: {missing}")
    if extra:
        # keep only what the model expects
        X_manual = X_manual[[c for c in expected_cols if c in X_manual.columns]]

st.markdown("---")

if st.button("🔍 Predict Risk"):
    try:
        pred = model.predict(X_manual)[0]
        # If your model outputs {0,1}
        if int(pred) == 1:
            st.error("⚠️ Student is **AT RISK**")
        else:
            st.success("✅ Student is **NOT AT RISK**")

        # Optional: show what we sent to the model (helps debugging mismatched columns)
        with st.expander("Show model input payload"):
            st.dataframe(X_manual)

    except Exception as e:
        st.error("Prediction failed. This usually means the model expects different columns or preprocessing.")
        st.exception(e)
        st.info("Tip: make sure the model you committed to GitHub includes the preprocessing (e.g., a sklearn Pipeline).")
