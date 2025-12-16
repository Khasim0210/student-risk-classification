# streamlit_app.py
# GitHub + Streamlit Cloud friendly (no tokens, no DagsHub, no MLflow).
# Expects a saved sklearn Pipeline at: models/model.joblib

import streamlit as st
import pandas as pd
from pathlib import Path
import joblib

# ----------------------------
# Config
# ----------------------------
st.set_page_config(
    page_title="Student Risk Classification",
    layout="centered",
)

MODEL_PATH = Path("models/model.joblib")

REQUIRED_COLUMNS = [
    "school", "sex", "age", "address", "famsize", "Pstatus",
    "Medu", "Fedu", "internet", "studytime", "failures", "absences", "subject"
]

# ----------------------------
# Model loader
# ----------------------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at: {MODEL_PATH}\n\n"
            "Fix:\n"
            "1) Create a folder named 'models' in your repo\n"
            "2) Save your trained pipeline as 'models/model.joblib'\n"
            "   Example (in notebook):\n"
            "   import joblib\n"
            "   joblib.dump(pipeline, 'models/model.joblib')\n"
            "3) Commit & push to GitHub, then redeploy Streamlit Cloud"
        )
    return joblib.load(MODEL_PATH)

# ----------------------------
# Helpers
# ----------------------------
def make_single_input_df(
    school, sex, age, address, famsize, pstatus, medu, fedu,
    internet, studytime, failures, absences, subject
) -> pd.DataFrame:
    row = {
        "school": school,
        "sex": sex,
        "age": int(age),
        "address": address,
        "famsize": famsize,
        "Pstatus": pstatus,
        "Medu": int(medu),
        "Fedu": int(fedu),
        "internet": internet,
        "studytime": int(studytime),
        "failures": int(failures),
        "absences": int(absences),
        "subject": subject
    }
    return pd.DataFrame([row], columns=REQUIRED_COLUMNS)

def predict(model, X: pd.DataFrame):
    # Returns: pred_label (0/1), pred_proba (float or None)
    pred = model.predict(X)
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]  # probability of class "1"
    return int(pred[0]), (float(proba[0]) if proba is not None else None)

def validate_and_prepare_uploaded_df(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "Uploaded CSV is missing required columns:\n"
            + ", ".join(missing)
            + "\n\nRequired columns:\n"
            + ", ".join(REQUIRED_COLUMNS)
        )

    # Keep only required columns in correct order
    X = df[REQUIRED_COLUMNS].copy()

    # Type casting (safe)
    for col in ["age", "Medu", "Fedu", "studytime", "failures", "absences"]:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    if X[["age", "Medu", "Fedu", "studytime", "failures", "absences"]].isnull().any().any():
        raise ValueError(
            "Some numeric columns contain non-numeric or missing values. "
            "Please fix your CSV and try again."
        )

    # Cast to int for neatness
    X["age"] = X["age"].astype(int)
    X["Medu"] = X["Medu"].astype(int)
    X["Fedu"] = X["Fedu"].astype(int)
    X["studytime"] = X["studytime"].astype(int)
    X["failures"] = X["failures"].astype(int)
    X["absences"] = X["absences"].astype(int)

    return X

# ----------------------------
# UI
# ----------------------------
st.title("🎓 Student Risk Classification")
st.write(
    "Predict whether a student is **AT RISK** (1) or **NOT AT RISK** (0) "
    "based on academic + demographic factors."
)

# Load model (with friendly error)
try:
    model = load_model()
except Exception as e:
    st.error(str(e))
    st.stop()

st.markdown("---")
st.subheader("Single Student Prediction")

col1, col2 = st.columns(2)

with col1:
    school = st.selectbox("School", ["GP", "MS"], index=0)
    sex = st.selectbox("Sex", ["F", "M"], index=0)
    age = st.number_input("Age", min_value=10, max_value=30, value=18, step=1)
    address = st.selectbox("Address", ["U", "R"], index=0)  # Urban/Rural
    famsize = st.selectbox("Family Size", ["LE3", "GT3"], index=1)  # <=3 / >3
    pstatus = st.selectbox("Parents Cohabitation Status (Pstatus)", ["T", "A"], index=0)  # Together/Apart

with col2:
    medu = st.selectbox("Mother Education (Medu: 0-4)", [0, 1, 2, 3, 4], index=2)
    fedu = st.selectbox("Father Education (Fedu: 0-4)", [0, 1, 2, 3, 4], index=2)
    internet = st.selectbox("Internet Access", ["yes", "no"], index=0)
    studytime = st.selectbox("Weekly Study Time (1-4)", [1, 2, 3, 4], index=1)
    failures = st.selectbox("Past Failures (0-3)", [0, 1, 2, 3], index=0)
    absences = st.number_input("Absences", min_value=0, max_value=100, value=5, step=1)
    subject = st.selectbox("Subject", ["math", "portuguese"], index=0)

X_single = make_single_input_df(
    school, sex, age, address, famsize, pstatus, medu, fedu,
    internet, studytime, failures, absences, subject
)

if st.button("🔍 Predict Risk"):
    pred_label, pred_proba = predict(model, X_single)

    if pred_label == 1:
        st.error("⚠️ Prediction: **AT RISK**")
    else:
        st.success("✅ Prediction: **NOT AT RISK**")

    if pred_proba is not None:
        st.write(f"Risk probability (class=1): **{pred_proba:.3f}**")

    with st.expander("Show input used for prediction"):
        st.dataframe(X_single)

st.markdown("---")
st.subheader("Batch Prediction via CSV Upload")

st.write(
    "Upload a CSV containing these columns:\n\n"
    + ", ".join(REQUIRED_COLUMNS)
)

uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded is not None:
    try:
        df_up = pd.read_csv(uploaded)
        X_batch = validate_and_prepare_uploaded_df(df_up)

        preds = model.predict(X_batch)
        out = df_up.copy()
        out["pred_at_risk"] = preds

        # Add probability if available
        if hasattr(model, "predict_proba"):
            out["pred_risk_proba"] = model.predict_proba(X_batch)[:, 1]

        st.success(f"✅ Completed predictions for {len(out)} rows.")
        st.dataframe(out.head(50))

        csv_bytes = out.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download predictions as CSV",
            data=csv_bytes,
            file_name="student_risk_predictions.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(str(e))

st.markdown("---")
st.caption(
    "Deployment note: This app loads the model from your GitHub repo (models/model.joblib). "
    "Do NOT hardcode secrets/tokens in public repos."
)
