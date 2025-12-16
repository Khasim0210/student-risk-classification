import streamlit as st
import pandas as pd
from pathlib import Path
import joblib

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Student Risk Classification", layout="centered")

# ----------------------------
# Model path (ROOT)
# ----------------------------
MODEL_PATH = Path("model.joblib")

# These MUST match the columns your pipeline was trained on (from SQL JOIN).
REQUIRED_COLUMNS = [
    "school", "sex", "age", "address", "famsize", "Pstatus",
    "Medu", "Fedu", "internet", "studytime", "failures", "absences", "subject"
]

# ----------------------------
# Load model (cached)
# ----------------------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "model.joblib not found in repo root.\n\n"
            "Expected structure:\n"
            "  your-repo/\n"
            "    streamlit_app.py\n"
            "    requirements.txt\n"
            "    model.joblib\n\n"
            "Fix: commit and push model.joblib to GitHub, then redeploy Streamlit Cloud."
        )
    return joblib.load(MODEL_PATH)

def make_single_input_df(**kwargs) -> pd.DataFrame:
    # Ensure correct column order
    return pd.DataFrame([{c: kwargs[c] for c in REQUIRED_COLUMNS}], columns=REQUIRED_COLUMNS)

def validate_and_prepare_uploaded_df(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "Uploaded CSV is missing required columns:\n"
            + ", ".join(missing)
            + "\n\nRequired columns:\n"
            + ", ".join(REQUIRED_COLUMNS)
        )

    X = df[REQUIRED_COLUMNS].copy()

    # Cast numeric columns safely
    num_cols = ["age", "Medu", "Fedu", "studytime", "failures", "absences"]
    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    if X[num_cols].isnull().any().any():
        bad = X[num_cols].isnull().sum()
        raise ValueError(
            "Some numeric columns have invalid/missing values. Fix your CSV.\n\n"
            + "\n".join([f"{k}: {int(v)} invalid" for k, v in bad.items() if v > 0])
        )

    # Keep ints
    for c in num_cols:
        X[c] = X[c].astype(int)

    return X

def predict_one(model, X: pd.DataFrame):
    pred = int(model.predict(X)[0])
    proba = None
    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(X)[:, 1][0])  # probability for class 1
    return pred, proba


# ----------------------------
# App UI
# ----------------------------
st.title("🎓 Student Risk Classification")
st.write("Predict whether a student is **AT RISK** (1) or **NOT AT RISK** (0).")
st.markdown("---")

# Load model with nice error
try:
    model = load_model()
except Exception as e:
    st.error(str(e))
    st.stop()

st.subheader("Single Student Prediction")

c1, c2 = st.columns(2)

with c1:
    school = st.selectbox("School", ["GP", "MS"], index=0)
    sex = st.selectbox("Sex", ["F", "M"], index=0)
    age = st.number_input("Age", min_value=10, max_value=30, value=18, step=1)
    address = st.selectbox("Address", ["U", "R"], index=0)  # Urban/Rural
    famsize = st.selectbox("Family Size", ["LE3", "GT3"], index=1)
    pstatus = st.selectbox("Pstatus (Parents Together/Apart)", ["T", "A"], index=0)

with c2:
    medu = st.selectbox("Mother Education (Medu 0-4)", [0, 1, 2, 3, 4], index=2)
    fedu = st.selectbox("Father Education (Fedu 0-4)", [0, 1, 2, 3, 4], index=2)
    internet = st.selectbox("Internet", ["yes", "no"], index=0)
    studytime = st.selectbox("Study Time (1-4)", [1, 2, 3, 4], index=1)
    failures = st.selectbox("Failures (0-3)", [0, 1, 2, 3], index=0)
    absences = st.number_input("Absences", min_value=0, max_value=100, value=5, step=1)
    subject = st.selectbox("Subject", ["math", "portuguese"], index=0)

X_single = make_single_input_df(
    school=school,
    sex=sex,
    age=int(age),
    address=address,
    famsize=famsize,
    Pstatus=pstatus,
    Medu=int(medu),
    Fedu=int(fedu),
    internet=internet,
    studytime=int(studytime),
    failures=int(failures),
    absences=int(absences),
    subject=subject,
)

if st.button("🔍 Predict Risk"):
    pred_label, pred_proba = predict_one(model, X_single)

    if pred_label == 1:
        st.error("⚠️ Prediction: **AT RISK**")
    else:
        st.success("✅ Prediction: **NOT AT RISK**")

    if pred_proba is not None:
        st.write(f"Risk probability (class=1): **{pred_proba:.3f}**")

    with st.expander("Show input"):
        st.dataframe(X_single)

st.markdown("---")
st.subheader("Batch Prediction via CSV Upload")

st.write("Upload a CSV with these columns:")
st.code(", ".join(REQUIRED_COLUMNS))

uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded is not None:
    try:
        df_up = pd.read_csv(uploaded)
        X_batch = validate_and_prepare_uploaded_df(df_up)

        preds = model.predict(X_batch)
        out = df_up.copy()
        out["pred_at_risk"] = preds

        if hasattr(model, "predict_proba"):
            out["pred_risk_proba"] = model.predict_proba(X_batch)[:, 1]

        st.success(f"✅ Predicted {len(out)} rows.")
        st.dataframe(out.head(50))

        st.download_button(
            "⬇️ Download predictions CSV",
            data=out.to_csv(index=False).encode("utf-8"),
            file_name="student_risk_predictions.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error(str(e))

st.caption("Model is loaded from repo root: model.joblib (no tokens, no MLflow).")
