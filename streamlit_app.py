import os
import streamlit as st
import mlflow
import dagshub
import pandas as pd

# Must be the first Streamlit command
st.set_page_config(page_title="Student Risk Classification", layout="centered")

# --- Auth / config (NO token in code) ---
# Works on Streamlit Cloud if you set secrets, and locally if you set env var
DAGSHUB_USER_TOKEN = os.getenv("DAGSHUB_USER_TOKEN", None)

# If running on Streamlit Cloud, you can also use st.secrets
if DAGSHUB_USER_TOKEN is None and "DAGSHUB_USER_TOKEN" in st.secrets:
    DAGSHUB_USER_TOKEN = st.secrets["DAGSHUB_USER_TOKEN"]

if not DAGSHUB_USER_TOKEN:
    st.error("DAGSHUB_USER_TOKEN is missing. Set it in environment variables or Streamlit secrets.")
    st.stop()

dagshub.auth.add_app_token(DAGSHUB_USER_TOKEN)

mlflow.set_tracking_uri("https://dagshub.com/Khasim0210/student-risk-classification.mlflow")


@st.cache_resource
def load_model():
    # NOTE: "latest" may fail depending on your MLflow registry.
    # Replace with Production or a version number if needed.
    return mlflow.pyfunc.load_model("models:/Logistic_Regression_Baseline/Production")


model = load_model()

# =========================
# UI
# =========================
st.title("🎓 Student Risk Classification")
st.write("Predict whether a student is **at risk** based on academic and demographic factors.")
st.markdown("---")

age = st.number_input("Age", min_value=10, max_value=30, value=18)
studytime = st.selectbox("Weekly Study Time", [1, 2, 3, 4])
failures = st.selectbox("Past Class Failures", [0, 1, 2, 3])
absences = st.number_input("Number of Absences", min_value=0, max_value=100, value=5)

schoolsup = st.selectbox("School Support", ["yes", "no"])
famsup = st.selectbox("Family Support", ["yes", "no"])
activities = st.selectbox("Extra Curricular Activities", ["yes", "no"])

# If your model expects numeric encoding, uncomment this:
# yn = {"yes": 1, "no": 0}

input_data = pd.DataFrame([{
    "age": age,
    "studytime": studytime,
    "failures": failures,
    "absences": absences,
    "schoolsup": schoolsup,   # or yn[schoolsup]
    "famsup": famsup,         # or yn[famsup]
    "activities": activities  # or yn[activities]
}])

if st.button("🔍 Predict Risk"):
    pred = model.predict(input_data)[0]
    if int(pred) == 1:
        st.error("⚠️ Student is **AT RISK**")
    else:
        st.success("✅ Student is **NOT AT RISK**")