import streamlit as st
import mlflow.pyfunc
import os
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Student Risk Classification", layout="centered")

# --- Load model from local repo folder ---
@st.cache_resource
def load_model():
    model_dir = Path(__file__).parent / "model"  # ./model in repo
    return mlflow.pyfunc.load_model(str(model_dir))

model = load_model()

# =========================
# Streamlit UI
# =========================
st.title("🎓 Student Risk Classification")
st.write("Predict whether a student is **at risk** based on academic and demographic factors.")
st.markdown("---")

# =========================
# User Input Section
# =========================
age = st.number_input("Age", min_value=10, max_value=30, value=18)
studytime = st.selectbox("Weekly Study Time", [1, 2, 3, 4])
failures = st.selectbox("Past Class Failures", [0, 1, 2, 3])
absences = st.number_input("Number of Absences", min_value=0, max_value=100, value=5)

schoolsup = st.selectbox("School Support", ["yes", "no"])
famsup = st.selectbox("Family Support", ["yes", "no"])
activities = st.selectbox("Extra Curricular Activities", ["yes", "no"])

# =========================
# Prepare input data
# =========================
input_data = pd.DataFrame([{
    "age": age,
    "studytime": studytime,
    "failures": failures,
    "absences": absences,
    "schoolsup": schoolsup,
    "famsup": famsup,
    "activities": activities
}])

# =========================
# Prediction
# =========================
if st.button("🔍 Predict Risk"):
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("⚠️ Student is **AT RISK**")
    else:
        st.success("✅ Student is **NOT AT RISK**")
