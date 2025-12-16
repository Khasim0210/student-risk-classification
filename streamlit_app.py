import streamlit as st
import mlflow.pyfunc
import os
import pandas as pd
from pathlib import Path

# --- MLflow Configuration (Connect to DAGsHub/MLflow Server) ---
# NOTE: The credentials (MLFLOW_TRACKING_URI, USERNAME, PASSWORD) 
# must be set in Streamlit Secrets, NOT hardcoded here.
if "MLFLOW_TRACKING_URI" in st.secrets:
    os.environ["MLFLOW_TRACKING_URI"] = st.secrets["MLFLOW_TRACKING_URI"]
    if "MLFLOW_TRACKING_USERNAME" in st.secrets:
        os.environ["MLFLOW_TRACKING_USERNAME"] = st.secrets["MLFLOW_TRACKING_USERNAME"]
        os.environ["MLFLOW_TRACKING_PASSWORD"] = st.secrets["MLFLOW_TRACKING_PASSWORD"]


st.set_page_config(page_title="Student Risk Classification", layout="centered")

# --- Load model from remote MLflow Registry URI ---
@st.cache_resource
def load_model():
    # 🎯 IMPORTANT: This URI MUST match the model location in your MLflow Registry
    # This URI is copied from your app.py.
    MODEL_URI = "models:/Logistic_Regression_Baseline/Production" 
    
    st.info(f"Attempting to load model from: {MODEL_URI}")
    
    try:
        # We use mlflow.pyfunc.load_model because that is what Streamlit is expecting
        # (even though the model was saved as sklearn, pyfunc is more generic).
        model = mlflow.pyfunc.load_model(MODEL_URI) 
        return model
    except Exception as e:
        # If this fails, the issue is with the URI or the secrets/authentication
        st.error("Model loading FAILED. Check the model URI and Streamlit Secrets.")
        st.exception(e)
        st.stop()


model = load_model()

# =========================
# Streamlit UI (Rest of the code remains the same)
# =========================
st.title("🎓 Student Risk Classification")
st.write("Predict whether a student is **at risk** based on academic and demographic factors.")
st.markdown("---")

# ... (The rest of your Streamlit UI code goes here)
with st.sidebar:
    st.header("Student Input Features")
    age = st.number_input("Age", min_value=10, max_value=30, value=18)
    studytime = st.selectbox("Weekly Study Time (1: <2h, 4: >10h)", [1, 2, 3, 4])
    failures = st.selectbox("Past Class Failures", [0, 1, 2, 3], index=0)
    absences = st.number_input("Number of Absences", min_value=0, max_value=100, value=5)

    # Binary features
    schoolsup = st.selectbox("School Support", ["yes", "no"], index=1)
    famsup = st.selectbox("Family Support", ["yes", "no"], index=0)
    activities = st.selectbox("Extra Curricular Activities", ["yes", "no"], index=0)


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
    try:
        prediction = model.predict(input_data)[0]

        st.subheader("Prediction Result")
        if prediction == 1:
            st.error("⚠️ Student is **AT RISK**")
            st.metric(label="Risk Indicator", value="High", delta="Needs Intervention")
        else:
            st.success("✅ Student is **NOT AT RISK**")
            st.metric(label="Risk Indicator", value="Low", delta="On Track")
            
    except Exception as e:
        st.exception(e)
        st.error("Prediction failed. Check model compatibility with input data.")
