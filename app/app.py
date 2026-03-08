import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# -------------------------------
# Load Paths
# -------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

model_path = BASE_DIR / "models" / "logistic_model.pkl"
scaler_path = BASE_DIR / "models" / "scaler.pkl"
columns_path = BASE_DIR / "models" / "columns.pkl"

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
columns = joblib.load(columns_path)

# -------------------------------
# Page Config
# -------------------------------

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

st.title("❤️ Heart Disease Prediction")
st.caption("ML Model by Saurabh Navale")

# -------------------------------
# Compact Input Layout
# -------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", 18, 100, 40)
    sex = st.selectbox("Sex", ["M", "F"])
    chest_pain = st.selectbox("Chest Pain", ["ATA", "NAP", "TA", "ASY"])
    resting_bp = st.number_input("Rest BP", 80, 200, 120)

with col2:
    cholesterol = st.number_input("Cholesterol", 100, 600, 200)
    fasting_bs = st.selectbox("Fasting BS >120", [0, 1])
    resting_ecg = st.selectbox("Rest ECG", ["Normal", "ST", "LVH"])
    max_hr = st.slider("Max HR", 60, 220, 150)

with col3:
    exercise_angina = st.selectbox("Exercise Angina", ["Y", "N"])
    oldpeak = st.slider("Oldpeak", 0.0, 6.0, 1.0)
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

st.write("")

# -------------------------------
# Prediction
# -------------------------------

if st.button("Predict"):

    raw_data = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,
        "Sex_" + sex: 1,
        "ChestPainType_" + chest_pain: 1,
        "RestingECG_" + resting_ecg: 1,
        "ExerciseAngina_" + exercise_angina: 1,
        "ST_Slope_" + st_slope: 1
    }

    input_df = pd.DataFrame([raw_data])

    for col in columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[columns]

    scaled_input = scaler.transform(input_df)

    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    if prediction == 1:
        st.error(f"High Risk ({probability*100:.1f}% probability)")
    else:
        st.success(f"Low Risk ({(1-probability)*100:.1f}% probability)")