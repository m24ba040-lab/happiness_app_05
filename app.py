import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("happiness_prediction_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Happiness Score Prediction App")

st.write("Enter your details below:")

# Input fields (must match training features)
sleep_hours = st.number_input("Sleep Hours", 0.0, 24.0, 7.0)
sleep_quality = st.number_input("Sleep Quality (1–10)", 1, 10, 7)
anxiety_score = st.number_input("Anxiety Score (1–10)", 1, 10, 5)
depression_score = st.number_input("Depression Score (1–10)", 1, 10, 5)
stress_level = st.number_input("Stress Level (1–10)", 1, 10, 5)
focus_score = st.number_input("Focus Score (1–10)", 1, 10, 6)
digital_dependence_score = st.number_input("Digital Dependence Score (1–10)", 1, 10, 5)
productivity_score = st.number_input("Productivity Score (1–10)", 1, 10, 6)
physical_activity_days = st.number_input("Physical Activity Days per Week (0–7)", 0, 7, 3)
study_mins = st.number_input("Daily Study Minutes", 0, 600, 120)

features = np.array([[
    sleep_hours, sleep_quality, anxiety_score, depression_score,
    stress_level, focus_score, digital_dependence_score, productivity_score,
    physical_activity_days, study_mins
]])

# Scale the input
scaled_features = scaler.transform(features)

# Predict
if st.button("Predict Happiness Score"):
    prediction = model.predict(scaled_features)[0]
    st.success(f"Predicted Happiness Score: {prediction:.2f}")
