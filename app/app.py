import streamlit as st
import numpy as np
import tensorflow as tf
import joblib

# Load model and scaler
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = tf.keras.models.load_model(
    os.path.join(BASE_DIR, "diabetes_model.keras")
)

scaler = joblib.load(
    os.path.join(BASE_DIR, "scaler.pkl")
)


# Title
st.title("Diabetes Prediction using Feed Forward Neural Network")

st.write("Enter patient details below")

# Input fields
preg = st.number_input("Pregnancies", min_value=0)

glucose = st.number_input("Glucose", min_value=0.0)

bp = st.number_input("Blood Pressure", min_value=0.0)

skin = st.number_input("Skin Thickness", min_value=0.0)

insulin = st.number_input("Insulin", min_value=0.0)

bmi = st.number_input("BMI", min_value=0.0)

dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)

age = st.number_input("Age", min_value=1)

# Predict button
if st.button("Predict"):

    input_data = np.array([[preg,
                            glucose,
                            bp,
                            skin,
                            insulin,
                            bmi,
                            dpf,
                            age]])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)

    probability = prediction[0][0]

    if probability >= 0.5:
        st.error("⚠️ Patient is likely Diabetic")
    else:
        st.success("✅ Patient is Not Diabetic")

    st.metric("Diabetes Risk", f"{probability * 100:.2f}%")

    st.info(
    "⚠️ This application is for educational purposes only and should not be used as a substitute for professional medical advice."
)