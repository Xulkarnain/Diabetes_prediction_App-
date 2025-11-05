import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import sklearn
import sklearn.compose._column_transformer
import joblib

# Fix for "_RemainderColsList" error
if not hasattr(sklearn.compose._column_transformer, "_RemainderColsList"):
    sklearn.compose._column_transformer._RemainderColsList = list

MODEL_PATH = "D:\ML_Project\models\diabetes_model.pkl"

model = joblib.load(MODEL_PATH)

# App title
st.title("🩺 Diabetes Prediction App")
st.markdown("Enter patient details to predict diabetes status.")

# Input fields
Pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=0)
Glucose = st.number_input("Glucose Level", min_value=0.0, max_value=300.0, value=120.0)
BloodPressure = st.number_input("Blood Pressure", min_value=0.0, max_value=200.0, value=70.0)
SkinThickness = st.number_input("Skin Thickness", min_value=0.0, max_value=100.0, value=20.0)
Insulin = st.number_input("Insulin", min_value=0.0, max_value=900.0, value=80.0)
BMI = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
Age = st.number_input("Age", min_value=1, max_value=120, value=30)

if st.button("🔍 Predict"):
    # Create a DataFrame (not NumPy array)
    input_data = pd.DataFrame({
        'Pregnancies': [Pregnancies],
        'Glucose': [Glucose],
        'BloodPressure': [BloodPressure],
        'SkinThickness': [SkinThickness],
        'Insulin': [Insulin],
        'BMI': [BMI],
        'DiabetesPedigreeFunction': [DiabetesPedigreeFunction],
        'Age': [Age]
    })

    # Predict using the pipeline
    prediction = model.predict(input_data)[0]

    # Display result
    if prediction == 1:
        st.error("⚠️ The person is likely **Diabetic.**")
    else:
        st.success("✅ The person is **Non-Diabetic.**")
