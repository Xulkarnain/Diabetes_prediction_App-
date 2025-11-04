import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("models/diabetes_model.pkl")

# App title
st.title("🩺 Diabetes Prediction App")
st.write("Predict whether a person is diabetic based on medical inputs.")

# Input fields
Pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=0)
Glucose = st.number_input("Glucose Level", min_value=0.0, max_value=300.0, value=120.0)
BloodPressure = st.number_input("Blood Pressure", min_value=0.0, max_value=200.0, value=70.0)
SkinThickness = st.number_input("Skin Thickness", min_value=0.0, max_value=100.0, value=20.0)
Insulin = st.number_input("Insulin", min_value=0.0, max_value=900.0, value=80.0)
BMI = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
Age = st.number_input("Age", min_value=1, max_value=120, value=30)

# Prediction button
if st.button("Predict"):
    # Prepare data for prediction
    input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
    
    # Predict
    prediction = model.predict(input_data)[0]
    
    # Display result
    if prediction == 1:
        st.error("⚠️ The model predicts this person is likely **Diabetic.**")
    else:
        st.success("✅ The model predicts this person is **Non-Diabetic.**")
