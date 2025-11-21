import os
import joblib
import pandas as pd
import streamlit as st


class DiabetesPredictionApp:
    def __init__(self, model_path: str = "models/diabetes_model.pkl"):
        """
        model_path: path to the saved sklearn pipeline (joblib file)
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        # load trained pipeline (preprocessor + logistic regression)
        self.clf = joblib.load(model_path)

        # MUST match the columns used during training
        self.feature_names = [
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",  # include only if used in training
            "Age",
        ]

    def predict(self, input_df: pd.DataFrame):
        """
        input_df: single-row DataFrame with same columns as feature_names
        """
        pred = self.clf.predict(input_df)[0]

        prob = None
        if hasattr(self.clf, "predict_proba"):
            prob = self.clf.predict_proba(input_df)[0][1]  # probability of class 1

        return pred, prob


# ---------- Streamlit UI ----------

def main():
    st.set_page_config(page_title="Diabetes Prediction App", page_icon="🩺")

    st.title("🩺 Diabetes Prediction App")
    st.write("Provide patient details to predict the likelihood of diabetes.")

    # initialise model
    try:
        app = DiabetesPredictionApp(model_path="models/diabetes_model.pkl")
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    st.sidebar.header("Input Features")

    pregnancies = st.sidebar.number_input("Pregnancies", min_value=0, max_value=20, value=2)
    glucose = st.sidebar.number_input("Glucose", min_value=0.0, max_value=300.0, value=130.0)
    blood_pressure = st.sidebar.number_input("Blood Pressure", min_value=0.0, max_value=200.0, value=80.0)
    skin_thickness = st.sidebar.number_input("Skin Thickness", min_value=0.0, max_value=100.0, value=25.0)
    insulin = st.sidebar.number_input("Insulin", min_value=0.0, max_value=900.0, value=85.0)
    bmi = st.sidebar.number_input("BMI", min_value=0.0, max_value=70.0, value=28.5)
    dpf = st.sidebar.number_input(
        "Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01
    )
    age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=45)

    # assemble into DataFrame in the SAME order as training
    input_data = pd.DataFrame(
        [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]],
        columns=app.feature_names,
    )

    st.subheader("Entered Data")
    st.dataframe(input_data)

    if st.button("🔍 Predict"):
        pred, prob = app.predict(input_data)

        if pred == 1:
            st.error("⚠️ The model predicts this person is **Diabetic**.")
        else:
            st.success("✅ The model predicts this person is **Non-Diabetic**.")

        if prob is not None:
            st.write(f"Estimated probability of diabetes: **{prob:.2%}**")


if __name__ == "__main__":
    main()
