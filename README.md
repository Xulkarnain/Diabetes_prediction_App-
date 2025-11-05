# 🩺 Diabetes Prediction App

Predict whether a person is diabetic or not using medical attributes like glucose level, BMI, insulin, etc.
This project demonstrates a full Machine Learning pipeline, from preprocessing to model training, saving, and real-time prediction using Streamlit.
Special Thanks to Krishnaik academy and Monal Kumar sir for guidance , and making this project come to life 🖤.

## Features
```
End-to-End ML Pipeline (data preprocessing → model training → saving → real-time prediction)

Handles missing/zero values with SimpleImputer

Standardizes data with StandardScaler

Logistic Regression classifier

Streamlit web app interface

Joblib model saving/loading

One-click .bat file to launch the app 
```
##  Tech Stack
```
Component	Library/Tool

Language	Python 3.11+

Framework	Streamlit

Machine Learning	scikit-learn

Data Processing	pandas, NumPy

Model Serialization	joblib

IDE (optional)	VS Code / Jupyter / PyCharm
```

## 📂 Project Structure
```
ML_Project/
│
├── data/
│   └── diabetes.csv                  # Dataset used for training
│
├── models/
│   └── diabetes_model.pkl            # Trained model saved by joblib
│
├── app/
│   ├── main.py                       # Streamlit UI for prediction
│   ├── preprocessing.py              # Handles data cleaning & pipeline setup
│   ├── prediction.py                 # Trains model and saves it to models/
│   └── run_app.bat                   # Windows batch file to auto-run app
│
├── README.md                         # Project documentation
└── requirements.txt                  # All dependencies
```
## 📈 Model Overview
```
Algorithm: Logistic Regression

Preprocessing:

Median imputation for zeros in critical columns (Glucose, BMI, etc.)

Standard scaling of numeric features

Evaluation:

Stratified train-test split (90%-10%)

Pipeline-based preprocessing for consistency
```
## ⚙️ Installation & Setup

1. Clone the repository
```
git clone https://github.com/<your-username>/Diabetes-Prediction-App.git
cd Diabetes-Prediction-App
```
2. Create a virtual environment
```
python -m venv penv
```
- Activate it:

- Windows:
```
penv\Scripts\activate
```
- Linux/Mac:
```
source penv/bin/activate
```
3. Install dependencies
```
pip install -r requirements.txt
```

## 👟Training the Model

- Run:
```
python app/prediction.py
```
- This script:
1.Imports preprocessing steps

2.Trains the Logistic Regression model

3.Saves the trained model to models/diabetes_model.pkl

If- successful, you’ll see:
```
✅ Model saved to 'models/diabetes_model.pkl'
```

## Running the Streamlit App

- Option 1 — Manual:
```
streamlit run app/main.py
```
- Option 2 — Using .bat file (Windows only):
```
Double-click run_app.bat in the project directory.
It will automatically launch the Streamlit web app in your browser.
```

## Author
```
Zulkarnain
🎓 B.Tech CSE | Data Science Enthusiast
💡 Focused on building strong ML & Python foundations
```

## 🪪 License
```
This project is open-source under the MIT License — you’re free to use, modify, and distribute it with proper credit.
```
