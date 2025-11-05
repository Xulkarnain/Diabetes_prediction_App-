import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv(r'D:\ML_Project\data\diabetes.csv')

target_column = 'Outcome'
x = df.drop(columns=[target_column])
y = df[target_column]

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.1, random_state=42, stratify=y
)

# Columns where 0 means missing
columns_with_zero = ["Insulin", "SkinThickness", "BloodPressure", "BMI", "Glucose"]

zero_missing_cols = [c for c in columns_with_zero if c in x_train.columns]

numeric_cols = x.select_dtypes(include=[np.number]).columns.tolist()
remaining_cols = [c for c in numeric_cols]

transformers = []

if zero_missing_cols:
    impute_then_scale = SklearnPipeline([
        ("imputer_zero_as_missing", SimpleImputer(missing_values=0, strategy="median")),
        ("scaler", StandardScaler())
    ])
    transformers.append(("impute_scale", impute_then_scale, zero_missing_cols))

    # Remaining numerical columns
    remaining_after_impute = [c for c in remaining_cols if c not in zero_missing_cols]
    if remaining_after_impute:
        transformers.append(("scaler_all", SklearnPipeline([("scaler", StandardScaler())]), remaining_after_impute))
else:
    transformers.append(("scaler_all", SklearnPipeline([("scaler", StandardScaler())]), remaining_cols))

preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

# Logistic Regression model
clf = LogisticRegression(solver="liblinear", max_iter=1000, random_state=42)

# Final pipeline
pipeline = SklearnPipeline([
    ("preprocessor", preprocessor),
    ("clf", clf)
])

# Fit the pipeline
pipeline.fit(x_train, y_train)
print("✅ Pipeline fitted successfully!")
