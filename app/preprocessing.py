import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('D:\ML_Project\data\diabetes.csv')

target_column = 'Outcome'
x = df.drop(columns=[target_column])
y = df[target_column]

x_train, x_test , y_train, y_test = train_test_split(
    x , y, test_size=0.1, random_state=42, stratify=y
)

# Step 1 : do the imputation
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

    # numerical colms & not part of imputation
    remaining_after_impute = [c for c in remaining_cols if c not in zero_missing_cols]
    if remaining_after_impute:
        transformers.append(("scaler_all", SklearnPipeline([("scaler", StandardScaler())]), remaining_after_impute))

else :
    transformers.append(("scaler_all", SklearnPipeline([("scaler", StandardScaler())]), remaining_cols))

preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

clf = LogisticRegression(solver="liblinear", max_iter=1000, random_state=42)

pipeline = SklearnPipeline([("preprocessor", preprocessor), ("clf", clf)])

pipeline.fit(x_train, y_train)
print("Pipeline fitted to training data.")