import preprocessing
from preprocessing import pipeline
from preprocessing import x_train, x_test

y_train_pred = pipeline.predict(x_train)
y_test_pred = pipeline.predict(x_test)

y_train_pred, y_test_pred

import joblib

# After fitting pipeline
pipeline.fit(x_train, x_test)
print("Pipeline fitted to training data.")

# Save the model for later use
joblib.dump(pipeline, "models/diabetes_model.pkl")
print("Model saved to 'models/diabetes_model.pkl'")