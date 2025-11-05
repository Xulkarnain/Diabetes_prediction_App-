from preprocessing import pipeline, x_train, x_test, y_train, y_test
import joblib

# Evaluate model
y_train_pred = pipeline.predict(x_train)
y_test_pred = pipeline.predict(x_test)

print("Train Predictions:", y_train_pred[:5])
print("Test Predictions:", y_test_pred[:5])

# Refit the pipeline (just to be sure)
pipeline.fit(x_train, y_train)
print("Pipeline fitted to training data.")

# Save the trained model properly
joblib.dump(pipeline, "models/diabetes_model.pkl")
print("✅ Model saved to 'models/diabetes_model.pkl'")
