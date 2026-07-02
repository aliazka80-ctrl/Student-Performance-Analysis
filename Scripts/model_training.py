import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

df = pd.read_csv("student_performance_encoded.csv")

print("Dataset Shape:", df.shape)

X = df[
    [
        'Attendance',
        'Study_Hours',
        'Sleep',
        'Stress_Level',
        'Motivation',
        'Social_Media_Usage',
        'Time_Management'
    ]
]

y = df['CGPA']


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

model = LinearRegression()

model.fit(X_train, y_train)

print("\nModel trained successfully!")
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = mse ** 0.5

mae = mean_absolute_error(y_test, y_pred)

print("\n========== MODEL PERFORMANCE ==========")

print("R² Score :", round(r2, 4))
print("MAE      :", round(mae, 4))
print("MSE      :", round(mse, 4))
print("RMSE     :", round(rmse, 4))

results = pd.DataFrame({
    'Actual_CGPA': y_test,
    'Predicted_CGPA': y_pred
})

print("\nSample Predictions:\n")
print(results.head(10))

coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})

print("\nFeature Coefficients:\n")
print(coefficients.sort_values(
    by='Coefficient',
    ascending=False
))