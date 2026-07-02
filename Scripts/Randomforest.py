
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

df = pd.read_csv("student_performance_encoded.csv")


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


rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

print("Random Forest Model Trained Successfully!")


y_pred_rf = rf_model.predict(X_test)

r2 = r2_score(y_test, y_pred_rf)

mse = mean_squared_error(y_test, y_pred_rf)

rmse = mse ** 0.5

mae = mean_absolute_error(y_test, y_pred_rf)

print("\n========== RANDOM FOREST RESULTS ==========")

print("R² Score :", round(r2, 4))
print("MAE      :", round(mae, 4))
print("MSE      :", round(mse, 4))
print("RMSE     :", round(rmse, 4))


results = pd.DataFrame({
    'Actual_CGPA': y_test,
    'Predicted_CGPA': y_pred_rf
})

print("\nSample Predictions:\n")
print(results.head(10))

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by='Importance',
    ascending=False
)

print("\n========== FEATURE IMPORTANCE ==========\n")

print(feature_importance)


import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))

plt.bar(
    feature_importance['Feature'],
    feature_importance['Importance']
)

plt.title("Random Forest Feature Importance")

plt.xlabel("Features")
plt.ylabel("Importance Score")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()