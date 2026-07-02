import pandas as pd
import shap

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load data
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

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Train Random Forest
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

# SHAP
explainer = shap.TreeExplainer(rf_model)

shap_values = explainer.shap_values(X_test)

# Summary Plot
shap.summary_plot(
    shap_values,
    X_test
)

# Bar Plot
shap.summary_plot(
    shap_values,
    X_test,
    plot_type="bar"
)