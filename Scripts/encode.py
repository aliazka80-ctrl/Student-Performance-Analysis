import pandas as pd

# =====================================
# STEP 1: LOAD DATASET
# =====================================

df = pd.read_csv("synthetic_student_performance_200.csv")

print("Original Shape:", df.shape)

# =====================================
# STEP 2: ORDINAL ENCODING
# =====================================

attendance_map = {
    "Below 50%": 1,
    "50–70%": 2,
    "70–85%": 3,
    "Above 85%": 4
}

study_hours_map = {
    "Less than 1 hour": 1,
    "1–2 hours": 2,
    "2–4 hours": 3,
    "More than 4 hours": 4
}

stress_map = {
    "Very Low": 1,
    "Low": 2,
    "Moderate": 3,
    "High": 4,
    "Very High": 5
}

motivation_map = {
    "Very Low": 1,
    "Low": 2,
    "Moderate": 3,
    "High": 4,
    "Very High": 5
}

time_management_map = {
    "Very Poor": 1,
    "Poor": 2,
    "Average": 3,
    "Good": 4,
    "Excellent": 5
}

sleep_map = {
    "Less than 5 hours": 1,
    "5–6 hours": 2,
    "6–7 hours": 3,
    "7–8 hours": 4,
    "More than 8 hours": 5
}

social_media_map = {
    "Less than 1 hour": 1,
    "1–2 hours": 2,
    "2–4 hours": 3,
    "4–6 hours": 4,
    "More than 6 hours": 5
}

# Apply mappings

df["Attendance"] = df["Attendance"].map(attendance_map)
df["Study_Hours"] = df["Study_Hours"].map(study_hours_map)
df["Stress_Level"] = df["Stress_Level"].map(stress_map)
df["Motivation"] = df["Motivation"].map(motivation_map)
df["Time_Management"] = df["Time_Management"].map(time_management_map)
df["Sleep"] = df["Sleep"].map(sleep_map)
df["Social_Media_Usage"] = df["Social_Media_Usage"].map(social_media_map)

# =====================================
# STEP 3: YES / NO ENCODING
# =====================================

yes_no_columns = [
    "Study_Schedule",
    "Dedicated_Study_Environment"
]

for col in yes_no_columns:
    df[col] = df[col].map({
        "Yes": 1,
        "No": 0
    })

# =====================================
# STEP 4: ONE-HOT ENCODING
# =====================================

categorical_columns = [
    "Gender",
    "Year",
    "Department",
    "Revise_Notes",
    "Exam_Preparation",
    "Assignments_On_Time",
    "Class_Participation",
    "Ask_Questions",
    "Exercise",
    "Stress_Affects_Studies",
    "Most_Used_Platform",
    "Distracted_By_Social_Media",
    "Educational_Resources_Usage",
    "Main_Educational_Resource",
    "Reason_For_Degree",
    "Biggest_Factor",
    "Suggested_Improvement"
]

df_encoded = pd.get_dummies(
    df,
    columns=categorical_columns,
    drop_first=True
)

# =====================================
# STEP 5: VERIFY DATA TYPES
# =====================================

print("\nEncoded Data Types:\n")
print(df_encoded.dtypes)

print("\nEncoded Shape:", df_encoded.shape)

# =====================================
# STEP 6: SAVE DATASET
# =====================================

df_encoded.to_csv(
    "student_performance_encoded.csv",
    index=False
)

print("\nEncoded dataset saved successfully!")