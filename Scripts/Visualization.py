import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Optional: Better-looking plots
sns.set_style("whitegrid")
df = pd.read_csv("synthetic_student_performance_200.csv")

# Check first few rows
print(df.head())
plt.figure(figsize=(8,5))

plt.hist(df['CGPA'], bins=10)

plt.title('Distribution of CGPA')
plt.xlabel('CGPA')
plt.ylabel('Number of Students')

plt.show()
attendance_counts = df['Attendance'].value_counts()

plt.figure(figsize=(8,5))

attendance_counts.plot(kind='bar')

plt.title('Attendance Frequency')
plt.xlabel('Attendance Category')
plt.ylabel('Number of Students')

plt.show()
study_counts = df['Study_Hours'].value_counts()

plt.figure(figsize=(8,5))

study_counts.plot(kind='bar')

plt.title('Study Hours Distribution')
plt.xlabel('Study Hours')
plt.ylabel('Number of Students')

plt.show()
stress_counts = df['Stress_Level'].value_counts()

plt.figure(figsize=(8,5))

stress_counts.plot(kind='bar')

plt.title('Stress Level Distribution')
plt.xlabel('Stress Level')
plt.ylabel('Number of Students')

plt.show()
plt.figure(figsize=(8,5))

sns.boxplot(
    x='Attendance',
    y='CGPA',
    data=df
)

plt.title('CGPA vs Attendance')
plt.xlabel('Attendance Category')
plt.ylabel('CGPA')

plt.show()
plt.figure(figsize=(8,5))

sns.boxplot(
    x='Social_Media_Usage',
    y='CGPA',
    data=df
)

plt.title('CGPA vs Social Media Usage')
plt.xlabel('Daily Social Media Usage')
plt.ylabel('CGPA')

plt.show()

df_encoded = df.copy()

for col in df_encoded.columns:
    if df_encoded[col].dtype != 'int64' and df_encoded[col].dtype != 'float64':
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

plt.figure(figsize=(12,8))

sns.heatmap(
    df_encoded.corr(),
    annot=True,
    cmap='coolwarm'
)

plt.title('Correlation Heatmap')

plt.show()