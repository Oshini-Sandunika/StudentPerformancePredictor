# =========================================
# TRAIN STUDENT PERFORMANCE MODEL - RANDOM FOREST REGRESSOR (TUNED)
# =========================================

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# ---------------- LOAD DATASET ----------------
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(project_dir, "data", "StudentPerformanceFactors.csv")

df = pd.read_csv(data_path)
print("Columns in dataset:", df.columns.tolist())
print("Number of rows:", df.shape[0])

# ---------------- SELECT ESSENTIAL FEATURES ----------------
essential_features = [
    "Hours_Studied",
    "Attendance",
    "Sleep_Hours",
    "Previous_Scores",
    "Motivation_Level",
    "Internet_Access",
    "Tutoring_Sessions",
    "Physical_Activity"
]

df = df[essential_features + ["Exam_Score"]]

# ---------------- ENCODE CATEGORICAL FEATURES ----------------
for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

# ---------------- TARGET VARIABLE ----------------
TARGET_COL = "Exam_Score"  # Predict raw exam score

# ---------------- CREATE FEATURE INTERACTIONS ----------------
df["Study_Motivation"] = df["Hours_Studied"] * df["Motivation_Level"]
df["Attendance_Effort"] = df["Attendance"] * df["Tutoring_Sessions"]

# Final feature set
features = essential_features + ["Study_Motivation", "Attendance_Effort"]

# ---------------- SPLIT FEATURES & TARGET ----------------
X = df[features]
y = df[TARGET_COL]  # Continuous exam score

# ---------------- TRAIN-TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- SCALE NUMERIC FEATURES ----------------
numeric_cols = features

scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

# ---------------- TRAIN RANDOM FOREST REGRESSOR ----------------
model = RandomForestRegressor(
    n_estimators=400,
    max_depth=12,
    min_samples_split=4,
    min_samples_leaf=2,
    max_features='log2',
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ---------------- CROSS-VALIDATION ----------------
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
print(f"\nAverage CV R² Score: {np.mean(cv_scores):.3f}")

# ---------------- EVALUATE MODEL ----------------
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\n✅ Test Results:")
print(f"R² Score: {r2:.3f}")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")

# ---------------- SAVE MODEL & SCALER ----------------
model_path = os.path.join(project_dir, "model.pkl")
scaler_path = os.path.join(project_dir, "scaler.pkl")

joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)

print(f"\n✅ Model saved to: {model_path}")
print(f"✅ Scaler saved to: {scaler_path}")
