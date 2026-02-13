import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Setup Paths 
BASE_DIR = "/mnt/ml-data"
DATA_PATH = os.path.join(BASE_DIR, "datasets/raw.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
LOG_DIR = os.path.join(BASE_DIR, "logs")


columns = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", 
           "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", 
           "hours-per-week", "native-country", "income"]

# 2. Load and Preprocess Data 
df = pd.read_csv(DATA_PATH, names=columns, skipinitialspace=True)
le = LabelEncoder()
df['income'] = le.fit_transform(df['income'])

# Selecting numeric features for the pipeline
X = df.select_dtypes(include=[np.number]).drop(columns=['income'])
y = df['income']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train Two Models 
print("Model 1: Logistic Regression...")
m1 = LogisticRegression()
m1.fit(X_train_scaled, y_train)
acc1 = accuracy_score(y_test, m1.predict(X_test_scaled))

print("Model 2: Random Forest...")
m2 = RandomForestClassifier(n_estimators=10, random_state=42)
m2.fit(X_train_scaled, y_train)
acc2 = accuracy_score(y_test, m2.predict(X_test_scaled))

# 5. Select Best Model Automatically
if acc1 > acc2:
    best_model, best_name, best_acc = m1, "LogisticRegression", acc1
else:
    best_model, best_name, best_acc = m2, "RandomForest", acc2

print(f"--- Results ---")
print(f"Model: {best_name}")
print(f"Accuracy: {best_acc:.2f}")
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d')}")

# 6. Save Model and Log Metrics to EBS 
joblib.dump(best_model, os.path.join(MODEL_DIR, "best_model.pkl"))

with open(os.path.join(LOG_DIR, "metrics.log"), "a") as f:
    f.write(f"Timestamp: {datetime.now()} | Model: {best_name} | Accuracy: {best_acc:.2f}\n")

print(f"Model and logs saved to {BASE_DIR}")
