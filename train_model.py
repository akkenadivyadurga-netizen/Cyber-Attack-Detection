import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import numpy as np

df = pd.read_csv("dataset/train_binary.csv")

df.columns = df.columns.str.strip()

# Replace infinity values with NaN
df = df.replace([np.inf, -np.inf], np.nan)

# Replace missing values with 0
df = df.fillna(0)

X = df.drop("Label", axis=1)
y = df["Label"]

# Keep only numeric columns
X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training Random Forest model...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Model training completed!")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

joblib.dump(model, "model/cyber_attack_model.pkl")

print("\nModel saved successfully!")
