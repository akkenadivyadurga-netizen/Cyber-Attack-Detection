import pandas as pd
import joblib
import numpy as np

# Load trained model
model = joblib.load("model/cyber_attack_model.pkl")

# Load test data
df = pd.read_csv("dataset/train_binary.csv")

df.columns = df.columns.str.strip()

# Clean data
df = df.replace([np.inf, -np.inf], np.nan)
df = df.fillna(0)

# Remove Label
X = df.drop("Label", axis=1)

# Make predictions
predictions = model.predict(X)

print("Prediction completed!")

print("\nFirst 20 predictions:")
print(predictions[:20])

print("\n0 = Normal")
print("1 = Attack")

print("\nPrediction counts:")
print(pd.Series(predictions).value_counts())