import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix

# Page settings
st.set_page_config(
    page_title="Cyber Attack Detection",
    page_icon="🛡️",
    layout="wide"
)

# Load model
model = joblib.load("model/cyber_attack_model.pkl")

# Load dataset
df = pd.read_csv("dataset/train_binary.csv")

# Clean data
df.columns = df.columns.str.strip()
df = df.replace([float("inf"), float("-inf")], 0)
df = df.fillna(0)

# Features and label
X = df.drop("Label", axis=1)
y = df["Label"]

# Title
st.title("🛡️ Cyber Attack Detection System")
st.write("Machine Learning Based Network Traffic Detection")

st.divider()

# Dashboard
total_records = len(df)
normal_count = int((y == 0).sum())
attack_count = int((y == 1).sum())
attack_percentage = (attack_count / total_records) * 100

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("📊 Total Traffic", total_records)

with col2:
    st.metric("🟢 Normal Traffic", normal_count)

with col3:
    st.metric("🔴 Attack Traffic", attack_count)

with col4:
    st.metric("⚠️ Attack %", f"{attack_percentage:.2f}%")

st.divider()

# Attack Detection
st.subheader("🔍 Detect Network Attack")

sample_number = st.number_input(
    "Select a traffic record",
    min_value=0,
    max_value=len(X) - 1,
    value=0
)

if st.button("🔍 Detect Attack"):

    sample = X.iloc[[sample_number]]

    prediction = model.predict(sample)[0]

    st.divider()

    if prediction == 0:
        st.success("🟢 NORMAL TRAFFIC")
        st.write("No cyber attack detected.")

    else:
        st.error("🔴 CYBER ATTACK DETECTED")
        st.write("The network traffic has been classified as an attack.")

st.divider()

# Model Performance
st.subheader("📊 Model Performance")

accuracy = 0.99965

col1, col2 = st.columns(2)

with col1:
    st.metric("🎯 Model Accuracy", f"{accuracy * 100:.2f}%")

with col2:
    st.metric("🤖 Model", "Random Forest")

st.write(
    "The model was trained using 80% of the dataset "
    "and tested using 20%."
)

st.progress(accuracy)

st.divider()

# Confusion Matrix
st.subheader("📈 Confusion Matrix")

y_pred = model.predict(X)

cm = confusion_matrix(y, y_pred)

cm_df = pd.DataFrame(
    cm,
    index=["Actual Normal", "Actual Attack"],
    columns=["Predicted Normal", "Predicted Attack"]
)

st.dataframe(cm_df)

st.divider()

# Dataset Information
st.subheader("📋 Dataset Information")

info_col1, info_col2 = st.columns(2)

with info_col1:
    st.write("**Dataset Records:**", total_records)
    st.write("**Features:**", X.shape[1])

with info_col2:
    st.write("**Normal Records:**", normal_count)
    st.write("**Attack Records:**", attack_count)