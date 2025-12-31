import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

st.set_page_config(page_title="AI-Based NIDS", layout="wide")

# -----------------------------
# Data Loader
# -----------------------------
def load_data(simulated=True):
    if simulated:
        np.random.seed(42)
        data = {
            'duration': np.random.randint(0, 1000, 1000),
            'src_bytes': np.random.randint(0, 50000, 1000),
            'dst_bytes': np.random.randint(0, 50000, 1000),
            'count': np.random.randint(0, 100, 1000),
            'srv_count': np.random.randint(0, 100, 1000),
            'label': np.random.choice([0, 1], 1000)  # 0=Normal, 1=Attack
        }
        return pd.DataFrame(data)
    else:
        return pd.read_csv("dataset/cic_ids_sample.csv")

# -----------------------------
# Train Model
# -----------------------------
def train_model(df):
    X = df.drop('label', axis=1)
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    return model, y_test, y_pred

# -----------------------------
# UI
# -----------------------------
st.title("🔐 AI-Based Network Intrusion Detection System")

st.markdown("""
This system uses **Machine Learning (Random Forest)** to detect  
**malicious network activity** in real time.
""")

mode = st.sidebar.radio("Select Mode", ["Simulation Mode", "Production Mode"])

if st.sidebar.button("🚀 Train Model Now"):
    with st.spinner("Training Model..."):
        df = load_data(simulated=(mode == "Simulation Mode"))
        model, y_test, y_pred = train_model(df)
        st.success("Model Trained Successfully!")

        st.subheader("📊 Classification Report")
        st.text(classification_report(y_test, y_pred))

        st.subheader("📉 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)

# -----------------------------
# Live Traffic Prediction
# -----------------------------
st.sidebar.markdown("### 🧪 Live Traffic Simulator")

duration = st.sidebar.number_input("Duration", 0, 10000, 100)
src_bytes = st.sidebar.number_input("Source Bytes", 0, 100000, 2000)
dst_bytes = st.sidebar.number_input("Destination Bytes", 0, 100000, 1500)
count = st.sidebar.number_input("Count", 0, 500, 10)
srv_count = st.sidebar.number_input("Service Count", 0, 500, 5)

if st.sidebar.button("🔍 Predict Traffic"):
    df = load_data()
    model, _, _ = train_model(df)
    sample = np.array([[duration, src_bytes, dst_bytes, count, srv_count]])
    prediction = model.predict(sample)

    if prediction[0] == 1:
        st.error("🚨 Intrusion Detected!")
    else:
        st.success("✅ Normal Traffic")
