import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Heart Disease Prediction", layout="centered")
st.title("❤️ Heart Disease Prediction App")

# -----------------------------
# Load dataset
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("heart_disease_data.csv")  # Make sure this file is in your repo

df = load_data()

# -----------------------------
# File paths
# -----------------------------
MODEL_FILE = "heart_disease.sav"
SCALER_FILE = "scaler.sav"

# -----------------------------
# Train and save model + scaler
# -----------------------------
def train_and_save_model(test_size=0.2):
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = SVC(probability=True, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Save model and scaler
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)
    with open(SCALER_FILE, "wb") as f:
        pickle.dump(scaler, f)

    return model, scaler, X_train_scaled, X_test_scaled, y_train, y_test

# -----------------------------
# Load model and scaler
# -----------------------------
def load_model_and_scaler():
    if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
        with open(MODEL_FILE, "rb") as f:
            model = pickle.load(f)
        with open(SCALER_FILE, "rb") as f:
            scaler = pickle.load(f)
        return model, scaler
    else:
        return None, None

# -----------------------------
# Sidebar menu
# -----------------------------
menu = st.sidebar.radio("Navigation", ["Data Overview", "Model Training", "Prediction"])

# -----------------------------
# Data Overview
# -----------------------------
if menu == "Data Overview":
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Correlation Heatmap")
    corr = df.corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r")
    st.plotly_chart(fig)

    st.subheader("Target Distribution")
    fig2 = px.histogram(df, x="target", color="target", barmode="group")
    st.plotly_chart(fig2)

# -----------------------------
# Model Training
# -----------------------------
elif menu == "Model Training":
    st.subheader("Train SVC Model with MinMaxScaler")

    test_size_slider = st.slider("Test size (%)", 10, 40, 20)
    model, scaler, X_train_scaled, X_test_scaled, y_train, y_test = train_and_save_model(test_size_slider / 100)

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    st.write(f"**Train Accuracy:** {accuracy_score(y_train, model.predict(X_train_scaled)):.4f}")
    st.write(f"**Test Accuracy:** {accuracy_score(y_test, y_pred):.4f}")
    st.write(f"**ROC AUC Score:** {roc_auc_score(y_test, y_proba):.4f}")

    # Plot real ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))
    fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    st.plotly_chart(fig)

    st.success("✅ Model & scaler saved successfully!")

# -----------------------------
# Prediction
# -----------------------------
elif menu == "Prediction":
    st.subheader("Predict Heart Disease Risk")

    model, scaler = load_model_and_scaler()
    if not model or not scaler:
        st.error("❌ Model not found. Please train the model first in the 'Model Training' section.")
    else:
        features = {}
        for col in df.drop("target", axis=1).columns:
            features[col] = st.number_input(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))

        input_data = np.array(list(features.values())).reshape(1, -1)
        input_scaled = scaler.transform(input_data)

        if st.button("Predict"):
            pred
