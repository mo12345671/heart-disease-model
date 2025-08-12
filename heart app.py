import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc
import plotly.express as px
import plotly.graph_objects as go

# Constants for file paths
MODEL_FILE = "heart_disease.sav"
DATA_FILE = "heart.csv"

st.set_page_config(page_title="Heart Disease Prediction App", layout="centered")
st.title("❤️ Heart Disease Prediction App")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv(DATA_FILE)

df = load_data()

# Train model function (no scaling)
def train_model(test_size=0.2):
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    model = SVC(probability=True, random_state=42)
    model.fit(X_train, y_train)

    # Save model only
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)

    return model, X_train, X_test, y_train, y_test

# Load model function
def load_model():
    if os.path.exists(MODEL_FILE):
        model = pickle.load(open(MODEL_FILE, "rb"))
        return model
    else:
        return None

# Sidebar menu
menu = st.sidebar.radio("Navigation", ["Data Overview", "Model Training", "Prediction"])

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

elif menu == "Model Training":
    st.subheader("Train SVC Model (No Scaling)")

    test_size = st.slider("Test size (%)", 10, 40, 20)
    model, X_train, X_test, y_train, y_test = train_model(test_size / 100)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    st.write(f"**Train Accuracy:** {accuracy_score(y_train, model.predict(X_train)):.4f}")
    st.write(f"**Test Accuracy:** {accuracy_score(y_test, y_pred):.4f}")
    st.write(f"**ROC AUC Score:** {roc_auc_score(y_test, y_proba):.4f}")

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC curve (AUC = {roc_auc:.2f})'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random guess', line=dict(dash='dash')))
    fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    st.plotly_chart(fig)

    st.success("✅ Model saved as heart_disease.sav!")

elif menu == "Prediction":
    st.subheader("Make a Prediction")

    model = load_model()
    if model is None:
        st.error("Model not found. Please train the model first in 'Model Training' section.")
    else:
        input_data = {}
        for feature in df.drop("target", axis=1).
