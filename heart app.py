import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import plotly.express as px

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
    df = pd.read_csv("heart_disease_data.csv")
    return df

df = load_data()

# -----------------------------
# Sidebar navigation
# -----------------------------
menu = st.sidebar.radio("Navigation", ["Data Overview", "Model Training", "Prediction"])

# -----------------------------
# Data Overview
# -----------------------------
if menu == "Data Overview":
    st.subheader("Dataset Preview")
    st.write(df.head())

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

    X = df.drop("target", axis=1)
    y = df["target"]

    test_size = st.slider("Test size (%)", 10, 40, 20) / 100
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = SVC(probability=True, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    st.write(f"**Train Accuracy:** {accuracy_score(y_train, model.predict(X_train)):.4f}")
    st.write(f"**Test Accuracy:** {accuracy_score(y_test, y_pred):.4f}")
    st.write(f"**ROC AUC Score:** {roc_auc_score(y_test, y_proba):.4f}")

    # Save model & scaler
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    st.success("Model & scaler saved successfully!")

# -----------------------------
# Prediction
# -----------------------------
elif menu == "Prediction":
    st.subheader("Predict Heart Disease Risk")

    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    # Input fields for features
    features = {}
    for col in df.drop("target", axis=1).columns:
        features[col] = st.number_input(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))

    # Convert to array and scale
    input_data = np.array(list(features.values())).reshape(1, -1)
    input_scaled = scaler.transform(input_data)

    if st.button("Predict"):
        prediction = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]
        if prediction == 1:
            st.error(f"⚠️ High risk of heart disease (Probability: {prob:.2f})")
        else:
            st.success(f"✅ Low risk of heart disease (Probability: {prob:.2f})")
