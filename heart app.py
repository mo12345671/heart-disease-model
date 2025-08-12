import streamlit as st
import pandas as pd
import numpy as np
import pickle

MODEL_FILE = "heart_disease.sav"
DATA_FILE = "heart.csv"

st.title("❤️ Heart Disease Prediction")

# Load dataset to get feature names and ranges
@st.cache_data
def load_data():
    return pd.read_csv(DATA_FILE)

df = load_data()

# Load model
@st.cache_resource
def load_model():
    return pickle.load(open(MODEL_FILE, "rb"))

model = None
try:
    model = load_model()
except FileNotFoundError:
    st.error(f"Model file '{MODEL_FILE}' not found. Please train the model first.")

if model:
    # Create inputs dynamically based on features (excluding target)
    user_input = {}
    for feature in df.drop("target", axis=1).columns:
        min_val = float(df[feature].min())
        max_val = float(df[feature].max())
        mean_val = float(df[feature].mean())
        user_input[feature] = st.number_input(
            label=feature, min_value=min_val, max_value=max_val, value=mean_val
        )

    input_df = pd.DataFrame([user_input])

    if st.button("Predict"):
        prediction = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        if prediction == 1:
            st.error(f"⚠️ High risk of heart disease (probability: {prob:.2f})")
        else:
            st.success(f"✅ Low risk of heart disease (probability: {prob:.2f})")
