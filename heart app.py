import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load saved objects
@st.cache_resource
def load_model():
    with open("heart_disease_model.sav", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_scaler():
    with open("scaler.sav", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_features():
    with open("selected_features.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()
scaler = load_scaler()
selected_features = load_features()

st.title("❤️ Heart Disease Prediction")

user_input = {}
for feature in selected_features:
    val = st.number_input(f"Enter {feature}", 
                          min_value=0.0, max_value=300.0, value=100.0)
    user_input[feature] = val

input_df = pd.DataFrame([user_input])

# Scale and select features
input_scaled = scaler.transform(input_df[selected_features])

if st.button("Predict"):
    pred = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0][1]
    if pred == 1:
        st.error(f"⚠️ High risk of heart disease! Probability: {proba:.2f}")
    else:
        st.success(f"✅ Low risk of heart disease. Probability: {proba:.2f}")
