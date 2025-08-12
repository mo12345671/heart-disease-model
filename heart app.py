import streamlit as st
import pandas as pd
import pickle

@st.cache_resource
def load_model():
    with open("heart_disease.sav", "rb") as f:
        return pickle.load(f)
selected_features = X.columns[selector.get_support()].tolist()
@st.cache_data
def load_selected_features():
    with open("selected_features.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()
selected_features = load_selected_features()

st.title("❤️ Heart Disease Prediction")

user_input = {}
for feature in selected_features:
    user_input[feature] = st.number_input(f"Enter {feature}", value=0.0)

input_df = pd.DataFrame([user_input])

if st.button("Predict"):
    try:
        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]
        if pred == 1:
            st.error(f"⚠️ High risk of heart disease! Probability: {proba:.2f}")
        else:
            st.success(f"✅ Low risk of heart disease. Probability: {proba:.2f}")
    except Exception as e:
        st.error(f"Prediction error: {e}")
