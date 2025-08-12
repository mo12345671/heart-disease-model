import streamlit as st
import pandas as pd
import pickle


selected_features = X.columns[selector.get_support()].tolist()

with open("selected_features.pkl", "wb") as f:
    pickle.dump(selected_features, f)

# Load saved model and selected features
@st.cache_resource
def load_model():
    with open("heart_disease.sav", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_features():
    with open("selected_features.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()
selected_features = load_features()

st.title("❤️ Heart Disease Prediction (No Scaler)")

user_input = {}
for feature in selected_features:
    val = st.number_input(f"Enter {feature}", value=0.0)
    user_input[feature] = val

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
        st.error(f"Error during prediction: {e}")


