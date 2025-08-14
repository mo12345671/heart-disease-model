import streamlit as st
import pickle
import numpy as np

# -----------------------------
# Load the trained model
# -----------------------------
model = pickle.load(open('heart_disease.sav', 'rb'))

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Heart Disease Prediction", layout="centered")
st.title("❤️ Heart Disease Prediction App")
st.write("Enter patient details to predict the likelihood of heart disease.")

# -----------------------------
# Collect user input
# -----------------------------
# Replace these with the exact features you used in training
age = st.number_input("Age", min_value=1, max_value=120, value=40)
sex = st.selectbox("Sex", ("Male", "Female"))
cp = st.selectbox("Chest Pain Type (cp)", (0, 1, 2, 3))
trestbps = st.number_input("Resting Blood Pressure (trestbps)", min_value=80, max_value=200, value=120)
chol = st.number_input("Cholesterol (chol)", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", (0, 1))
restecg = st.selectbox("Resting ECG results (restecg)", (0, 1, 2))
thalach = st.number_input("Maximum Heart Rate Achieved (thalach)", min_value=60, max_value=220, value=150)
exang = st.selectbox("Exercise Induced Angina (exang)", (0, 1))
oldpeak = st.number_input("ST depression induced by exercise (oldpeak)", min_value=0.0, max_value=10.0, value=1.0)
slope = st.selectbox("Slope of peak exercise ST segment (slope)", (0, 1, 2))
ca = st.selectbox("Number of major vessels colored by fluoroscopy (ca)", (0, 1, 2, 3, 4))
thal = st.selectbox("Thalassemia (thal)", (0, 1, 2, 3))

# Map categorical inputs if necessary
sex = 1 if sex == "Male" else 0

# -----------------------------
# Predict button
# -----------------------------
if st.button("Predict"):
    features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                           thalach, exang, oldpeak, slope, ca, thal]])
    
    prediction = model.predict(features)[0]
    
    if prediction == 1:
        st.error("⚠️ The model predicts **Heart Disease**.")
    else:
        st.success("✅ The model predicts **No Heart Disease**.")

# -----------------------------
# To run:
# streamlit run app.py
# -----------------------------
