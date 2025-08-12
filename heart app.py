import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler

# Load saved model
MODEL_PATH = "heart_disease.sav"

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model

# Hardcoded selected features - change if needed
selected_features = ['age', 'trestbps', 'chol', 'thalach']  # example features

# Hardcoded feature min and max values (replace these with your dataset's actual min/max)
feature_ranges = {
    'age': (29, 77),
    'trestbps': (94, 200),
    'chol': (126, 564),
    'thalach': (71, 202),
}

st.title("❤️ Heart Disease Prediction")

model = load_model()

# Get user input for each selected feature
user_input = {}
for feature in selected_features:
    min_val, max_val = feature_ranges[feature]
    user_input[feature] = st.number_input(
        label=f"{feature} ({min_val} - {max_val})",
        min_value=float(min_val),
        max_value=float(max_val),
        value=float((min_val + max_val) / 2),
    )

# Convert user input to dataframe
input_df = pd.DataFrame([user_input])

# Scale input using MinMaxScaler manually (same scaling as training)
scaler = MinMaxScaler()
scaler.min_, scaler.scale_ = np.array([-min_val for min_val, _ in feature_ranges.values()]), \
                             np.array([1/(max_val - min_val) for min_val, max_val in feature_ranges.values()])
input_scaled = scaler.transform(input_df)

if st.button("Predict"):
    pred = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0][1]

    if pred == 1:
        st.error(f"⚠️ High risk of heart disease! Probability: {proba:.2f}")
    else:
        st.success(f"✅ Low risk of heart disease. Probability: {proba:.2f}")
