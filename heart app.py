import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.graph_objects as go
import io
import base64
from datetime import datetime

# Custom CSS for improved UI
st.markdown("""
<style>
    .main {background-color: #f5f5f5;}
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stTextInput>div>input, .stNumberInput>div>input {
        border-radius: 5px;
        border: 1px solid #ccc;
    }
    .stSelectbox>div>div {
        border-radius: 5px;
    }
    .report-container {
        background-color: #e8f4f8;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .header {
        color: #2c3e50;
        font-size: 2.5em;
        text-align: center;
        margin-bottom: 20px;
    }
    .subheader {
        color: #34495e;
        font-size: 1.5em;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Load the pre-trained model
try:
    model_path = r"heart_disease.sav"
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Model file 'heart_disease.sav' not found. Please ensure the file exists.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# Load data for fitting scaler
try:
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    data = pd.read_csv(url, names=column_names, na_values="?")
    data = data.dropna()
    scaler = MinMaxScaler()
    scaler.fit(data[column_names[:-1]])
except Exception as e:
    st.error(f"Error loading dataset for scaler: {str(e)}")
    st.stop()

# Prediction function
def heart_disease_prediction(input_data):
    try:
        input_data_as_numpy_array = np.asarray(input_data, dtype=float)
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
        scaled_input = scaler.transform(input_data_reshaped)
        prediction = model.predict(scaled_input)[0]
        probability = model.decision_function(scaled_input)[0] if hasattr(model, 'decision_function') else None
        result = "The patient has heart disease (High Risk)." if prediction == 1 else "The patient does not have heart disease (Low Risk)."
        return result, probability
    except ValueError as e:
        return f"Error: Please ensure all inputs are valid numbers. Details: {str(e)}", None
    except Exception as e:
        return f"Prediction error: {str(e)}", None

# Generate downloadable report
def generate_report(input_data, prediction, probability):
    buffer = io.StringIO()
    buffer.write("Heart Disease Prediction Report\n")
    buffer.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    buffer.write("Patient Data:\n")
    labels = ['Age', 'Sex', 'Chest Pain Type', 'Resting Blood Pressure', 'Serum Cholesterol', 
              'Fasting Blood Sugar', 'Resting ECG', 'Maximum Heart Rate', 'Exercise Induced Angina', 
              'ST Depression', 'Slope of Peak Exercise ST', 'Major Vessels (Fluoroscopy)', 'Thalassemia']
    descriptive_values = [
        input_data[0],  # Age
        'Female' if input_data[1] == 0 else 'Male',  # Sex
        ['Typical angina', 'Atypical angina', 'Non-anginal pain', 'Asymptomatic'][int(input_data[2])],  # cp
        input_data[3],  # trestbps
        input_data[4],  # chol
        'No (>120 mg/dl)' if input_data[5] == 0 else 'Yes (≤120 mg/dl)',  # fbs
        ['Normal', 'ST-T wave abnormality', 'Left ventricular hypertrophy'][int(input_data[6])],  # restecg
        input_data[7],  # thalach
        'No' if input_data[8] == 0 else 'Yes',  # exang
        input_data[9],  # oldpeak
        ['Upsloping', 'Flat', 'Downsloping'][int(input_data[10])],  # slope
        input_data[11],  # ca
        ['Normal', 'Fixed defect', 'Reversable defect'][int(input_data[12]) - 1]  # thal
    ]
    for label, value in zip(labels, descriptive_values):
        buffer.write(f"{label}: {value}\n")
    buffer.write(f"\nPrediction: {prediction}\n")
    if probability is not None:
        buffer.write(f"Confidence Score: {probability:.4f}\n")
    return buffer.getvalue()

def main():
    st.markdown('<div class="header">Heart Disease Prediction Web App</div>', unsafe_allow_html=True)
    st.markdown("""
    This application predicts the likelihood of heart disease using a pre-trained Support Vector Classifier (SVC) model. 
    Enter the patient's medical details below to get a prediction, visualize the confidence score, and download a report.
    """)

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Prediction", "About"])

    if page == "Prediction":
        st.markdown('<div class="subheader">Enter Patient Medical Data</div>', unsafe_allow_html=True)
        
        # Input fields with descriptive dropdowns
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input('Age (years)', min_value=0, max_value=120, value=50, step=1)
            sex_options = ['Female', 'Male']
            sex = st.selectbox('Sex', sex_options)
            sex = 0 if sex == 'Female' else 1
            cp_options = ['Typical angina', 'Atypical angina', 'Non-anginal pain', 'Asymptomatic']
            cp = st.selectbox('Chest Pain Type', cp_options)
            cp = cp_options.index(cp)
            trestbps = st.number_input('Resting Blood Pressure (mm Hg)', min_value=0, max_value=300, value=120, step=1)
            chol = st.number_input('Serum Cholesterol (mg/dl)', min_value=0, max_value=600, value=200, step=1)
            fbs_options = ['No (>120 mg/dl)', 'Yes (≤120 mg/dl)']
            fbs = st.selectbox('Fasting Blood Sugar', fbs_options)
            fbs = 0 if fbs.startswith('No') else 1
            restecg_options = ['Normal', 'ST-T wave abnormality', 'Left ventricular hypertrophy']
            restecg = st.selectbox('Resting ECG', restecg_options)
            restecg = restecg_options.index(restecg)
        
        with col2:
            thalach = st.number_input('Maximum Heart Rate Achieved (bpm)', min_value=0, max_value=250, value=150, step=1)
            exang_options = ['No', 'Yes']
            exang = st.selectbox('Exercise Induced Angina', exang_options)
            exang = 0 if exang == 'No' else 1
            oldpeak = st.number_input('ST Depression Induced by Exercise (mm)', min_value=0.0, max_value=10.0, value=0.0, step=0.1)
            slope_options = ['Upsloping', 'Flat', 'Downsloping']
            slope = st.selectbox('Slope of Peak Exercise ST Segment', slope_options)
            slope = slope_options.index(slope)
            ca = st.number_input('Number of Major Vessels Colored by Fluoroscopy', min_value=0, max_value=3, value=0, step=1)
            thal_options = ['Normal', 'Fixed defect', 'Reversable defect']
            thal = st.selectbox('Thalassemia', thal_options)
            thal = thal_options.index(thal) + 1  # Adjust for 1-based indexing (1, 2, 3)
        
        input_data = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

        # Prediction button
        if st.button('Predict Heart Disease'):
            prediction, probability = heart_disease_prediction(input_data)
            if "Error" in prediction:
                st.error(prediction)
            else:
                st.success(prediction)
                if probability is not None:
                    # Gauge chart for confidence score
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=probability,
                        title={'text': "Prediction Confidence Score"},
                        gauge={
                            'axis': {'range': [-3, 3]},  # SVC decision function range
                            'bar': {'color': "#4CAF50" if prediction.startswith("The patient does not") else "#FF4B4B"},
                            'steps': [
                                {'range': [-3, 0], 'color': "lightgreen"},
                                {'range': [0, 3], 'color': "lightcoral"}
                            ],
                            'threshold': {
                                'line': {'color': "black", 'width': 4},
                                'thickness': 0.75,
                                'value': 0
                            }
                        }
                    ))
                    st.plotly_chart(fig)

                # Generate and offer downloadable report
                report = generate_report(input_data, prediction, probability)
                st.markdown('<div class="subheader">Download Prediction Report</div>', unsafe_allow_html=True)
                b64 = base64.b64encode(report.encode()).decode()
                href = f'<a href="data:file/txt;base64,{b64}" download="heart_disease_prediction_report.txt">Download Report</a>'
                st.markdown(href, unsafe_allow_html=True)

        # Display input data for debugging
        if st.checkbox("Show Input Data"):
            st.markdown('<div class="report-container">', unsafe_allow_html=True)
            st.write("**Input Data:**")
            input_df = pd.DataFrame([input_data], columns=column_names[:-1])
            st.dataframe(input_df)
            st.markdown('</div>', unsafe_allow_html=True)

    elif page == "About":
        st.markdown('<div class="subheader">About This Application</div>', unsafe_allow_html=True)
        st.markdown("""
        **Heart Disease Prediction Web App** is designed to predict the likelihood of heart disease using a pre-trained Support Vector Classifier (SVC) model. 
        The model was trained on the UCI Heart Disease dataset and saved as `heart_disease.sav`.

        ### Features:
        - **User-Friendly Inputs**: Enter patient data through intuitive dropdowns and numeric inputs with descriptive labels.
        - **Prediction Confidence**: Visualize the confidence score of predictions using an interactive gauge chart.
        - **Downloadable Reports**: Generate and download a detailed report of the input data and prediction results.
        - **Data Validation**: Ensures all inputs are valid to prevent errors during prediction.
        - **Responsive Design**: Clean and modern UI with custom styling for a better user experience.

        ### Model Details:
        - **Model Type**: Support Vector Classifier (SVC)
        - **Training Data**: UCI Heart Disease dataset
        - **Features Used**: Age, Sex, Chest Pain Type, Resting Blood Pressure, Serum Cholesterol, Fasting Blood Sugar, Resting ECG, Maximum Heart Rate, Exercise Induced Angina, ST Depression, Slope of Peak Exercise ST, Major Vessels (Fluoroscopy), Thalassemia
        - **Preprocessing**: Data scaled using MinMaxScaler

        For any issues or feedback, please contact the developer.
        """)

if __name__ == '__main__':

    main()

