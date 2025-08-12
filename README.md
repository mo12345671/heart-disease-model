# Heart Disease Prediction Model

A machine learning project to predict the likelihood of heart disease using clinical and demographic patient data.  
This project includes data preprocessing, exploratory data analysis (EDA), feature selection, scaling, and model evaluation using metrics such as **Accuracy** and **ROC AUC Score**.

---

## 📌 Project Overview
Heart disease is one of the leading causes of death worldwide.  
By analyzing patient data and medical attributes, this project aims to build a predictive model to assist in early diagnosis.

The workflow includes:
- Data loading & cleaning
- Exploratory Data Analysis (EDA) with correlation and visualization
- Feature scaling and selection
- Model training and evaluation (Support Vector Classifier & other algorithms)
- ROC AUC and accuracy measurement
- Optional: ROC curve visualization

---

## 📂 Dataset
The dataset used is the **Heart Disease Dataset** (commonly available from the UCI Machine Learning Repository or Kaggle).

**Features include:**
- `age` - Age of the patient
- `sex` - Gender (1 = male, 0 = female)
- `cp` - Chest pain type
- `trestbps` - Resting blood pressure
- `chol` - Serum cholesterol (mg/dl)
- `fbs` - Fasting blood sugar > 120 mg/dl
- `restecg` - Resting ECG results
- `thalach` - Maximum heart rate achieved
- `exang` - Exercise-induced angina
- `oldpeak` - ST depression induced by exercise
- `slope` - Slope of peak exercise ST segment
- `ca` - Number of major vessels colored by fluoroscopy
- `thal` - Thalassemia type
- `target` - Presence of heart disease (1 = yes, 0 = no)

---

## 🛠 Installation & Requirements
Make sure you have Python installed (3.8+ recommended).

```bash
# Clone repository
git clone https://github.com/yourusername/heart-disease-model.git
cd heart-disease-model

# Install dependencies
pip install -r requirements.txt
