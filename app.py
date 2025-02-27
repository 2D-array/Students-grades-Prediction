import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import load_model

# Load trained models (Random Forest & Deep Learning)
with open("random_forest.pkl", "rb") as f:
    rf_model = pickle.load(f)
dl_model = load_model("deep_learning_model.h5")

# Load encoders and scaler
with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Define categorical and numerical columns
categorical_cols = ['Gender', 'Department', 'Extracurricular_Activities', 'Internet_Access_at_Home',
                    'Parent_Education_Level', 'Family_Income_Level']
numerical_cols = ['Age', 'Attendance (%)', 'Midterm_Score', 'Final_Score', 'Assignments_Avg',
                  'Quizzes_Avg', 'Participation_Score', 'Projects_Score', 'Total_Score',
                  'Study_Hours_per_Week', 'Stress_Level (1-10)', 'Sleep_Hours_per_Night']

# Define dummy data for quick testing
dummy_data = {
    "Gender": "Female",
    "Department": "Business",
    "Extracurricular_Activities": "No",
    "Internet_Access_at_Home": "No",
    "Parent_Education_Level": "Bachelor's",
    "Family_Income_Level": "High",
    "Age": 19.0,
    "Attendance (%)": 7.9,
    "Midterm_Score": 8.5,
    "Final_Score": 14.0,
    "Assignments_Avg": 4.0,
    "Quizzes_Avg": 3.8,
    "Participation_Score": 4.7,
    "Projects_Score": 4.5,
    "Total_Score": 9.6,
    "Study_Hours_per_Week": 8.0,
    "Stress_Level (1-10)": 5.0,
    "Sleep_Hours_per_Night": 7.8
}

# Preprocessing function to prepare input data for prediction
def preprocess_input(data):
    try:
        # Encode categorical data
        for col in categorical_cols:
            if data[col] not in label_encoders[col].classes_:
                raise ValueError(f"❌ Unexpected category '{data[col]}' in column '{col}'")
            data[col] = label_encoders[col].transform([data[col]])[0]

        # Convert to DataFrame
        df = pd.DataFrame([data])

        # Scale numerical values
        df[numerical_cols] = scaler.transform(df[numerical_cols])

        # Align feature order with trained model
        df = df[rf_model.feature_names_in_]

        return df
    except Exception as e:
        st.error(f"🚨 Data preprocessing error: {e}")
        return None

# Streamlit app UI
st.title("📊 Student Grade Prediction App")
st.write("Enter student details to predict their grade using Random Forest and Deep Learning models.")

# Load dummy input button
if st.button("Load Dummy Input"):
    for key, value in dummy_data.items():
        st.session_state[key] = value

# Collect user input with organized sections
data = {}

### Personal Details Section
st.subheader("Personal Details")
col1, col2, col3 = st.columns(3)
with col1:
    data["Gender"] = st.selectbox("Select Gender", label_encoders["Gender"].classes_, key="Gender")
with col2:
    data["Age"] = st.number_input("Enter Age", min_value=0.0, max_value=100.0, step=0.1, key="Age")
with col3:
    data["Department"] = st.selectbox("Select Department", label_encoders["Department"].classes_, key="Department")

### Academic Performance Section
st.subheader("Academic Performance")
col1, col2, col3 = st.columns(3)
with col1:
    data["Attendance (%)"] = st.number_input("Enter Attendance (%)", min_value=0.0, max_value=100.0, step=0.1, key="Attendance (%)", help="Percentage of classes attended")
with col2:
    data["Midterm_Score"] = st.number_input("Enter Midterm Score", min_value=0.0, max_value=100.0, step=0.1, key="Midterm_Score")
with col3:
    data["Final_Score"] = st.number_input("Enter Final Score", min_value=0.0, max_value=100.0, step=0.1, key="Final_Score")

col1, col2, col3 = st.columns(3)
with col1:
    data["Assignments_Avg"] = st.number_input("Enter Assignments Average", min_value=0.0, max_value=100.0, step=0.1, key="Assignments_Avg")
with col2:
    data["Quizzes_Avg"] = st.number_input("Enter Quizzes Average", min_value=0.0, max_value=100.0, step=0.1, key="Quizzes_Avg")
with col3:
    data["Participation_Score"] = st.number_input("Enter Participation Score", min_value=0.0, max_value=100.0, step=0.1, key="Participation_Score")

col1, col2 = st.columns(2)
with col1:
    data["Projects_Score"] = st.number_input("Enter Projects Score", min_value=0.0, max_value=100.0, step=0.1, key="Projects_Score")
with col2:
    data["Total_Score"] = st.number_input("Enter Total Score", min_value=0.0, max_value=100.0, step=0.1, key="Total_Score")

### Extracurricular Activities and Lifestyle Section
st.subheader("Extracurricular Activities and Lifestyle")
col1, col2 = st.columns(2)
with col1:
    data["Extracurricular_Activities"] = st.selectbox("Select Extracurricular Activities", label_encoders["Extracurricular_Activities"].classes_, key="Extracurricular_Activities")
with col2:
    data["Internet_Access_at_Home"] = st.selectbox("Select Internet Access at Home", label_encoders["Internet_Access_at_Home"].classes_, key="Internet_Access_at_Home")

col1, col2, col3 = st.columns(3)
with col1:
    data["Study_Hours_per_Week"] = st.number_input("Enter Study Hours per Week", min_value=0.0, max_value=100.0, step=0.1, key="Study_Hours_per_Week")
with col2:
    data["Stress_Level (1-10)"] = st.number_input("Enter Stress Level (1-10)", min_value=1.0, max_value=10.0, step=0.1, key="Stress_Level (1-10)", help="Self-reported stress level from 1 to 10")
with col3:
    data["Sleep_Hours_per_Night"] = st.number_input("Enter Sleep Hours per Night", min_value=0.0, max_value=24.0, step=0.1, key="Sleep_Hours_per_Night")

### Family Background Section
st.subheader("Family Background")
col1, col2 = st.columns(2)
with col1:
    data["Parent_Education_Level"] = st.selectbox("Select Parent Education Level", label_encoders["Parent_Education_Level"].classes_, key="Parent_Education_Level")
with col2:
    data["Family_Income_Level"] = st.selectbox("Select Family Income Level", label_encoders["Family_Income_Level"].classes_, key="Family_Income_Level")

# Predict button and prediction logic
if st.button("Predict Grade"):
    input_data = preprocess_input(data)
    if input_data is not None:
        # Predict with Random Forest
        rf_pred = rf_model.predict(input_data)[0]
        predicted_grade_rf = label_encoders['Grade'].inverse_transform([rf_pred])[0]
        
        # Predict with Deep Learning
        dl_pred = np.argmax(dl_model.predict(input_data), axis=1)[0]
        predicted_grade_dl = label_encoders['Grade'].inverse_transform([dl_pred])[0]
        
        # Display predictions
        st.success(f"📌 Random Forest Predicted Grade: {predicted_grade_rf}")
        st.success(f"📌 Deep Learning Predicted Grade: {predicted_grade_dl}")