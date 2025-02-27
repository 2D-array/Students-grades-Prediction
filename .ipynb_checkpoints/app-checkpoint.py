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

def preprocess_input(data):
    # Encode categorical data
    for col in categorical_cols:
        data[col] = label_encoders[col].transform([data[col]])[0]
    
    # Convert to DataFrame
    df = pd.DataFrame([data])
    
    # Scale numerical values
    df[numerical_cols] = scaler.transform(df[numerical_cols])
    
    return df

st.title("📊 Student Grade Prediction App")
st.write("Enter student details to predict their grade")

# Collect user input
data = {}
for col in categorical_cols:
    data[col] = st.selectbox(f"Select {col}", label_encoders[col].classes_)
for col in numerical_cols:
    data[col] = st.number_input(f"Enter {col}", min_value=0.0, max_value=100.0, step=0.1)

if st.button("Predict Grade"):
    input_data = preprocess_input(data)
    
    # Predict with Random Forest
    rf_pred = rf_model.predict(input_data)[0]
    predicted_grade_rf = label_encoders['Grade'].inverse_transform([rf_pred])[0]
    
    # Predict with Deep Learning
    dl_pred = np.argmax(dl_model.predict(input_data), axis=1)[0]
    predicted_grade_dl = label_encoders['Grade'].inverse_transform([dl_pred])[0]
    
    st.success(f"📌 Random Forest Predicted Grade: {predicted_grade_rf}")
    st.success(f"📌 Deep Learning Predicted Grade: {predicted_grade_dl}")
