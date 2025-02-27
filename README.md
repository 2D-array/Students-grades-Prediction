# Student Grade Prediction App

![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Grade%20Prediction-blue)

A **Student Grade Prediction** application using **Machine Learning and Deep Learning** models, deployed with **Streamlit**.

## 🚀 Features
- Predicts student grades based on various academic and personal factors.
- Implements **Random Forest, Gradient Boosting, SVM, KNN, Logistic Regression, and Deep Learning**.
- **Streamlit** web app for user-friendly interaction.
- Optimized with **Hyperparameter Tuning (GridSearchCV)**.

## 📂 Project Structure
```
├── app.py                 # Streamlit Web App
├── Students_Grading_Dataset.csv  # Dataset
├── model_training.ipynb   # Model Training Notebook
├── random_forest.pkl      # Trained Random Forest Model
├── deep_learning_model.h5 # Trained Deep Learning Model
├── label_encoders.pkl     # Encoders for categorical features
├── scaler.pkl             # Standard Scaler for numerical features
├── requirements.txt       # Dependencies
└── README.md              # Project Documentation
```

## 📊 Dataset Overview
- **Features:** Age, Attendance, Midterm Score, Final Score, Assignments Avg, Quizzes Avg, Participation, Projects Score, Study Hours, Stress Level, Sleep Hours, etc.
- **Target:** Grade classification (A, B, C, D, F)

## 🔧 Installation & Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/student-grade-prediction.git
   cd student-grade-prediction
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## 🏗️ Model Training
- Train ML models and save the best-performing model using `model_training.ipynb`.
- Save the trained models as `.pkl` and `.h5` files.

## 🌍 Deployment
- Deploy on **Streamlit Cloud**, **Heroku**, or **AWS**.
- Ensure `random_forest.pkl`, `deep_learning_model.h5`, `label_encoders.pkl`, and `scaler.pkl` are included.

## 🎯 Results
- Model evaluation includes **Accuracy, R² Score, and Execution Time**.
- Random Forest and Deep Learning models show high accuracy.

## 📌 Future Enhancements
- Improve UI with advanced visualizations.
- Integrate more ML models.
- Deploy on cloud services.



