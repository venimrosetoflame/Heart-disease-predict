import streamlit as st
import numpy as np
import joblib

# Load the saved model
model_path = 'best_model_heart_disease_prediction.joblib'
model = joblib.load(model_path)

# Title and description
st.title("Heart Disease Prediction App")
st.write("Enter the patient details below to predict the likelihood of heart disease:")

# Input form
age = st.number_input("Age", min_value=0, max_value=120, value=25)
sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3], format_func=lambda x: f"Type {x}")
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=300, value=120)
chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=0, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], format_func=lambda x: "True" if x == 1 else "False")
restecg = st.selectbox("Resting Electrocardiographic Results", options=[0, 1, 2], format_func=lambda x: f"Result {x}")
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=0, max_value=250, value=150)
exang = st.selectbox("Exercise Induced Angina", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
oldpeak = st.number_input("ST Depression Induced by Exercise Relative to Rest", min_value=0.0, max_value=10.0, value=1.0)
slope = st.selectbox("Slope of the Peak Exercise ST Segment", options=[0, 1, 2], format_func=lambda x: f"Type {x}")
ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", options=[0, 1, 2, 3, 4])
thal = st.selectbox("Thalassemia", options=[0, 1, 2, 3], format_func=lambda x: f"Type {x}")

# Collect input data
input_data = np.array([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]).reshape(1, -1)

# Predict and display the result
if st.button('Predict'):
    try:
        prediction = model.predict(input_data)
        if prediction[0] == 1:
            st.success("The patient is likely to have heart disease.")
        else:
            st.success("The patient is unlikely to have heart disease.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Add footer
st.write("This is a simple prediction app for educational purposes only.")

# Deploy link and GitHub repo
st.markdown("## Links")
st.markdown("[Deployed App](link_to_your_deployed_app)")
st.markdown("[GitHub Repository](link_to_your_github_repo)")
