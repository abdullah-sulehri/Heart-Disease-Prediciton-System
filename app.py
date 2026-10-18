import streamlit as st
import pandas as pd
import pickle

# Page Configuration
st.set_page_config(
    page_title="Heart Disease Prediction System",
    page_icon="🫀",
    layout="centered"
)

# Load the trained SVC model once using cache_resource
@st.cache_resource
def load_model():
    with open('svc_trained_model.pkl', 'rb') as file:
        return pickle.load(file)

model = load_model()

# App Header
st.title("🫀 Heart Disease Prediction System")
st.markdown(
    "Enter the patient's 4 clinical attributes below to predict the presence "
    "and severity of heart disease using the trained Support Vector Classifier (SVC)."
)
st.divider()

# Input Form matching the exact 4 features from Task 2: ['age', 'sex', 'cp', 'thalach']
with st.form("heart_disease_form"):
    col1, col2 = st.columns(2)

    with col1:
        age_input = st.number_input(
            "Age (in years)",
            min_value=1.0,
            max_value=120.0,
            value=63.0,
            step=1.0
        )

        sex_label = st.selectbox(
            "Biological Sex",
            options=["Male (1.0)", "Female (0.0)"],
            index=0
        )
        sex_input = 1.0 if "Male" in sex_label else 0.0

    with col2:
        cp_label = st.selectbox(
            "Chest Pain Type (CP)",
            options=[
                "1.0 - Typical Angina",
                "2.0 - Atypical Angina",
                "3.0 - Non-anginal Pain",
                "4.0 - Asymptomatic"
            ],
            index=3
        )
        cp_input = float(cp_label.split(" - ")[0])

        thalach_input = st.number_input(
            "Max Heart Rate Achieved (Thalach)",
            min_value=50.0,
            max_value=250.0,
            value=150.0,
            step=1.0
        )

    submit_button = st.form_submit_button("Predict Diagnosis", use_container_width=True)

# Prediction Logic (Matches Step 8.2 & 8.5.1 of your Notebook)
if submit_button:
    # Construct the exact DataFrame expected by the model
    user_input = pd.DataFrame({
        'age': [age_input],
        'sex': [sex_input],
        'cp': [cp_input],
        'thalach': [thalach_input]
    })

    st.subheader("Patient Input Feature Vector")
    st.dataframe(user_input, hide_index=True, use_container_width=True)

    # Make prediction
    predicted_target = model.predict(user_input)[0]

    st.subheader("Prediction Result")
    if predicted_target == 0:
        st.success("✅ **NO HEART DISEASE (Class 0)** — The model predicts absence of heart disease.")
    elif predicted_target == 1:
        st.warning("⚠️ **HEART DISEASE PRESENT (Severity 1)**")
    elif predicted_target == 2:
        st.warning("⚠️ **HEART DISEASE PRESENT (Severity 2)**")
    elif predicted_target == 3:
        st.error("🚨 **HEART DISEASE PRESENT (Severity 3)**")
    elif predicted_target == 4:
        st.error("🚨 **HEART DISEASE PRESENT (Severity 4)**")
    else:
        st.info(f"Prediction Output: {predicted_target}")
