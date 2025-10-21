import streamlit as st
import pickle
import numpy as np

# Load the trained XGBoost model
with open('xgboost_model_frailty.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Set the title of the web app
st.markdown("<h1 style='text-align: center; font-family: Arial, sans-serif; color: #2c3e50;'>Stroke Frailty Prediction Model</h1>", unsafe_allow_html=True)

# Sidebar header for input features
st.sidebar.header("Enter Patient Features")

# Input fields for each feature in the sidebar
education_level = st.sidebar.selectbox("Education Level", ["Primary school or below", "Middle school", "High school or above"])
polypharmacy = st.sidebar.selectbox("Polypharmacy (Multiple Medications)", ["No", "Yes"])
personal_income = st.sidebar.number_input("Personal Income", min_value=0, value=10000)
smoking = st.sidebar.selectbox("Smoking", ["No", "Yes"])
health_insurance = st.sidebar.selectbox("Health Insurance", ["No", "Yes"])
mobility_impairment = st.sidebar.selectbox("Mobility Impairment", ["No", "Yes"])
adl_impairment = st.sidebar.selectbox("ADL Impairment (Activities of Daily Living)", ["No", "Yes"])
number_of_children = st.sidebar.number_input("Number of Children", min_value=0, value=2)
hearing = st.sidebar.selectbox("Hearing", ["Very Poor", "Poor", "Average", "Good", "Excellent"])
life_satisfaction = st.sidebar.selectbox("Life Satisfaction", ["Very Dissatisfied", "Dissatisfied", "Average", "Satisfied", "Very Satisfied"])

# Transform the user input into a format the model can accept
education_level_mapping = {"Primary school or below": 1, "Middle school": 2, "High school or above": 3}
polypharmacy_mapping = {"No": 0, "Yes": 1}
smoking_mapping = {"No": 0, "Yes": 1}
health_insurance_mapping = {"No": 0, "Yes": 1}
mobility_impairment_mapping = {"No": 0, "Yes": 1}
adl_impairment_mapping = {"No": 0, "Yes": 1}
hearing_mapping = {"Very Poor": 1, "Poor": 2, "Average": 3, "Good": 4, "Excellent": 5}
life_satisfaction_mapping = {"Very Dissatisfied": 1, "Dissatisfied": 2, "Average": 3, "Satisfied": 4, "Very Satisfied": 5}

X_input = np.array([
    education_level_mapping[education_level],
    polypharmacy_mapping[polypharmacy],
    personal_income,
    smoking_mapping[smoking],
    health_insurance_mapping[health_insurance],
    mobility_impairment_mapping[mobility_impairment],
    adl_impairment_mapping[adl_impairment],
    number_of_children,
    hearing_mapping[hearing],
    life_satisfaction_mapping[life_satisfaction]
]).reshape(1, -1)

# Main section for displaying results
st.markdown("<h2 style='text-align: center; font-family: Arial, sans-serif; color: #34495e;'>Risk Prediction</h2>", unsafe_allow_html=True)

# When the user clicks the button to make a prediction
if st.button('Make Prediction'):
    # Use the model to make predictions
    prediction = model.predict(X_input)
    probability = model.predict_proba(X_input)[0, 1]  # Probability of the "frailty" class

    # Enhanced prediction result display with modern design
    if prediction[0] == 1:
        st.markdown("<div style='background-color: #f8d7da; padding: 20px; border-radius: 10px;'>", unsafe_allow_html=True)
        st.markdown(f"### **Prediction: Frailty**", unsafe_allow_html=True)
        st.markdown(f"**Probability: {probability * 100:.2f}%**", unsafe_allow_html=True)
        st.progress(int(probability * 100))  # Display a progress bar for the probability
        st.success("⚠️ **The patient is predicted to be frail.** Consider initiating rehabilitation and regular monitoring.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='background-color: #d4edda; padding: 20px; border-radius: 10px;'>", unsafe_allow_html=True)
        st.markdown(f"### **Prediction: Non-Frailty**", unsafe_allow_html=True)
        st.markdown(f"**Probability: {(1 - probability) * 100:.2f}%**", unsafe_allow_html=True)
        st.progress(int((1 - probability) * 100))  # Display a progress bar for the probability
        st.info("✅ **The patient is predicted to be non-frail.** Continue regular follow-ups and maintain a healthy lifestyle.")
        st.markdown("</div>", unsafe_allow_html=True)

    # Provide specific clinical suggestions based on stroke frailty prediction
    st.markdown("<h3 style='text-align: center; font-family: Arial, sans-serif; color: #5f6368;'>Clinical Suggestions</h3>", unsafe_allow_html=True)
    if prediction[0] == 1:
        st.write("1. Initiate early rehabilitation to improve mobility.")
        st.write("2. Monitor cognitive function regularly.")
        st.write("3. Ensure proper nutrition and hydration.")
        st.write("4. Regularly assess the risk of secondary complications, including stroke recurrence.")
    else:
        st.write("The patient is not frail, but continue to encourage regular exercise and health monitoring.")
        st.write("Maintain a balanced diet and manage chronic conditions such as hypertension or diabetes.")
