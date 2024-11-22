import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
from PIL import Image



# Set up page configuration
st.set_page_config(
    page_title="Credit Defaulter Predictor",
    page_icon="💳",
    layout="wide",
)

# Sidebar Navigation
st.sidebar.title("Navigation")
tabs = st.sidebar.radio("Go to:", ["Introduction", "Prediction"])

# Tab 1: Introduction
if tabs == "Introduction":
    st.title("💳 Credit Defaulter Predictor")
    st.image("neologo.jpg", use_column_width=True)
    st.markdown("""
    Welcome to the **Credit Defaulter Predictor** application! This platform leverages machine learning to help financial institutions predict potential defaulters and manage credit risks effectively.

    ### Features:
    - **Advanced ML Models**: Built with state-of-the-art algorithms for accurate predictions.
    - **User-Friendly Interface**: Easy-to-navigate interface for seamless user experience.
    - **Fast Predictions**: Get results instantly for quick decision-making.

    ### Why Use This Application?
    - Reduce credit risk and losses by identifying defaulters in advance.
    - Enhance decision-making in loan approvals.
    - Save time and effort with automated insights.

    ### How It Works:
    1. **Input Applicant Data**: Enter details such as age, income, credit score, and more in the Prediction tab.
    2. **Get Prediction**: The system will classify the applicant as a **Defaulter** or **Non-Defaulter**.
    3. **Act Accordingly**: Use the insights to make informed decisions about loan approvals.

    ### Technologies Used:
    - **Machine Learning**: Trained models to predict credit defaulters.
    - **Streamlit**: Interactive and dynamic frontend.
    - **Python**: Backend development for predictions.

    Start exploring by selecting the **Prediction** tab on the sidebar!
    """)

# Tab 2: Prediction
elif tabs == "Prediction":
    st.title("📊 Credit Defaulter Prediction")
    st.image("loan.jpg", use_column_width=True)
    st.markdown("Enter the applicant details below to predict if they are a defaulter or non-defaulter.")

    # Input fields
    age = st.number_input("Age", min_value=18, max_value=100, step=1)
    income = st.number_input("Monthly Income (in $)", min_value=0, step=1000)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, step=1)
    loan_amount = st.number_input("Loan Amount (in $)", min_value=0.0, step=1000.0)
    account_balance = st.number_input("Account Balance (in $)", min_value=0.0, step=1000.0)
    transaction_count = st.number_input("Monthly Transactions Count", min_value=0, step=1)
    job_type = st.selectbox(
        "Job Type",
        options=["Entry-Level", "Executive", "IT Professional", "Managerial", "Skilled Worker", "Other"],
    )
    loan_term = st.selectbox("Loan Term", options=[12,24,36,48,60,72])
    # Job encoding
    job_mapping = {
        "Entry-Level": 0,
        "Executive": 1,
        "IT Professional": 2,
        "Managerial": 3,
        "Skilled Worker": 4,
        "Other": 5,
    }
    job_encoded = job_mapping[job_type]

    # Load pre-trained model and scaler
    @st.cache_resource()
    def load_models():
        model = load("Random_Forest_model.joblib")
        scaler = load("scaler_model1.joblib")
        return model, scaler

    model, scaler = load_models()

    # Prediction logic
    if st.button("Predict"):
        input_data = pd.DataFrame(
            {
                "Age": [age],
                "Income": [income],
                "Credit_Score": [credit_score],
                "Loan_Amount": [loan_amount],
                "Loan_Term" : [loan_term],
                "Transaction_Count": [transaction_count],
                "Account_Balance": [account_balance],
                "Job": [job_encoded],
            }
        )
        # Scale continuous features
        continuous_features = ["Age", "Income", "Credit_Score", "Loan_Amount", "Account_Balance", "Transaction_Count"]
        input_data[continuous_features] = scaler.transform(input_data[continuous_features])

        # Get prediction
        prediction = model.predict(input_data)
        result = "Non-Defaulter" if prediction[0] == 0 else "Defaulter"

        # Display the result
        st.subheader("Prediction Result")
        if result == "Non-Defaulter":
            st.success(f"The applicant is predicted to be a **{result}** ✅.")
        else:
            st.error(f"The applicant is predicted to be a **{result}** ❌.")

    st.info("Ensure all input fields are correctly filled before predicting.")

# Footer
st.markdown("---")
st.markdown("**Credit Defaulter Predictor App © 2024**")