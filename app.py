import streamlit as st
import pandas as pd
import joblib
import json

# 1. Load the saved model and column names
model = joblib.load('churn_model.pkl')
with open('model_columns.json', 'r') as f:
    model_columns = json.load(f)

# 2. Build the Web App Interface
st.set_page_config(page_title="Telecom Churn Predictor", layout="wide")
st.title("📉 Telecom Customer Churn Predictor")
st.markdown("Enter the customer's details below to predict if they are at risk of leaving.")

# Create input columns for a clean layout
col1, col2, col3 = st.columns(3)

with col1:
    tenure = st.number_input("Tenure (Months)", min_value=0, max_value=72, value=12)
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=15.0, max_value=120.0, value=75.0)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=9000.0, value=900.0)

with col2:
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
    payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

with col3:
    senior_citizen = st.selectbox("Senior Citizen?", ["No", "Yes"])
    dependents = st.selectbox("Has Dependents?", ["No", "Yes"])
    paperless_billing = st.selectbox("Paperless Billing?", ["Yes", "No"])

# 3. The "Predict" Button Logic
if st.button("Predict Churn Risk", type="primary"):
    
    # Pack user inputs into a dictionary
    input_data = {
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
        'Partner': 0, 'Dependents': 1 if dependents == "Yes" else 0,
        'PhoneService': 1, 'PaperlessBilling': 1 if paperless_billing == "Yes" else 0,
        'gender_Male': 0, 'gender_Female': 0, # Defaulting minor features
        f'Contract_{contract}': 1,
        f'InternetService_{internet_service}': 1,
        f'PaymentMethod_{payment_method}': 1
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Reindex to match the 46 columns the model was trained on, filling missing ones with 0
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    # 4. predicting the possibilities
    # Get the raw probability of churning (Index 1)
    churn_probability = model.predict_proba(input_df)[0][1]
    
    # Apply our optimized 30% threshold!
    prediction = 1 if churn_probability >= 0.30 else 0

    # 5. Display Results
    st.markdown("---")
    if prediction == 1:
        st.error(f"🚨 **HIGH RISK**: This customer is likely to churn. (Probability: {churn_probability * 100:.1f}%)")
        st.warning("Recommendation: Offer a promotional discount or switch them to a 1-year contract.")
    else:
        st.success(f"✅ **SAFE**: This customer is likely to stay. (Probability: {churn_probability * 100:.1f}%)")