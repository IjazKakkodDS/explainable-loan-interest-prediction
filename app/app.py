import streamlit as st
import pandas as pd
import joblib

# Loading the trained model and feature list
model = joblib.load("model.pkl")
features_list = joblib.load("features_list.pkl") # for reindexing

# Mapping the purpose variable (key-value pairs)
purpose_mapping = {
    'Debt Consolidation': 'debt_consolidation',
    'Other': 'other',
    'Credit Card': 'credit_card',
    'Home Improvement': 'home_improvement',
    'Small Business': 'small_business',
    'Major Purchase': 'major_purchase',
    'House': 'house',
    'Moving': 'moving',
    'Medical': 'medical',
    'Car': 'car',
    'Vacation': 'vacation',
    'Renewable Energy': 'renewable_energy',
    'Wedding': 'wedding',
    'Educational': 'educational'
}

# user inputs
def user_info():
    st.title("Enter the details to get interest rate")

    # Collect user input
    loan_amnt = st.number_input("Enter Loan Amount")
    installment = st.number_input("Enter Installment Amount")
    annual_inc = st.number_input("Enter Annual Income")
    revol_util = st.number_input("Enter Revolving Credit Utilization")
    total_rec_int = st.number_input("Enter Total Received Interest")
    inq_last_6mths = st.number_input("Number of Inquiries in Last 6 Months", min_value=0, max_value=100)
    term = st.selectbox("Loan Term", ['36 months', '60 months'])
    purpose = st.selectbox("Purpose Of Loan", list(purpose_mapping.keys()))
    verification_status = st.selectbox("Verification Status", ['Verified', 'Not Verified', 'Source Verified'])

    if st.button("Submit"):
        user_input = {
            "loan_amnt": loan_amnt,
            "installment": installment,
            "annual_inc": annual_inc,
            "revol_util": revol_util,
            "total_rec_int": total_rec_int,
            "inq_last_6mths": inq_last_6mths,
            "term": term.lower().replace(' ', '_'),
            "purpose": purpose_mapping[purpose],
            "verification_status": verification_status.lower().replace(' ', '_')
        }

        # converting user input into dataframe
        input_data = pd.DataFrame(user_input, index=[0])
        
        # Encoding user input
        encode_user_input = pd.get_dummies(input_data)

        # Reindex to match the model's expected feature list
        encode_user_input = encode_user_input.reindex(columns=features_list, fill_value=0)

        # Debug: Check encoded user input
        st.write("Encoded User Input:")
        st.write(encode_user_input)

        # Making Predictions
        input_pred = model.predict(encode_user_input)[0]

        # Debug: Check the prediction
        st.write(f"The Predicted interest rate is {input_pred}")

user_info()
