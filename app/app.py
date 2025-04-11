import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Loan Interest Rate Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model and features (SHAP explainer will be recreated dynamically)
model = joblib.load("app/model.pkl")
features_list = joblib.load("app/features_list.pkl")

# Recreate SHAP explainer dynamically
explainer = shap.Explainer(model)

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

# Sidebar: user input
st.sidebar.header("Input Features")
loan_amnt = st.sidebar.slider("Loan Amount", 500, 40000, 10000, help="Total loan amount requested")
installment = st.sidebar.slider("Installment Amount", 50, 2000, 400, help="Monthly payment towards the loan")
annual_inc = st.sidebar.slider("Annual Income", 10000, 250000, 60000, help="Your annual income")
revol_util = st.sidebar.slider("Revolving Credit Utilization (%)", 0.0, 150.0, 50.0, help="Credit utilization ratio")
total_rec_int = st.sidebar.slider("Total Received Interest", 0.0, 25000.0, 1000.0, help="Total interest received so far")
inq_last_6mths = st.sidebar.slider("Inquiries in Last 6 Months", 0, 20, 2, help="Hard inquiries on credit report")
term = st.sidebar.selectbox("Loan Term", ['36 months', '60 months'], help="Loan repayment duration")
purpose = st.sidebar.selectbox("Purpose of Loan", list(purpose_mapping.keys()), help="Reason for the loan")
verification_status = st.sidebar.selectbox("Verification Status", ['Verified', 'Not Verified', 'Source Verified'], help="Income verification status")
theme = st.sidebar.radio("Choose Theme", ["Light", "Dark"], help="Switch between light and dark mode")

# Apply theme
if theme == "Dark":
    st.markdown("""
        <style>
        .stApp { background-color: #0e1117; color: #f1f5f9; }
        .css-1d391kg { color: #e0f2fe; }
        </style>
    """, unsafe_allow_html=True)

# Prepare input
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

input_df = pd.DataFrame(user_input, index=[0])
encoded_input = pd.get_dummies(input_df)
encoded_input = encoded_input.reindex(columns=features_list, fill_value=0)

# Prediction and SHAP
prediction = model.predict(encoded_input)[0]
shap_values = explainer(encoded_input)

# Layout
st.title("Loan Interest Rate Explainability")

# Timestamp
timestamp = datetime.now().strftime("%b %d %Y • %I:%M %p")
st.caption(f"Last Updated: {timestamp}")

# KPI Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Predicted Interest Rate", f"{prediction:.2f}%")
col2.metric("Loan Purpose", purpose)
col3.metric("Annual Income", f"${annual_inc:,.0f}")

# Encoded Input
with st.expander("Encoded Input Data"):
    st.dataframe(encoded_input.style.format(precision=2))

# SHAP Grid Plots
st.subheader("Feature Contribution Visuals")
with st.container():
    shap1, shap2 = st.columns(2)
    with shap1:
        st.markdown("**SHAP Waterfall Plot**")
        shap.plots.waterfall(shap_values[0], max_display=8, show=False)
        st.pyplot(plt.gcf()); plt.clf()
    with shap2:
        st.markdown("**SHAP Bar Plot**")
        shap.plots.bar(shap_values, max_display=8, show=False)
        st.pyplot(plt.gcf()); plt.clf()

    viz1, viz2 = st.columns(2)
    with viz1:
        st.markdown("**Interest Rate Distribution**")
        df = pd.read_csv("data/interest_rate_df_engineered.csv")
        fig, ax = plt.subplots()
        sns.histplot(df['int_rate'], bins=30, kde=True, ax=ax)
        st.pyplot(fig)

    with viz2:
        st.markdown("**Feature Correlation Heatmap**")
        numeric_df = df.select_dtypes(include='number')
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(numeric_df.corr(), cmap="viridis", ax=ax)
        st.pyplot(fig)

# Supplementary Visualizations
st.subheader("Additional Visual Insights")
if st.checkbox("Enable More Visual Insights"):
    options = ["Boxplot by Purpose", "Histogram of Income", "Heatmap"]
    vis_input = st.text_input("Search Visualization Type")
    vis_choice = st.selectbox("Choose Visualization", [o for o in options if vis_input.lower() in o.lower()])

    if vis_choice == "Boxplot by Purpose":
        fig, ax = plt.subplots()
        sns.boxplot(x="purpose", y="int_rate", data=df, ax=ax)
        ax.set_title("Interest Rate by Purpose")
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)
    elif vis_choice == "Histogram of Income":
        fig, ax = plt.subplots()
        sns.histplot(df['annual_inc'], bins=30, kde=True, ax=ax)
        ax.set_title("Income Distribution")
        st.pyplot(fig)
    elif vis_choice == "Heatmap":
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(numeric_df.corr(), cmap="viridis", ax=ax)
        st.pyplot(fig)

# Batch Prediction Upload
st.subheader("Batch Prediction Analysis")
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    batch_df = pd.read_csv(uploaded_file)
    encoded_batch = pd.get_dummies(batch_df)
    encoded_batch = encoded_batch.reindex(columns=features_list, fill_value=0)
    batch_predictions = model.predict(encoded_batch)
    batch_df['Predicted Interest Rate'] = batch_predictions

    st.subheader("Batch KPIs")
    avg_rate = batch_predictions.mean()
    top_purpose = batch_df['purpose'].mode()[0] if 'purpose' in batch_df.columns else "N/A"
    high_interest_count = (batch_df['Predicted Interest Rate'] > 25).sum()

    k1, k2, k3 = st.columns(3)
    k1.metric("Avg Predicted Rate", f"{avg_rate:.2f}%")
    k2.metric("Top Loan Purpose", top_purpose)
    k3.metric("Loans > 25%", high_interest_count)

    st.subheader("Distribution of Predicted Rates")
    fig3, ax3 = plt.subplots()
    sns.histplot(batch_predictions, bins=20, kde=True, ax=ax3)
    st.pyplot(fig3)

    csv = batch_df.to_csv(index=False).encode()
    st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")
