import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Page Configuration
st.set_page_config(
    page_title="Telco Churn Predictor",
    layout="wide"
)

# 2. Modern Purple Custom CSS Styling
custom_css = """
<style>
/* Background */
.stApp {
    background: linear-gradient(135deg, #0f0c1b 0%, #1a102f 50%, #2a1240 100%);
    color: #ffffff;
}

/* Headings */
h1, h2, h3, h4 {
    color: #e0b0ff !important;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

/* Fix Input Labels Visibility */
label, [data-testid="stWidgetLabel"] p, .stMarkdown p {
    color: #e9d5ff !important;
    font-weight: 600 !important;
    font-size: 15px !important;
}

/* Button Styling */
.stButton>button {
    background: linear-gradient(90deg, #7b2cbf 0%, #9d4edd 100%);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 12px 28px;
    font-size: 18px;
    font-weight: bold;
    width: 100%;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    background: linear-gradient(90deg, #9d4edd 0%, #c77dff 100%);
    box-shadow: 0px 4px 15px rgba(157, 78, 221, 0.4);
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# 3. Load Saved Artifacts Safely
@st.cache_resource
def load_assets():
    model_path = os.path.join('models', 'logistic_regression_model.pkl')
    scaler_path = os.path.join('models', 'scaler.pkl')
    columns_path = os.path.join('models', 'model_columns.pkl')
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    columns = joblib.load(columns_path)
    return model, scaler, columns

try:
    model, scaler, model_columns = load_assets()
except Exception as e:
    st.error(f"Error loading model assets. Please verify 'models' folder contents: {e}")

# Header Section
st.title("Customer Churn Prediction Intelligence")
st.markdown("Analyze customer profile parameters to predict churn risk and generate automated retention strategies.")
st.markdown("---")

# Input Form Structured in Columns (Clean & Balanced UI)
with st.container():
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Customer & Demographics")
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Has Partner", ["No", "Yes"])
        dependents = st.selectbox("Has Dependents", ["No", "Yes"])
        tenure = st.number_input("Tenure (Months)", min_value=1, max_value=72, value=12)

    with col2:
        st.subheader("Core Services")
        phone_service = st.selectbox("Phone Service", ["No", "Yes"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes"])

    with col3:
        st.subheader("Account & Billing")
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=65.0)
        total_charges = st.number_input("Total Charges ($)", min_value=18.0, max_value=8500.0, value=780.0)

st.markdown("---")

# Recommendation Engine Logic
def get_recommendations(contract_type, payment_meth, tech_sup, online_sec):
    recs = []
    if contract_type == "Month-to-month":
        recs.append("Offer a 1-year or 2-year contract with a 15% promotional discount to increase commitment.")
    if payment_meth == "Electronic check":
        recs.append("Encourage switching to Automatic Payment (Credit Card/Bank Transfer) with a $5 bill credit.")
    if tech_sup == "No":
        recs.append("Provide 3 months of complimentary VIP Tech Support to address technical dissatisfaction.")
    if online_sec == "No":
        recs.append("Include free Online Security features to improve service value proposition.")
    if len(recs) == 0:
        recs.append("Customer risk is low. Recommend inclusion in loyalty rewards program.")
    return recs

# Prediction Action
if st.button("Analyze Churn Risk"):
    # حساب فئات الـ tenure التفصيلية بدقة لتطابق الـ Model
    t_3_6 = 1 if 3 <= tenure <= 6 else 0
    t_6_12 = 1 if 6 < tenure <= 12 else 0
    t_1_2y = 1 if 12 < tenure <= 24 else 0
    t_2_4y = 1 if 24 < tenure <= 48 else 0
    t_4p_y = 1 if tenure > 48 else 0

    # ربط المدخلات بأعمدة الموديل (مع إبقاء الخدمات الفرعية المحذوفة بصفر أو قيمتها افتراضياً)
    raw_input = {
        'gender': 1 if gender == "Male" else 0,
        'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
        'Partner': 1 if partner == "Yes" else 0,
        'Dependents': 1 if dependents == "Yes" else 0,
        'tenure': tenure,
        'PhoneService': 1 if phone_service == "Yes" else 0,
        'PaperlessBilling': 1 if paperless_billing == "Yes" else 0,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'MultipleLines_Yes': 1 if multiple_lines == "Yes" else 0,
        'InternetService_Fiber optic': 1 if internet_service == "Fiber optic" else 0,
        'InternetService_No': 1 if internet_service == "No" else 0,
        'OnlineSecurity_Yes': 1 if online_security == "Yes" else 0,
        'OnlineBackup_Yes': 0,
        'DeviceProtection_Yes': 0,
        'TechSupport_Yes': 1 if tech_support == "Yes" else 0,
        'StreamingTV_Yes': 0,
        'StreamingMovies_Yes': 0,
        'Contract_One year': 1 if contract == "One year" else 0,
        'Contract_Two year': 1 if contract == "Two year" else 0,
        'PaymentMethod_Credit card (automatic)': 1 if payment_method == "Credit card (automatic)" else 0,
        'PaymentMethod_Electronic check': 1 if payment_method == "Electronic check" else 0,
        'PaymentMethod_Mailed check': 1 if payment_method == "Mailed check" else 0,
        'tenure_detailed_1-2 Years': t_1_2y,
        'tenure_detailed_2-4 Years': t_2_4y,
        'tenure_detailed_3-6 Months': t_3_6,
        'tenure_detailed_4+ Years': t_4p_y,
        'tenure_detailed_6-12 Months': t_6_12,
    }

    input_df = pd.DataFrame([raw_input])
    
    # التأكد من تطابق الأعمدة وترتيبها مع ما تدرب عليه الموديل
    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0
            
    input_df = input_df[model_columns]

    # تحويل القيم الرقمية باستخدام الـ Scaler المحفوظ
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    input_df[num_cols] = scaler.transform(input_df[num_cols])

    # توقع النسبة
    churn_prob = model.predict_proba(input_df)[0][1] * 100

    st.markdown("### Prediction Output")
    
    res_col1, res_col2 = st.columns([1, 2])
    
    with res_col1:
        st.markdown("**Churn Probability**")
        
        if churn_prob < 30:
            neon_color = "#00FF66"  # أخضر نيون
        elif 30 <= churn_prob <= 60:
            neon_color = "#FFA500"  # برتقالي نيون
        else:
            neon_color = "#FF3333"  # أحمر نيون
            
        st.markdown(
            f"""
            <div style="font-size: 42px; font-weight: bold; color: {neon_color}; text-shadow: 0 0 12px {neon_color}66; margin-bottom: 10px;">
                {churn_prob:.1f}%
            </div>
            """,
            unsafe_allow_html=True
        )
        
        if churn_prob >= 50:
            st.error("High Risk of Churn")
        else:
            st.success("Low Churn Risk")

    with res_col2:
        st.subheader("Recommended Retention Actions")
        recommendations = get_recommendations(contract, payment_method, tech_support, online_security)
        for rec in recommendations:
            st.write(rec)
            
            
            
            