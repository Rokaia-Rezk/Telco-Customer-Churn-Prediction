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
    page_icon="🔮",
    layout="wide"
)

# 2. Professional Data-Dashboard Styling (IBM Carbon–inspired, fitting the IBM Telco dataset)
custom_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap');

:root {
    --bg: #161616;
    --surface: #262626;
    --border: #393939;
    --text: #F4F4F4;
    --text-muted: #A8A8A8;
    --accent: #4589FF;
    --risk-high: #FA4D56;
    --risk-medium: #F1C21B;
    --risk-low: #42BE65;
}

/* Background: dark base + subtle network dot-grid + faint diagonal signal lines,
   evoking a telecom/connectivity feel instead of flat empty black */
.stApp {
    background-color: var(--bg);
    background-image:
        radial-gradient(circle, rgba(69,137,255,0.16) 1px, transparent 1px),
        repeating-linear-gradient(135deg, rgba(69,137,255,0.05) 0px, rgba(69,137,255,0.05) 1px, transparent 1px, transparent 28px);
    background-size: 26px 26px, 40px 40px;
    color: var(--text);
}

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', -apple-system, sans-serif;
}

/* Headings */
h1, h2, h3, h4 {
    color: var(--text) !important;
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 600 !important;
    letter-spacing: -0.01em;
}

h1 {
    border-bottom: 1px solid var(--border);
    padding-bottom: 16px;
}

/* Input Labels */
label, [data-testid="stWidgetLabel"] p {
    color: var(--text-muted) !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

.stMarkdown p {
    color: var(--text-muted) !important;
}

/* Input widgets: flat surfaces, hairline borders, no rounded-bubble look */
div[data-baseweb="select"] > div, .stNumberInput input {
    background-color: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 2px !important;
    color: var(--text) !important;
}

/* Disabled inputs (Auto-calculated Total Charges) use -webkit-text-fill-color
   separately from color in WebKit browsers — without this override the text
   is invisible against the dark background */
.stNumberInput input:disabled {
    -webkit-text-fill-color: var(--text) !important;
    color: var(--text) !important;
    opacity: 1 !important;
}

/* Section containers */
[data-testid="column"] {
    background-color: var(--surface);
    border: 1px solid var(--border);
    border-radius: 2px;
    padding: 20px;
}

/* Button: flat, solid, no gradient/glow */
.stButton>button {
    background-color: var(--accent);
    color: #161616;
    border: none;
    border-radius: 2px;
    padding: 12px 28px;
    font-size: 15px;
    font-weight: 600;
    width: 100%;
    transition: background-color 0.15s ease;
}
.stButton>button:hover {
    background-color: #6FA1FF;
}

/* Alerts: flat card with left accent bar instead of default rounded pill */
[data-testid="stAlert"] {
    background-color: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 2px !important;
    border-left: 3px solid var(--text-muted) !important;
}
[data-testid="stAlertContentError"], [data-testid="stAlertContentError"] p {
    color: var(--risk-high) !important;
}
[data-testid="stAlertContentSuccess"], [data-testid="stAlertContentSuccess"] p {
    color: var(--risk-low) !important;
}

/* Data / metric numbers use the monospaced Plex Mono */
.metric-number {
    font-family: 'IBM Plex Mono', monospace;
}

/* Cute floating crystal ball + pulsing signal rings (landing page) */
@keyframes float-orb {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-10px); }
}
@keyframes signal-pulse {
    0%   { transform: scale(0.6); opacity: 0.55; }
    100% { transform: scale(2.2); opacity: 0; }
}
.orb-wrap {
    position: relative;
    width: 110px;
    height: 110px;
    margin: 0 auto 18px auto;
    display: flex;
    align-items: center;
    justify-content: center;
}
.signal-ring {
    position: absolute;
    width: 70px;
    height: 70px;
    border: 1.5px solid var(--accent);
    border-radius: 50%;
    animation: signal-pulse 2.6s ease-out infinite;
}
.signal-ring.ring-2 { animation-delay: 0.9s; }
.signal-ring.ring-3 { animation-delay: 1.8s; }
.orb-emoji {
    position: relative;
    font-size: 46px;
    animation: float-orb 3.2s ease-in-out infinite;
    filter: drop-shadow(0 0 10px rgba(69,137,255,0.5));
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Sidebar — project credit + repository link
with st.sidebar:
    st.markdown(
        """
        <div style="padding: 8px 0 20px 0; border-bottom: 1px solid var(--border); margin-bottom: 16px;">
            <div style="font-size: 12px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em;">Project</div>
            <div style="font-size: 15px; color: var(--text); font-weight: 600; margin-top: 4px;">Telco Churn Predictor</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.link_button("View on GitHub", "https://github.com/Rokaia-Rezk/Telco-Customer-Churn-Prediction", use_container_width=True)
    st.markdown(
        """
        <div style="font-size: 12px; color: var(--text-muted); margin-top: 16px;">
            Built by <span style="color: var(--text);">Rokaia Rezk</span>
        </div>
        """,
        unsafe_allow_html=True
    )

# 3. Landing / Welcome Screen
if "started" not in st.session_state:
    st.session_state.started = False

if not st.session_state.started:
    st.markdown("<div style='height: 8vh;'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="orb-wrap">
            <div class="signal-ring"></div>
            <div class="signal-ring ring-2"></div>
            <div class="signal-ring ring-3"></div>
            <div class="orb-emoji">🔮</div>
        </div>
        <div style="text-align:center; max-width:640px; margin:0 auto;">
            <div style="font-size:13px; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.08em; margin-bottom:14px;">
                Telco Churn Predictor
            </div>
            <h1 style="border:none; font-size:38px; margin-bottom:18px;">Hi there.</h1>
            <p style="font-size:17px; color:var(--text-muted); line-height:1.6;">
                Every customer leaves a trail before they leave for good.<br>
                Ready to find out if yours is about to walk away?
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.write("")
    spacer1, center, spacer2 = st.columns([2, 1, 2])
    with center:
        if st.button("I'm Ready", use_container_width=True):
            st.session_state.started = True
            st.rerun()
    st.stop()

# 4. Load Saved Artifacts Safely
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
        # Total Charges يتحسب أوتوماتيك = عدد شهور الاشتراك × الفاتورة الشهرية
        total_charges = tenure * monthly_charges
        st.markdown(
            f"""
            <div style="margin-bottom:4px;">
                <div style="font-size:13px; font-weight:500; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.04em; margin-bottom:6px;">
                    Total Charges ($) — Auto-calculated
                </div>
                <div style="border:1px solid var(--border); border-radius:2px; background-color:var(--surface);
                            padding:10px 14px; font-size:15px; color:var(--text); font-family:'IBM Plex Mono', monospace;">
                    {total_charges:,.2f}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

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
        if churn_prob < 30:
            risk_color = "var(--risk-low)"
            risk_label = "Low Risk"
        elif churn_prob <= 60:
            risk_color = "var(--risk-medium)"
            risk_label = "Medium Risk"
        else:
            risk_color = "var(--risk-high)"
            risk_label = "High Risk"

        marker_pos = min(max(churn_prob, 0), 100)

        st.markdown(
            f"""
            <div style="border:1px solid var(--border); border-radius:2px; padding:20px; background-color:var(--surface);">
                <div style="font-size:12px; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.05em; margin-bottom:6px;">
                    Churn Probability
                </div>
                <div class="metric-number" style="font-size:40px; font-weight:600; color:{risk_color}; line-height:1;">
                    {churn_prob:.1f}%
                </div>
                <div style="font-size:13px; font-weight:600; color:{risk_color}; margin-top:6px; margin-bottom:16px;">
                    {risk_label}
                </div>
                <div style="position:relative; height:6px; border-radius:3px;
                            background:linear-gradient(90deg, var(--risk-low) 0%, var(--risk-low) 30%, var(--risk-medium) 30%, var(--risk-medium) 60%, var(--risk-high) 60%, var(--risk-high) 100%);">
                    <div style="position:absolute; left:{marker_pos}%; top:-4px; transform:translateX(-50%);
                                width:2px; height:14px; background-color:var(--text);"></div>
                </div>
                <div style="display:flex; justify-content:space-between; font-size:11px; color:var(--text-muted); margin-top:6px;">
                    <span>0</span><span>30</span><span>60</span><span>100</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")
        if churn_prob >= 50:
            st.error("High Risk of Churn", icon=None)
        else:
            st.success("Low Churn Risk", icon=None)

    with res_col2:
        st.subheader("Recommended Retention Actions")
        recommendations = get_recommendations(contract, payment_method, tech_support, online_security)
        recs_html = "".join(
            f"""<div style="border:1px solid var(--border); border-left:3px solid var(--accent);
                            border-radius:2px; background-color:var(--surface);
                            padding:12px 16px; margin-bottom:10px; font-size:14px; color:var(--text);">
                    {rec}
                </div>"""
            for rec in recommendations
        )
        st.markdown(recs_html, unsafe_allow_html=True)

# Footer — subtle credit line
st.markdown(
    """
    <div style="margin-top:60px; padding-top:16px; border-top:1px solid var(--border);
                text-align:center; font-size:12px; color:var(--text-muted);">
        Developed by Rokaia Rezk
    </div>
    """,
    unsafe_allow_html=True
)