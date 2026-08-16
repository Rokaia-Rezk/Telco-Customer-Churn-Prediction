# 🔮 Telco Customer Churn Prediction & Retention Intelligence

An end-to-end Machine Learning web application designed to predict customer churn risk for telecommunications companies and provide automated, actionable retention strategies.

---

## 🚀 Live Demo
You can test the live application here : [https://telco-customer-churn-prediction-6maww7csk4lbdvmmtgziqh.streamlit.app/]

---

## 📌 Project Overview
Customer churn is a critical metric for subscription-based businesses. This project builds a predictive classification model using historical telecommunications data to identify customers who are likely to cancel their services. Alongside the prediction, the app features an automated **Retention Recommendation Engine** that suggests specific promotional or tactical interventions based on customer behavior and billing parameters.

---

## 🛠️ Tech Stack & Tools
* **Programming Language:** Python
* **Data Manipulation & Preprocessing:** Pandas, NumPy, Scikit-Learn
* **Machine Learning Model:** Logistic Regression / Classification Algorithms
* **Model Serialization:** Joblib
* **Deployment & UI:** Streamlit, Vercel / Streamlit Community Cloud
* **Version Control:** Git & GitHub

---

## 📊 Key Features
1. **Interactive User Interface:** Clean, modern, multi-column dashboard built with Streamlit.
2. **Real-Time Risk Scoring:** Instantly calculates the churn probability percentage with intuitive color-coded risk indicators (Low, Medium, High).
3. **Automated Retention Actions:** Dynamically generates tailored business recommendations (e.g., contract upgrades, auto-pay incentives, VIP tech support) based on user inputs.
4. **Optimized Feature Pipeline:** Matches exact data cleaning steps (handling Yes/No features, tenure groupings, and scaling).

---

## 📂 Project Structure
```text
├── models/
│   ├── logistic_regression_model.pkl
│   ├── scaler.pkl
│   └── model_columns.pkl
├── app.py
├── requirements.txt
└── README.md
⚙️ Local Installation & Setup
Clone the repository:

Bash
git clone [https://github.com/Rokaia2006/telco-customer-churn-prediction.git](https://github.com/Rokaia2006/telco-customer-churn-prediction.git)
cd telco-customer-churn-prediction
Install dependencies:

Bash
pip install -r requirements.txt
Run the Streamlit application:

Bash
streamlit run app.py
💡 Business Value & Impact
Helps telecom operators proactively identify high-risk customers before churn occurs.

Shifts strategy from reactive customer service to proactive, data-driven retention marketing.

Reduces customer acquisition costs by focusing retention budgets on vulnerable customer segments.

👩‍💻 Author
Rokaia Hassan

GitHub Profile

