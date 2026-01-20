import streamlit as st
import joblib
import pandas as pd

pipe_disputes = joblib.load("pipe_disputes.pkl")
pipe_rating = joblib.load("pipe_rating.pkl")
pipe_fraud = joblib.load("pipe_fraud.pkl")

st.title("💻 Fraud Detection in Freelancing Platforms")
st.markdown("### 📝 Enter Account Details")

with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        name_na = st.text_input("👤 Name", "Ethan")
        account_type = st.selectbox(
            "🏷 Account Type",
            ["Freelancer", "Client", "Hybrid"]
        )

    with col2:
        account_age_days = st.number_input(
            "📅 Account Age (Days)", min_value=1, value=180
        )
        completed_projects = st.number_input(
            "📂 Completed Projects / Posted Jobs", min_value=0, value=63
        )

    submit_btn = st.form_submit_button("🚀 Predict")


if submit_btn:
    try:
        new_data = pd.DataFrame({
            "Name": [name_na],
            "Account_Type": [account_type],
            "Account_Age_Days": [account_age_days],
            "Completed_Projects / Posted_Jobs": [completed_projects]
        })

        disputes = pipe_disputes.predict(new_data)[0]
        avg_rating = pipe_rating.predict(new_data)[0]
        is_fraud = pipe_fraud.predict(new_data)[0]

        st.subheader("📊 Prediction Dashboard")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("📌 Disputes", int(disputes))

        with col2:
            st.metric("⭐ Avg Rating", f"{avg_rating:.2f}")

        with col3:
            fraud_text = "Fraud Detected ⚠" if is_fraud == 1 else "Not Fraudulent ✅"
            st.metric("🚨 Fraud Status", fraud_text)

    except Exception as e:
        st.error(f"⚠ Prediction failed: {e}")


