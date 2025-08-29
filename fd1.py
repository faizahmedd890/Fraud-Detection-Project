import streamlit as st
import joblib
import pandas as pd
from scipy.sparse import hstack, csr_matrix

encoder1 = joblib.load("encoder1.pkl")
encoder2 = joblib.load("encoder2.pkl")
encoder3 = joblib.load("encoder3.pkl")
encoder4 = joblib.load("encoder4.pkl")

model1 = joblib.load("model1.pkl")
model2 = joblib.load("model2.pkl")
model3 = joblib.load("model3.pkl")

st.title("💻 Fraud Detection In Freelancing Platforms Prediction App")
st.markdown("### 📝 Enter Account Details Below")


with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        name_na = st.text_input("👤 Name", "Ethan")
        account_type = st.selectbox("🏷 Account Type", ["Freelancer", "Client", "Hybrid"], index=2)

    with col2:
        account_age_days = st.number_input("📅 Account Age (Days)", min_value=1, value=180, step=1)
        completed_projects = st.number_input("📂 Completed Projects / Posted Jobs", min_value=0, value=63, step=1)

    submit_btn = st.form_submit_button("🚀 Predict")

if submit_btn:
    try:
        name_col = encoder1.feature_names_in_[0]
        account_type_col = encoder2.feature_names_in_[0]
        account_age_days_col = encoder3.feature_names_in_[0]
        completed_projects_col = encoder4.feature_names_in_[0]

        new_data1 = pd.DataFrame({name_col: [name_na]})
        new_data2 = pd.DataFrame({account_type_col: [account_type]})
        new_data3 = pd.DataFrame({account_age_days_col: [account_age_days]})
        new_data4 = pd.DataFrame({completed_projects_col: [completed_projects]})

        r1 = csr_matrix(encoder1.transform(new_data1))
        r2 = encoder2.transform(new_data2)
        r3 = csr_matrix(encoder3.transform(new_data3))
        r4 = csr_matrix(encoder4.transform(new_data4))

        inp = hstack([r1, r2, r3, r4])

        Disputes = model1.predict(inp)[0]
        Avg_Rating = model2.predict(inp)[0]
        Is_Fraud = model3.predict(inp)[0]

 
        st.subheader("📊 Prediction Dashboard")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(label="📌 Disputes", value=int(Disputes))

        with col2:
            st.metric(label="⭐ Avg Rating", value=f"{Avg_Rating:.2f}")

        with col3:
            fraud_text = "Fraud Detected" if Is_Fraud == 1 else "Not Fraudulent"
            st.metric(label="⚠ Fraudulent?", value=fraud_text)


    except Exception as e:
        st.error(f"⚠ Prediction failed: {e}")
