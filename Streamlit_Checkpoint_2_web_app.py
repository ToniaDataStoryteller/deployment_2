# Creating Streamlit Application for Financial Inclusion predictor
# This app allows users to input features and predict if someone has a bank account.

import streamlit as st
import pandas as pd
import pickle

# 1. Load saved model & encoders
with open("bank_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

with open("num_imputer.pkl", "rb") as f:
    num_imputer = pickle.load(f)

with open("cat_imputer.pkl", "rb") as f:
    cat_imputer = pickle.load(f)

with open("feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)

# 2. Streamlit app UI
st.title("🏦 Financial Inclusion in Africa Predictor")
st.write("Fill the form below to predict if a person is likely to have a bank account.")

# Input fields (must match dataset features used in training)
country = st.selectbox("Country", ["Kenya", "Rwanda", "Tanzania", "Uganda"])
location_type = st.selectbox("Location Type", ["Rural", "Urban"])
cellphone_access = st.selectbox("Cellphone Access", ["Yes", "No"])
age_of_respondent = st.number_input("Age of Respondent", min_value=16, max_value=100, value=30)
gender_of_respondent = st.selectbox("Gender", ["Male", "Female"])
relationship_with_head = st.selectbox("Relationship with Head",
    ["Head of Household", "Spouse", "Child", "Parent", "Other relative", "Other non-relatives", "Dont know"])
marital_status = st.selectbox("Marital Status",
    ["Married/Living together", "Single/Never Married", "Divorced/Seperated", "Widowed"])
education_level = st.selectbox("Education Level",
    ["No formal education", "Primary education", "Secondary education", "Tertiary education", "Other/Dont know/RTA"])
job_type = st.selectbox("Job Type",
    ["Self employed", "Government Dependent", "Formally employed Private", "Formally employed Government",
     "Informally employed", "Farming and Fishing", "Remittance Dependent", "Other Income", "No Income", "Dont Know/Refuse to answer"])

# 3. Convert input into dataframe
input_data = pd.DataFrame({
    "country": [country],
    "location_type": [location_type],
    "cellphone_access": [cellphone_access],
    "age_of_respondent": [age_of_respondent],
    "gender_of_respondent": [gender_of_respondent],
    "relationship_with_head": [relationship_with_head],
    "marital_status": [marital_status],
    "education_level": [education_level],
    "job_type": [job_type]
})

# 4. Encode categorical features
for col in input_data.columns:
    if col in label_encoders:
        input_data[col] = label_encoders[col].transform(input_data[col])

# 5. Prediction
if st.button("Predict"):
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("✅ This person is LIKELY to have a bank account.")
    else:
        st.error("❌ This person is UNLIKELY to have a bank account.")