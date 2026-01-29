import streamlit as st
import pandas as pd
import numpy as np
import sklearn
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
label_encoder = LabelEncoder()
scaler = StandardScaler()

# load saved model and data
model = pickle.load(open('customer_churn_prediction_model.pkl','rb'))
# df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# create app
#st.title("Customer Churn Prediction App")
st.title("📉 Customer Churn Prediction App")
st.write("Predict whether a customer is likely to churn") 

# Collect user input
gender = st.selectbox("Select Gender", options=['Female','Male'])
SeniorCitizen = st.selectbox("Are you a senior citizen?", options=['Yes','No'])
Partner = st.selectbox("Do you have partner?", options=['Yes','No'])
Dependents	 = st.selectbox("Are you dependent on other?", options=['Yes','No'])
tenure = st.text_input("Enter Your tenure?")
PhoneService = st.selectbox("Do you have phone service?", options=['Yes','No'])
MultipleLines = st.selectbox("Do you have multiple lines servics?", options=['Yes','No','no phone service'])
Contract = st.selectbox("Your Contracts?", options=['One year','Two year','Month-to-month'])
TotalCharges = st.text_input("Enter your Total charges?")

# Prediction
def do_prediction(gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, Contract, TotalCharges):
    input_data = {
            'gender': [gender],
            'SeniorCitizen': [SeniorCitizen],
            'Partner': [Partner],
            'Dependents': [Dependents],
            'tenure': [tenure],
            'PhoneService': [PhoneService],
            'MultipleLines': [MultipleLines],
            'Contract': [Contract],
            'TotalCharges': [TotalCharges]
    }
    # Create a DataFrame from the dictionary
    df = pd.DataFrame(input_data)


    # Encode the categorical columns
    categorical_columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'Contract']
    for column in categorical_columns:
        df[column] = label_encoder.fit_transform(df[column])
    df = scaler.fit_transform(df)

    result = model.predict(df).reshape(1, -1)
    return result[0]


if st.button("Predict"):
    result = do_prediction(gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, Contract, TotalCharges)
    if result == 1:
        st.error("Customer is likely to CHURN")
    else:
        st.success("Customer is NOT likely to churn")