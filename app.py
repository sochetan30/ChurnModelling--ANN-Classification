from tensorflow.keras.models import load_model
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import streamlit as st

# Load the train model
model=tf.keras.models.load_model('ann_model.h5')


# Load scaler, encoder:

with open('label_encoder_gender.pkl','rb') as obj:
    label_encoder_gender=pickle.load(obj)

with open('onehotencoder_geo.pkl','rb') as obj:
    onehotencoder_geo=pickle.load(obj)

with open('scaler.pkl','rb') as obj:
    scaler=pickle.load(obj)


## streamlit app
st.title("Customer Churn Prediction")

#User Input 
# Input fields for the user
credit_score = st.number_input("Credit Score")
geography = st.selectbox("Geography", onehotencoder_geo.categories_[0])
gender = st.radio("Gender", label_encoder_gender.classes_)
age = st.slider("Age",18,92)
tenure = st.slider("Tenure (years)", 0,10)
balance = st.number_input("Balance")
num_of_products = st.slider("num of products",1,4)
has_cr_card = st.radio("Has Credit Card", ["Yes", "No"])
is_active_member = st.radio("Is Active Member", ["Yes", "No"])
estimated_salary = st.number_input("Estimated Salary")


# Convert the responses into the input_data dictionary
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [1 if has_cr_card == "Yes" else 0],
    'IsActiveMember': [1 if is_active_member == "Yes" else 0],
    'EstimatedSalary': [estimated_salary]
})

## geography

geo_encoded = onehotencoder_geo.transform(np.array([[geography]]).reshape(-1, 1))
geo_encoded_df= pd.DataFrame(geo_encoded, columns=onehotencoder_geo.get_feature_names_out())

input_data_df=pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

## Saling the input
input_scaled= scaler.transform(input_data_df)

# Predcit Churn
prediction=model.predict(input_scaled)
prediction_proba=prediction[0][0]

st.write(f"Churn Probability:  {prediction_proba: .2f}")
if prediction_proba > 0.5:
    st.write("The customer is likely to churn")
else:
    st.write("The customer is not likely to churn")