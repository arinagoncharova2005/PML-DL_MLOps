# streamlit_app.py
import streamlit as st
import requests

# FastAPI endpoint
FASTAPI_URL = "http://localhost:8000/predict"

# Streamlit app UI
st.title("Iris Flower Classifier")


fuel_types = ["Petrol", "Diesel", "CNG"]
seller_types = ['Dealer', 'Individual']
transmission_types = ['Manual', 'Automatic']
# Input fields for the Iris flower data
Car_Name = st.text_input("Car Name")
Year = st.number_input("Year")
Present_Price = st.number_input("Present Price")
Kms_Driven = st.number_input("Kms Driven")
Fuel_Type = st.selectbox("Fuel Type", fuel_types)
Seller_Type = st.selectbox("Seller Type", seller_types)
Transmission = st.selectbox("Transmission", transmission_types)
Owner = st.number_input("Owner")

# Make prediction when the button is clicked
if st.button("Predict"):
    # Prepare the data for the API request
    input_data = {
        "Car_Name": Car_Name,
        "Year": Year,
        "Present_Price": Present_Price,
        "Kms_Driven": Kms_Driven,
        "Fuel_Type": Fuel_Type,
        "Seller_Type": Seller_Type,
        "Transmission":Transmission,
        "Owner":Owner
    }
    
    # Send a request to the FastAPI prediction endpoint
    response = requests.post(FASTAPI_URL, json=input_data)
    prediction = response.json()["prediction"]
    
    # Display the result
    st.success(f"The model predicts class: {prediction}")
