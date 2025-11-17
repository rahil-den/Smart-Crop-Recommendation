import streamlit as st
import joblib
import numpy as np

pipeline = joblib.load("../model/pipeline.pkl")

st.title("Smart Crop Recommendation System")

st.write("Enter soil & climate values to get the recommended crop.")

# Input fields (adjust according to your dataset)
nitrogen = st.number_input("Nitrogen (N)", min_value=0.0)
phosphorus = st.number_input("Phosphorus (P)", min_value=0.0)
potassium = st.number_input("Potassium (K)", min_value=0.0)
temperature = st.number_input("Temperature (°C)", min_value=0.0)
humidity = st.number_input("Humidity (%)", min_value=0.0)
ph = st.number_input("pH Value", min_value=0.0)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0)

if st.button("Predict Crop"):
    user_input = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
    prediction = pipeline.predict(user_input)[0]

    st.subheader(f"Recommended Crop: {prediction}")
