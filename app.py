import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Smart Crop Recommendation", layout="wide")
st.title("Smart Crop Recommendation System")

# Load model
pipeline = joblib.load("model/pipeline.pkl")
accuracy = 0.9572

# Load dataset
df = pd.read_csv("data/Crop_dataset.csv")

st.header("Crop Prediction")

col1, col2, col3 = st.columns(3)

with col1:
    n = st.number_input("Nitrogen (N)", min_value=1, max_value=200, value=50)
    k = st.number_input("Potassium (K)", min_value=1, max_value=200, value=40)
    ph = st.number_input("pH Level", min_value=1.0, max_value=14.0, value=6.5)

with col2:
    p = st.number_input("Phosphorus (P)", min_value=1, max_value=200, value=40)
    humidity = st.number_input("Humidity (%)", min_value=1.0, max_value=100.0, value=60.0)
    rainfall = st.number_input("Rainfall (mm)", min_value=1.0, max_value=300.0, value=100.0)

with col3:
    temperature = st.number_input("Temperature (°C)", min_value=1.0, max_value=50.0, value=25.0)

if st.button("Predict Crop"):

   
    input_data = pd.DataFrame([{
        "n": n,
        "p": p,
        "k": k,
        "temperature": temperature,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall
    }])

    # Predict
    result = pipeline.predict(input_data)[0]
    st.success(f"Recommended Crop: {result}")

st.write("---")
st.info("Use the menu on the left to explore Model Accuracy and Data Visualizations.")
