import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Smart Crop Recommendation", layout="wide")
st.title("Smart Crop Recommendation System")

# Load model
pipeline = joblib.load("model/pipeline.pkl")
accuracy = 0.9572  # your accuracy

# Load dataset
df = pd.read_csv("data/Crop_dataset.csv")

st.header("Crop Prediction")

col1, col2, col3 = st.columns(3)

with col1:
    n = st.number_input("Nitrogen (N)", 0, 200, 50)
    k = st.number_input("Potassium (K)", 0, 200, 40)
    ph = st.number_input("pH Level", 0.0, 14.0, 6.5)

with col2:
    p = st.number_input("Phosphorus (P)", 0, 200, 40)
    humidity = st.number_input("Humidity (%)", 0.0, 100.0, 60.0)
    rainfall = st.number_input("Rainfall (mm)", 0.0, 300.0, 100.0)

with col3:
    temperature = st.number_input("Temperature (°C)", 0.0, 50.0, 25.0)

if st.button("Predict Crop"):
    # Use EXACT lowercase feature names used during training
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
