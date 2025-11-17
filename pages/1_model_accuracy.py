import streamlit as st
import joblib

st.title("Model Accuracy")

# Load model
pipeline = joblib.load("model/pipeline.pkl")
accuracy = 0.9572  # your accuracy

st.metric("Model Training Accuracy", f"{accuracy * 100:.2f}%")
