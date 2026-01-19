# 🌾 Smart Crop Recommendation System

A machine learning powered web application that recommends the most suitable crop based on soil and climate parameters.  
This project uses Logistic Regression and a preprocessing pipeline for accurate predictions.

---

##  Features

- Predict the best crop based on:
  - Nitrogen (N)
  - Phosphorus (P)
  - Potassium (K)
  - Temperature
  - Humidity
  - Soil pH
  - Rainfall
- Fully functional Streamlit web app
- Logistic Regression model with sklearn pipeline
- Clean folder structure for production-ready ML deployment
- Dataset stored locally inside `/data`

---

## 📁 Project Structure
```
Smart-Crop-Recommendation/
│
├── data/
│ └── crop_dataset.csv
│
├── model/
│ ├── train_pipeline.py
│ └── pipeline.pkl
│
├── app.py
│-|
│ └── pages/
│ ├── 1_📊_Model_Accuracy.py
│ └── 2_📈_Data_Visualization.py
│
├── requirements.txt
│
└── README.md
```

## Made with ❤️by Rahil & Talha

