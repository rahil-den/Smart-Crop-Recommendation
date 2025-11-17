import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

# Load dataset
df = pd.read_csv("../data/crop_dataset.csv")

# Split features & target
X = df.drop("label", axis=1)   # rename "label" to your target column name
y = df["label"]

# Logistic Regression Pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

# Train the pipeline
pipeline.fit(X, y)

# Save trained pipeline
joblib.dump(pipeline, "pipeline.pkl")

print("pipeline.pkl saved successfully with Logistic Regression model.")
