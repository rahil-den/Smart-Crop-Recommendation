import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Data Visualization")

df = pd.read_csv("data/crop_dataset.csv")


# ----------------------------
# Dataset Preview
# ----------------------------
st.subheader("Dataset Preview")
st.dataframe(df.head())


# ----------------------------
# Correlation Heatmap
# ----------------------------
st.subheader("Correlation Heatmap")

corr = df.corr(numeric_only=True)

fig1, ax1 = plt.subplots(figsize=(10, 6))
cax = ax1.matshow(corr, cmap="viridis")
fig1.colorbar(cax)

ax1.set_xticks(range(len(corr.columns)))
ax1.set_yticks(range(len(corr.columns)))
ax1.set_xticklabels(corr.columns, rotation=90)
ax1.set_yticklabels(corr.columns)

st.pyplot(fig1)


# ----------------------------
# Crop distribution
# ----------------------------
st.subheader("Crop Distribution")

crop_counts = df["label"].value_counts()

fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.bar(crop_counts.index, crop_counts.values)
ax2.set_xlabel("Crop Type")
ax2.set_ylabel("Count")
ax2.set_title("Crop Distribution")
plt.xticks(rotation=90)

st.pyplot(fig2)


# ----------------------------
# Feature Histograms
# ----------------------------
st.subheader("Feature Distribution")

numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

for col in numeric_cols:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df[col], bins=20)
    ax.set_title(f"Distribution of {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

