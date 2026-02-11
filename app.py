import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Page configuration
st.set_page_config(page_title="Weather Classifier", layout="wide")
st.title("K-Nearest Neighbor (KNN) Weather Classification")

st.markdown(
"""
This app uses the K-Nearest Neighbors (KNN) algorithm from scikit-learn to classify 
weather conditions based on temperature and humidity levels.
"""
)

# Dataset
X = np.array([
    [30, 70],
    [25, 80],
    [27, 60],
    [31, 65],
    [23, 85],
    [28, 75]
])

y = np.array([0, 1, 0, 0, 1, 1])  # 0 = Sunny, 1 = Rainy

label_map = {
    0: "Sunny",
    1: "Rainy"
}

# Sidebar inputs
st.sidebar.header("Input Parameters")

temperature = st.sidebar.slider("Temperature (°C)", min_value=20, max_value=35, value=26, step=1)
humidity = st.sidebar.slider("Humidity (%)", min_value=50, max_value=90, value=78, step=1)
n = st.sidebar.slider("KNN value (k)", min_value=1, max_value=10, value=3, step=1)

# Train model
knn = KNeighborsClassifier(n_neighbors=n)
knn.fit(X, y)

# Prediction
new_weather = np.array([[temperature, humidity]])
pred = knn.predict(new_weather)[0]
pred_proba = knn.predict_proba(new_weather)[0]

weather_label = label_map[pred]
confidence = pred_proba[pred] * 100

st.sidebar.markdown("---")
st.sidebar.subheader("Prediction Result")

if pred == 0:
    st.sidebar.success(f"Weather: {weather_label}")
else:
    st.sidebar.info(f"Weather: {weather_label}")

st.sidebar.metric("Confidence (%)", f"{confidence:.1f}")

# Visualization
col1, col2 = st.columns(2)

with col1:
    st.subheader("Classification Visualization")

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(X[y == 0, 0], X[y == 0, 1],
               color="orange", label="Sunny", s=100, edgecolor="k", alpha=0.7)

    ax.scatter(X[y == 1, 0], X[y == 1, 1],
               color="blue", label="Rainy", s=100, edgecolor="k", alpha=0.7)

    colors = ["orange", "blue"]

    ax.scatter(new_weather[0, 0], new_weather[0, 1],
               color=colors[pred], marker="*",
               s=300, edgecolor="black",
               label=f"New day: {weather_label}", zorder=5)

    ax.set_xlabel("Temperature (°C)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Humidity (%)", fontsize=12, fontweight="bold")
    ax.set_title("Weather Classification Model", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(20, 35)
    ax.set_ylim(50, 90)

    st.pyplot(fig)

with col2:
    st.subheader("Model Information")

    st.write("**Training Data Summary:**")
    st.write(f"- Total samples: {len(X)}")
    st.write(f"- Sunny days: {np.sum(y == 0)}")
    st.write(f"- Rainy days: {np.sum(y == 1)}")
    st.write(f"- K-neighbors: {n}")

    st.markdown("---")

    st.write("**Current Input:**")
    st.write(f"- Temperature: {temperature}°C")
    st.write(f"- Humidity: {humidity}%")
    st.write(f"- K value: {n}")

    st.markdown("---")

    st.write("**Prediction Details:**")
    st.metric("Sunny Probability (%)", f"{pred_proba[0] * 100:.1f}")
    st.metric("Rainy Probability (%)", f"{pred_proba[1] * 100:.1f}")

st.caption("KNN Weather Classification Model")
