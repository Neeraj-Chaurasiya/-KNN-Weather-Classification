# KNN Weather Classification Project

## 📌 Project Overview

This project is a simple Machine Learning application that predicts whether the weather is **Sunny** or **Rainy** based on two input features:

- Temperature  
- Humidity  

The model used in this project is **K-Nearest Neighbors (KNN)**, and the user interface is built using **Streamlit**.

---

## 🎯 Objective

The main goal of this project is to:

- Understand how KNN algorithm works
- Learn how to build an interactive ML app using Streamlit
- Visualize training data and predictions using Matplotlib

---

## 🧠 What is KNN (K-Nearest Neighbors)?

KNN is a supervised machine learning algorithm used for classification and regression.

How it works (simple explanation):

1. The model stores all training data.
2. When a new data point is given, it looks at the **K nearest neighbors**.
3. It checks which class appears most among those neighbors.
4. That class is assigned to the new data point.

In this project:
- K = 3
- Classes: Sunny (0) and Rainy (1)

---

## 📊 Dataset Used

We use a small sample dataset:

Temperature and Humidity values with labels:

| Temperature | Humidity | Label |
|-------------|----------|--------|
| 50          | 70       | Sunny  |
| 25          | 80       | Rainy  |
| 27          | 60       | Sunny  |
| 31          | 65       | Sunny  |
| 23          | 85       | Rainy  |
| 20          | 75       | Rainy  |

---

## 🛠️ Technologies Used

This project is built using:

- Python  
- Streamlit – for web interface  
- NumPy – for data handling  
- Matplotlib – for visualization  
- Scikit-learn – for KNN model  

---

## ▶️ How to Run the Project

### Step 1: Install Dependencies

Run this command in terminal:

