# CALIFORNIA_HOUSING_PRICE_PREDICTION_USING_MACHINE_LEARNING

🏠 California Housing Price Prediction using Machine Learning

An end-to-end Machine Learning project that predicts California housing prices and classifies properties into value categories. The project implements regression, classification, SVM, clustering with PCA, neural networks, and a Streamlit web deployment.

📌 Project Overview

This project uses the California Housing dataset from Scikit-learn to:

Predict median house prices (Regression)

Classify houses into Low, Medium, and High value categories

Perform clustering to group housing regions

Compare multiple machine learning algorithms

Deploy the final model using Streamlit

The dataset was split into:

70% Training

15% Validation

15% Testing

Proper validation methodology was followed to avoid data leakage.

🚀 Features Implemented
🔹 Regression

Multiple Linear Regression

Evaluation using MSE and R² Score

🔹 Classification

Logistic Regression

Decision Tree

Random Forest (Best Performing Model)

🔹 Support Vector Machine

SVM with RBF Kernel

Compared with Random Forest on test set

🔹 Clustering

KMeans Clustering

Optimal k selected using Elbow Method

Silhouette Score evaluation

🔹 PCA

Dimensionality reduction to 2D

Visualization of clusters

🔹 Neural Network

Multi-Layer Perceptron (64, 32 hidden layers)

ReLU activation

Early stopping enabled

🔹 Web Application

Built using Streamlit

Real-time price prediction

Output displayed in INR

Category classification (Low / Medium / High)

🛠 Tech Stack

Python

NumPy

Pandas

Scikit-learn

Matplotlib

Seaborn

Streamlit

Joblib
