import streamlit as st
import numpy as np
import joblib

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="California Housing ML App",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 California Housing Prediction System")
st.markdown("### Predict House Price (INR) and Category")

st.write("Fill in the housing details below:")

# --------------------------------------------------
# Load Models
# --------------------------------------------------
reg_model = joblib.load("regression_model.pkl")
clf_model = joblib.load("classification_model.pkl")
scaler = joblib.load("scaler.pkl")

# --------------------------------------------------
# Center Layout
# --------------------------------------------------
col1, col2, col3 = st.columns([1, 2, 1])

with col2:

    MedInc = st.number_input("Median Income", min_value=0.0, value=5.0)
    HouseAge = st.number_input("House Age", min_value=0, value=20)
    AveRooms = st.number_input("Average Rooms", min_value=0.0, value=5.0)
    AveBedrms = st.number_input("Average Bedrooms", min_value=0.0, value=1.0)
    Population = st.number_input("Population", min_value=0, value=1000)
    AveOccup = st.number_input("Average Occupancy", min_value=0.0, value=3.0)
    Latitude = st.number_input("Latitude", value=36.0)
    Longitude = st.number_input("Longitude", value=-120.0)

    predict = st.button("🔍 Predict")

# --------------------------------------------------
# Prediction Logic
# --------------------------------------------------
if predict:

    features = np.array([[MedInc, HouseAge, AveRooms, AveBedrms,
                          Population, AveOccup, Latitude, Longitude]])

    features_scaled = scaler.transform(features)

    # Regression Prediction
    pred_price = reg_model.predict(features_scaled)[0]
    usd_value = pred_price * 100000  # dataset stores in 100k USD
    inr_value = usd_value * 90.81    # USD → INR conversion

    # Classification Prediction
    pred_category = clf_model.predict(features_scaled)[0]
    labels = ["Low Value", "Medium Value", "High Value"]

    st.markdown("---")
    st.markdown("## 📊 Prediction Results")

    st.success(f"💰 Predicted House Price: $ {usd_value:,.2f}")
    st.success(f"💰 Predicted House Price: ₹ {inr_value:,.2f}")
    st.info(f"🏷 Housing Category: {labels[int(pred_category)]}")

    st.markdown("---")
    st.caption("Model: Random Forest (Classification) | Linear Regression (Price)")