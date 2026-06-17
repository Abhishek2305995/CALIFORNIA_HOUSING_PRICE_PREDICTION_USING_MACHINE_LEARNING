import streamlit as st
import numpy as np
import joblib

# ---------------- LOAD MODELS ----------------
# ---------------- LOAD MODELS ----------------
reg_model = joblib.load("regression_model.pkl")

logistic_model = joblib.load("logistic_model.pkl")
decision_tree_model = joblib.load("decision_tree_model.pkl")
random_forest_model = joblib.load("classification_model.pkl")

kmeans_model = joblib.load("kmeans_model.pkl")

scaler = joblib.load("scaler.pkl")
cluster_scaler = joblib.load("cluster_scaler.pkl")

neural_network = joblib.load("neural_network_model.pkl")

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Housing Prediction App", layout="centered")

st.title("🏠 California Housing Price Prediction")
st.markdown("Enter housing details below to predict price, category, and region.")

# ---------------- INPUT SECTION ----------------
st.subheader("📊 Enter Housing Features")

MedInc = st.number_input("Median Income", value=3.0)
HouseAge = st.number_input("House Age", value=20.0)
AveRooms = st.number_input("Average Rooms", value=5.0)
AveBedrms = st.number_input("Average Bedrooms", value=1.0)
Population = st.number_input("Population", value=1000.0)
AveOccup = st.number_input("Average Occupancy", value=3.0)
Latitude = st.number_input("Latitude", value=34.0)
Longitude = st.number_input("Longitude", value=-118.0)

# ---------------- MODEL SELECTION ----------------
st.subheader("🤖 Select Classification Model")

selected_model = st.selectbox(
    "Choose Model",
    ["Logistic Regression", "Decision Tree", "Random Forest"]
)

# ---------------- PREDICT BUTTON ----------------
if st.button("Predict"):

    features = [
        MedInc,
        HouseAge,
        AveRooms,
        AveBedrms,
        Population,
        AveOccup,
        Latitude,
        Longitude
    ]

    features_array = np.array([features])

    # -------- Scale for regression & classification --------
    features_scaled = scaler.transform(features_array)

    # ---------------- Regression ----------------
    predicted_price = reg_model.predict(features_scaled)[0]
    predicted_price_inr = predicted_price * 100000

    # ---------------- Classification ----------------
    if selected_model == "Logistic Regression":
        model_used = logistic_model
    elif selected_model == "Decision Tree":
        model_used = decision_tree_model
    else:
        model_used = random_forest_model

    class_prediction = model_used.predict(features_scaled)[0]

    if class_prediction == 0:
        predicted_class = "Low"
    elif class_prediction == 1:
        predicted_class = "Medium"
    else:
        predicted_class = "High"

    # ---------------- Clustering ----------------
    cluster_scaled = cluster_scaler.transform(features_array)
    cluster_number = kmeans_model.predict(cluster_scaled)[0]

    cluster_mapping = {
        0: "Affluent Coastal Residential Zone",
        1: "Mid-Income Urban Density Zone",
        2: "Emerging Inland Residential Area"
    }

    predicted_cluster = cluster_mapping.get(cluster_number, f"Region {cluster_number}")

    # ---------------- DISPLAY RESULTS ----------------
    st.subheader("📈 Prediction Results")

    st.success(f"💰 Predicted Price: ₹ {round(predicted_price_inr, 2)}")
    st.info(f"🏷 Category: {predicted_class}")
    st.warning(f"📍 Region Type: {predicted_cluster}")
    st.write(f"Model Used: {selected_model}")