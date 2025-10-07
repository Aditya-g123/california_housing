import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

st.set_page_config(page_title="California Housing", layout="centered")

st.title("California Housing — Explorer & Price Predictor")


@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv"
    return pd.read_csv(url)


def load_model():
    model_path = Path("regmodel.pkl")
    scaler_path = Path("scaler.pkl")
    if model_path.exists() and scaler_path.exists():
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        return model, scaler
    return None, None


data = load_data()
model, scaler = load_model()

st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Explorer", "Predict"]) 

if page == "Explorer":
    st.sidebar.header("Filter Options")
    median_income = st.sidebar.slider(
        "Median Income", float(data["median_income"].min()), float(data["median_income"].max()), (2.0, 6.0)
    )
    housing_median_age = st.sidebar.slider(
        "Housing Median Age", int(data["housing_median_age"].min()), int(data["housing_median_age"].max()), (10, 30)
    )

    filtered_data = data[
        (data["median_income"] >= median_income[0]) & (data["median_income"] <= median_income[1]) &
        (data["housing_median_age"] >= housing_median_age[0]) & (data["housing_median_age"] <= housing_median_age[1])
    ]

    st.write(f"Showing {filtered_data.shape[0]} rows after filtering.")
    # st.map accepts latitude/longitude columns directly
    st.map(filtered_data[["latitude", "longitude"]])
    st.dataframe(filtered_data)

else:
    st.header("Predict House Price")

    st.markdown("Provide feature values below; the app will use `regmodel.pkl` and `scaler.pkl` from the repository if available.")

    cols = st.columns(2)
    with cols[0]:
        MedInc = st.number_input("Median Income (MedInc)", value=float(data["median_income"].median()))
        HouseAge = st.number_input("House Age", value=float(data["housing_median_age"].median()))
        # estimate average rooms/bedrooms per household for placeholders
        AveRooms = st.number_input("Average Rooms", value=float(data["total_rooms"].median() / (data["households"].median() if data["households"].median() != 0 else 1)))
        AveBedrms = st.number_input("Average Bedrooms", value=float(data["total_bedrooms"].median() / (data["households"].median() if data["households"].median() != 0 else 1)))
    with cols[1]:
        Population = st.number_input("Population", value=float(data["population"].median()))
        AveOccup = st.number_input("Average Occupancy", value=float(data["households"].median() if data["households"].median() != 0 else 1))
        Latitude = st.number_input("Latitude", value=float(data["latitude"].median()))
        Longitude = st.number_input("Longitude", value=float(data["longitude"].median()))

    if model is None or scaler is None:
        st.warning("Model files not found in the repo. Place `regmodel.pkl` and `scaler.pkl` in the project root to enable predictions.")
    else:
        if st.button("Predict"):
            # Arrange features in the expected order
            features = np.array([MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]).reshape(1, -1)
            features_scaled = scaler.transform(features)
            pred = model.predict(features_scaled)
            st.success(f"Predicted median house price: ${pred[0]*100000:,.0f}")
