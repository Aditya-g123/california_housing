import streamlit as st

import streamlit as st
import numpy as np
import pickle
import pandas as pd

st.set_page_config(page_title="California Housing Price Predictor", layout="centered")

@st.cache_data
def load_model():
    model = pickle.load(open('regmodel.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    return model, scaler

model, scaler = load_model()

st.title("California Housing Price Predictor")
st.write("Enter the features below to predict the median house price (target in $100k units).")

col1, col2 = st.columns(2)
with col1:
    MedInc = st.number_input('Median Income (MedInc)', value=8.3252, step=0.01)
    HouseAge = st.number_input('House Age (HouseAge)', value=41.0, step=0.1)
    AveRooms = st.number_input('Average Rooms (AveRooms)', value=6.9841, step=0.01)
    AveBedrms = st.number_input('Average Bedrooms (AveBedrms)', value=1.0238, step=0.01)

with col2:
    Population = st.number_input('Population', value=322.0, step=1.0)
    AveOccup = st.number_input('Average Occupancy (AveOccup)', value=2.5556, step=0.01)
    Latitude = st.number_input('Latitude', value=37.88, step=0.01)
    Longitude = st.number_input('Longitude', value=-122.23, step=0.01)

if st.button('Predict'):
    input_data = np.array([[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]])
    input_scaled = scaler.transform(input_data)
    pred = model.predict(input_scaled)
    st.success(f"Predicted Median House Price: ${pred[0]*100000:,.2f}")

st.markdown("---")

st.write("Model summary:")
st.write(str(model))

# Example input shown as a table
st.markdown("#### Example input")
st.table(pd.DataFrame([{
    'MedInc': MedInc,
    'HouseAge': HouseAge,
    'AveRooms': AveRooms,
    'AveBedrms': AveBedrms,
    'Population': Population,
    'AveOccup': AveOccup,
    'Latitude': Latitude,
    'Longitude': Longitude
}]))
