import streamlit as st
import requests

API_URL = "http://localhost:8000/predict"

st.title("🌾 ML Irrigation & Crop Forecasting Dashboard")

district = st.text_input("Enter District Code (e.g., PK203)")
year = st.number_input("Enter Year", min_value=1980, max_value=2030, value=2010)

if st.button("Predict"):
    params = {"district": district, "year": year}
    
    try:
        response = requests.get(API_URL, params=params)
        data = response.json()

        if "error" in data:
            st.error(data["error"])
        else:
            st.success("Prediction Successful!")

            # Show rainfall forecasts
            rain = data["rainfall"]
            st.subheader("🌦 Rainfall Forecast (From LSTM)")
            st.write(f"Next 1h: {rain['next_1h']:.3f}")
            st.write(f"Next 24h avg: {rain['next_24h_avg']:.3f}")
            st.write(f"Next 7 days avg: {rain['next_7d_avg']:.3f}")

            # Show ML predictions
            st.subheader("🌾 Crop & Irrigation Predictions")
            st.write(f"Crop Yield: {data['crop_yield_prediction']:.3f}")
            st.write(f"Irrigation Area: {data['irrigation_area_prediction']:.3f}")

    except Exception as e:
        st.error(f"Request failed: {e}")
