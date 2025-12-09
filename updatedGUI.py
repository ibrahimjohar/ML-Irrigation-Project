import streamlit as st
import pandas as pd
import pickle

# --- Load your trained ML model ---
with open("trained_model.pkl", "rb") as f:
    model = pickle.load(f)

# If you have a scaler or encoder used during training
# with open("scaler.pkl", "rb") as f:
#     scaler = pickle.load(f)

# --- Streamlit app ---
st.title("Irrigation Predictor 🌱")

st.write("Enter the following details to predict irrigation needs:")

# Inputs from user
temp = st.number_input("Temperature (°C)", min_value=-10.0, max_value=50.0, value=25.0)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0)
soil_moisture = st.number_input("Soil Moisture (%)", min_value=0.0, max_value=100.0, value=30.0)

crop_type = st.selectbox("Crop Type", ["Wheat", "Rice", "Corn"])
month = st.selectbox("Month", list(range(1, 13)))

# Optional: preprocessing if your model needs encoding/scaling
def preprocess_input(temp, humidity, soil_moisture, crop, month):
    df = pd.DataFrame({
        "temperature": [temp],
        "humidity": [humidity],
        "soil_moisture": [soil_moisture],
        "crop_type": [crop],  # encode if your model was trained with encoded values
        "month": [month]
    })
    
    # Example: if you used one-hot encoding for crop type
    # df = pd.get_dummies(df, columns=["crop_type"])
    
    # Example: if you used a scaler
    # df[["temperature","humidity","soil_moisture"]] = scaler.transform(df[["temperature","humidity","soil_moisture"]])
    
    return df

# Prediction
if st.button("Predict Irrigation"):
    input_data = preprocess_input(temp, humidity, soil_moisture, crop_type, month)
    prediction = model.predict(input_data)[0]
    st.success(f"💧 Predicted Irrigation: {prediction:.2f} mm")
