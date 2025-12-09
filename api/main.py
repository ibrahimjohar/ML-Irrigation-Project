from fastapi import FastAPI
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

app = FastAPI()

# Load models
lstm_model = load_model("models/global_multistep_lstm.keras")
crop_model = joblib.load("models/Crop_Yield_RandomForest.joblib")
irr_model  = joblib.load("models/Irrigation_Area_CatBoost.joblib")

# Load dataset (for lookup if needed)
df = pd.read_csv("data/final_cleaned_district_dataset.csv")

@app.get("/")
def root():
    return {"status": "API is running"}

@app.get("/predict")
def predict(district: str, year: int):
    
    row = df[(df["District"] == district) & (df["Year"] == year)]
    if row.empty:
        return {"error": "District/Year not found"}

    row = row.iloc[0]

    #extract features for ML models
    ml_features = row[["Avg_Rainfall", "Avg_Temperature", "Rain_Next_1",
                       "Rain_Next_24", "Rain_Next_7d"]].values.reshape(1, -1)

    #predictions
    crop_pred = float(crop_model.predict(ml_features)[0])
    irr_pred  = float(irr_model.predict(ml_features)[0])

    #LSTM rainfall predictions are already stored in dataset
    next1  = float(row["Rain_Next_1"])
    next24 = float(row["Rain_Next_24"])
    next7  = float(row["Rain_Next_7d"])

    return {
        "district": district,
        "year": year,
        "rainfall":{
            "next_1h": next1,
            "next_24h_avg": next24,
            "next_7d_avg": next7
        },
        "crop_yield_prediction": crop_pred,
        "irrigation_area_prediction": irr_pred
    }
