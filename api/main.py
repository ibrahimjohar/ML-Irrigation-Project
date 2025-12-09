from pathlib import Path

from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model


app = FastAPI()

# Resolve paths relative to this file so the app works no matter the working dir.
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_PATH = BASE_DIR / "data" / "final_cleaned_district_dataset.csv"

# Load models
lstm_model = load_model(MODELS_DIR / "global_multistep_lstm.keras")
crop_model = joblib.load(MODELS_DIR / "Crop_Yield_RandomForest.joblib")
irr_model = joblib.load(MODELS_DIR / "Irrigation_Area_CatBoost.joblib")

# Load dataset (for lookup if needed)
df = pd.read_csv(DATA_PATH)

@app.get("/")
def root():
    return {"status": "API is running"}

@app.get("/predict")
def predict(district: str, year: int):
    try:
        row = df[(df["District"] == district) & (df["Year"] == year)]
        if row.empty:
            raise HTTPException(status_code=404, detail="District/Year not found")

        row = row.iloc[0]

        # -------------------------------
        # SAFE FEATURE ALIGNMENT (FIX)
        # -------------------------------
        # 1. Get training-time features
        crop_features = crop_model.feature_names_in_
        irr_features = irr_model.feature_names_in_

        # 2. Build prediction row with EXACT expected columns
        crop_input = row.reindex(crop_features, fill_value=0).to_frame().T
        irr_input = row.reindex(irr_features, fill_value=0).to_frame().T

        # 3. Predict
        crop_pred = float(crop_model.predict(crop_input)[0])
        irr_pred = float(irr_model.predict(irr_input)[0])
        # -------------------------------

        # LSTM rainfall predictions (already in dataset)
        next1 = float(row["Rain_Next_1"])
        next24 = float(row["Rain_Next_24"])
        next7 = float(row["Rain_Next_7d"])

        return {
            "district": district,
            "year": year,
            "rainfall": {
                "next_1h": next1,
                "next_24h_avg": next24,
                "next_7d_avg": next7
            },
            "crop_yield_prediction": crop_pred,
            "irrigation_area_prediction": irr_pred
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
