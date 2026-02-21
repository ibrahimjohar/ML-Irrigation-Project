from pathlib import Path
from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from pydantic import BaseModel, conlist, Field
from typing import List, Optional, TypeAlias, Annotated

app = FastAPI()

# ---------------------------
# PATH SETUP (Docker-safe)
# ---------------------------
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

DATA_PATH = DATA_DIR / "final_cleaned_district_dataset.csv"
RAINFALL_PATH = DATA_DIR / "Rain_fall_in_Pakistan.csv"
SCALER_PATH = MODELS_DIR / "lstm_scaler.pkl"


LSTM_FEATURES = ["rfh", "rfh_avg", "r1h", "r3h", "n_pixels"]
SEQ_LEN = 60
FUTURE_STEPS = 24

# ---------------------------
# LOAD MODELS + SCALER
# ---------------------------
# (these files must exist inside MODELS_DIR when you build the docker image)
lstm_model = load_model(MODELS_DIR / "global_multistep_lstm.keras")
crop_model = joblib.load(MODELS_DIR / "Crop_Yield_RandomForest.joblib")
irr_model = joblib.load(MODELS_DIR / "Irrigation_Area_CatBoost.joblib")
scaler_multi = joblib.load(SCALER_PATH)

# ---------------------------
# LOAD DATAFRAMES
# ---------------------------
# District-level dataset used for lookups
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

# Debug print so docker logs show what columns are present
print("District df columns:", df.columns.tolist())

# If Year missing, synthesize it from common alternatives
if "Year" not in df.columns:
    # try lowercase 'year'
    if "year" in df.columns:
        try:
            df["Year"] = df["year"].astype(int)
            print("Created df['Year'] from df['year'].")
        except Exception as e:
            raise RuntimeError(f"Failed to convert 'year' column to int: {e}")
    # try date column
    elif "date" in df.columns:
        try:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df["Year"] = df["date"].dt.year
            print("Created df['Year'] from df['date'].")
        except Exception as e:
            raise RuntimeError(f"Failed to parse 'date' and create Year: {e}")
    else:
        # no suitable column found — fail fast with helpful message
        raise RuntimeError("District dataset missing 'Year' (no 'Year', 'year', or 'date' column found).")

# final check
if df["Year"].isnull().any():
    print("Warning: Some Year values are NaN. Here are sample rows with missing Year:")
    print(df[df["Year"].isna()].head().to_dict(orient="records"))

# Rainfall timeseries used for LSTM inputs
rain = pd.read_csv(RAINFALL_PATH)
rain = rain.iloc[1:].copy()  # remove metadata row
rain.columns = rain.columns.str.strip()

# ensure required columns exist and create Year
if "date" not in rain.columns:
    raise RuntimeError("Rainfall CSV must contain a 'date' column.")
if "ADM2_PCODE" not in rain.columns:
    raise RuntimeError("Rainfall CSV must contain an 'ADM2_PCODE' column.")

rain["date"] = pd.to_datetime(rain["date"], errors="coerce")
rain["Year"] = rain["date"].dt.year

# ensure the LSTM features exist
for f in LSTM_FEATURES:
    if f not in rain.columns:
        raise RuntimeError(f"Rainfall CSV missing expected feature column: {f}")

# keep only useful columns and fill missing values
rain = rain[["date", "ADM2_PCODE"] + LSTM_FEATURES].copy()
rain[LSTM_FEATURES] = rain[LSTM_FEATURES].bfill().ffill()

rain[LSTM_FEATURES] = rain[LSTM_FEATURES].apply(pd.to_numeric, errors="coerce")
if rain[LSTM_FEATURES].isna().any().any():
    raise RuntimeError("Rainfall features still contain NaN after fill — check data.")

rain = rain.sort_values(["ADM2_PCODE", "date"]).reset_index(drop=True)

# ---------------------------------------------------------------------
# Helper: build latest window of 60 timesteps (scaled) for a region/year
# ---------------------------------------------------------------------
def get_latest_window(region_code: str, year: int):
    """
    Returns None if not enough historical rows (< SEQ_LEN).
    Otherwise returns a numpy array shaped (1, SEQ_LEN, n_features) scaled with scaler_multi.
    """
    sub = rain[(rain["ADM2_PCODE"] == region_code) & (rain["Year"] < year)]
    if len(sub) < SEQ_LEN:
        return None

    window = sub[LSTM_FEATURES].tail(SEQ_LEN).values  # shape (60,5)
    # scale using saved scaler (expects shape (n_rows, n_features))
    window_scaled = scaler_multi.transform(window)
    return window_scaled.reshape(1, SEQ_LEN, len(LSTM_FEATURES))

# ---------------------------------------------------------------------
# Forecast helpers
# ---------------------------------------------------------------------
def forecast_24(region_code: str, year: int):
    window = get_latest_window(region_code, year)
    if window is None:
        return None
    preds_scaled = lstm_model.predict(window, verbose=0)[0]  # (24,)
    preds_actual = []
    for v in preds_scaled:
        # inverse transform: pad to full feature vector before inverse
        padded = np.array([[v, 0.0, 0.0, 0.0, 0.0]])
        inv = scaler_multi.inverse_transform(padded)[0][0]
        preds_actual.append(float(max(inv, 0.0)))
    return preds_actual  # list of 24 floats

def forecast_7d(region_code: str, year: int):
    window_scaled = get_latest_window(region_code, year)
    if window_scaled is None:
        return None

    all_preds = []
    w = window_scaled.copy()  # shape (1, 60, 5)
    # recursive forecasting 7 days (168 steps) by calling predict in FUTURE_STEPS blocks
    loops = (7 * 24) // FUTURE_STEPS
    for _ in range(loops):
        preds_scaled = lstm_model.predict(w, verbose=0)[0]  # (24,)
        # inverse-scale each
        preds_actual = []
        for v in preds_scaled:
            padded = np.array([[v, 0.0, 0.0, 0.0, 0.0]])
            inv = scaler_multi.inverse_transform(padded)[0][0]
            preds_actual.append(float(max(inv, 0.0)))
        all_preds.extend(preds_actual)
        # build new scaled rows where column0 = preds_scaled, others 0
        new_scaled = np.zeros((FUTURE_STEPS, len(LSTM_FEATURES)))
        new_scaled[:, 0] = preds_scaled
        combined = np.vstack([w.reshape(SEQ_LEN, len(LSTM_FEATURES)), new_scaled])
        w = combined[-SEQ_LEN:].reshape(1, SEQ_LEN, len(LSTM_FEATURES))
    return all_preds  # list of 168 floats

# ---------- Pydantic + helpers for user-supplied windows ----------
# each element is a list of 5 floats in the order LSTM_FEATURES
WindowRow = Annotated[
    list[float],
    Field(min_items=len(LSTM_FEATURES), max_items=len(LSTM_FEATURES))
]

WindowList = Annotated[
    list[WindowRow],
    Field(min_items=1, max_items=SEQ_LEN)
]

class WindowInput(BaseModel):
    district: Optional[str] = None
    window: WindowList
    year: Optional[int] = None  #optional metadata

def prepare_window_from_user(rows: List[List[float]]):
    """
    rows: list of 1..60 rows, each row length == len(LSTM_FEATURES)
    returns scaled np.array shaped (1, SEQ_LEN, n_features) ready for the model.
    Pads by repeating the earliest row if len < SEQ_LEN.
    """
    arr = np.array(rows, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != len(LSTM_FEATURES):
        raise ValueError(f"Each row must be length {len(LSTM_FEATURES)}")
    # pad at top if too short
    if arr.shape[0] < SEQ_LEN:
        pad_n = SEQ_LEN - arr.shape[0]
        pad_rows = np.repeat(arr[0:1, :], pad_n, axis=0)
        arr = np.vstack([pad_rows, arr])
    # now arr.shape == (60, n_features)
    scaled = scaler_multi.transform(arr)  # scaler expects (n_rows, n_features)
    return scaled.reshape(1, SEQ_LEN, len(LSTM_FEATURES))

# helper that computes 7-day recursive forecast starting from a scaled window
def forecast_7d_from_window(window_scaled):
    """
    window_scaled: (1, SEQ_LEN, n_features) scaled
    returns list of 168 floats (7*24)
    """
    w = window_scaled.copy()
    all_preds = []
    loops = (7 * 24) // FUTURE_STEPS
    for _ in range(loops):
        preds_scaled = lstm_model.predict(w, verbose=0)[0]  # (24,)
        # inverse scale each pred and clip
        preds_actual = []
        for v in preds_scaled:
            padded = np.array([[v] + [0.0] * (len(LSTM_FEATURES) - 1)])
            inv = scaler_multi.inverse_transform(padded)[0][0]
            preds_actual.append(float(max(inv, 0.0)))
        all_preds.extend(preds_actual)
        # append scaled preds to create new window for next iteration
        new_scaled = np.zeros((FUTURE_STEPS, len(LSTM_FEATURES)))
        new_scaled[:, 0] = preds_scaled
        combined = np.vstack([w.reshape(SEQ_LEN, len(LSTM_FEATURES)), new_scaled])
        w = combined[-SEQ_LEN:].reshape(1, SEQ_LEN, len(LSTM_FEATURES))
    return all_preds

def get_latest_window_latest(region_code: str):
    """
    Return the most recent (1, SEQ_LEN, n_features) scaled window for region.
    Returns None if not enough rows.
    """
    sub = rain[rain["ADM2_PCODE"] == region_code]
    if len(sub) < SEQ_LEN:
        return None
    window = sub[LSTM_FEATURES].tail(SEQ_LEN).values
    window_scaled = scaler_multi.transform(window)
    return window_scaled.reshape(1, SEQ_LEN, len(LSTM_FEATURES))

# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------
@app.get("/")
def root():
    return {"status": "API is running"}

@app.get("/predict")
def predict(district: str, year: int):
    try:
        # ---- Defensive: ensure district DataFrame has Year ----
        if "Year" not in df.columns:
            # try lowercase 'year'
            if "year" in df.columns:
                try:
                    df["Year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
                    print("INFO: created df['Year'] from df['year']")
                except Exception as e:
                    print("ERROR: failed to create df['Year'] from 'year':", e)
            # try a date column
            elif "date" in df.columns:
                try:
                    df["date"] = pd.to_datetime(df["date"], errors="coerce")
                    df["Year"] = df["date"].dt.year.astype("Int64")
                    print("INFO: created df['Year'] from df['date']")
                except Exception as e:
                    print("ERROR: failed to create df['Year'] from 'date':", e)
            else:
                raise HTTPException(status_code=500, detail="District dataset missing 'Year' (no 'Year','year' or 'date')")

        # convert Year to ints where possible
        if df["Year"].dtype.name == "object":
            df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")

        # ---- Defensive: ensure rainfall table has Year too ----
        if "Year" not in rain.columns:
            if "date" in rain.columns:
                rain["date"] = pd.to_datetime(rain["date"], errors="coerce")
                rain["Year"] = rain["date"].dt.year.astype("Int64")
                print("INFO: created rain['Year'] from rain['date']")
            else:
                raise HTTPException(status_code=500, detail="Rainfall dataset missing 'Year' (and no 'date' column)")

        # validate district/year existence
        if district not in df["District"].unique():
            raise HTTPException(status_code=404, detail=f"District '{district}' not found in district dataset")

        # now safe to filter using Year
        row = df[(df["District"] == district) & (df["Year"] == int(year))]
        if row.empty:
            raise HTTPException(status_code=404, detail="District/Year not found")
        row = row.iloc[0]

        # ---------- safe feature alignment for sklearn pipelines ----------
        crop_features = getattr(crop_model, "feature_names_in_", None)
        irr_features = getattr(irr_model, "feature_names_in_", None)
        if crop_features is None:
            crop_features = [c for c in df.columns if c not in ("District","Year","Crop_Yield","Irrigation_Area")]
            print("WARN: crop_model missing feature_names_in_, using fallback features:", crop_features)
        if irr_features is None:
            irr_features = [c for c in df.columns if c not in ("District","Year","Crop_Yield","Irrigation_Area")]
            print("WARN: irr_model missing feature_names_in_, using fallback features:", irr_features)

        crop_input = row.reindex(crop_features, fill_value=0).to_frame().T
        irr_input = row.reindex(irr_features, fill_value=0).to_frame().T

        crop_pred = float(crop_model.predict(crop_input)[0])
        irr_pred = float(irr_model.predict(irr_input)[0])

        # ---------- LSTM forecasting ----------
        # Make sure we can compute LSTM forecasts using rainfall Year
        pred24_list = None
        pred7_list = None
        try:
            pred24_list = forecast_24(district, year)
            pred7_list = forecast_7d(district, year)
        except KeyError as e:
            # explicitly detect missing 'Year' in rain and return a clear message
            print("ERROR during LSTM forecast:", e)
            raise HTTPException(status_code=500, detail="Rainfall data missing required 'Year' column for historical forecast")
        except Exception as e:
            # log for debugging, but allow crop/irr predictions to still return
            print("WARN: LSTM forecasting failed, returning crop/irr predictions only. Details:", e)

        if pred24_list is None:
            next1 = next24 = next7 = None
            last60 = []
        else:
            next1 = float(pred24_list[0])
            next24 = float(np.mean(pred24_list))
            next7 = float(np.mean(pred7_list)) if (pred7_list is not None) else None
            last60 = rain[(rain["ADM2_PCODE"] == district) & (rain["Year"] < int(year))].tail(SEQ_LEN)["rfh"].tolist()

        return {
            "district": district,
            "year": int(year),
            "rainfall": {
                "last_60h": last60,
                "pred_24h": pred24_list or [],
                "pred_7d": pred7_list or [],
                "next_1h": next1,
                "next_24h_avg": next24 if pred24_list is not None else None,
                "next_7d_avg": next7 if pred7_list is not None else None,
            },
            "crop_yield_prediction": crop_pred,
            "irrigation_area_prediction": irr_pred,
        }

    except HTTPException:
        raise
    except Exception as exc:
        # Log the traceback in container logs for debugging
        import traceback
        traceback.print_exc()
        # Don't leak full internals in API responses on production
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/predict_now")
def predict_now(district: str):
    # get latest scaled window (no year)
    window = get_latest_window_latest(district)
    if window is None:
        raise HTTPException(status_code=404, detail="Not enough history for this district to make forecast")

    preds24_scaled = lstm_model.predict(window, verbose=0)[0]  # (24,)
    preds24_actual = [ float(max(scaler_multi.inverse_transform([[v] + [0.0]*(len(LSTM_FEATURES)-1)])[0,0], 0.0)) for v in preds24_scaled ]
    preds7_actual = forecast_7d_from_window(window)

    # also include last 60 rfh for UX (optional)
    last60 = rain[rain["ADM2_PCODE"] == district].tail(SEQ_LEN)["rfh"].tolist()

    return {
        "district": district,
        "forecast_24h": preds24_actual,
        "forecast_7d": preds7_actual,
        "last_60h_rfh": last60,
        "forecast_source": "computed"
    }

@app.post("/predict_with_window")
def predict_with_window(payload: WindowInput):
    # validate & prepare
    try:
        window_scaled = prepare_window_from_user(payload.window)  # (1,60,5)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid window: {e}")

    preds24_scaled = lstm_model.predict(window_scaled, verbose=0)[0]
    preds24_actual = [ float(max(scaler_multi.inverse_transform([[v] + [0.0]*(len(LSTM_FEATURES)-1)])[0,0], 0.0)) for v in preds24_scaled ]
    preds7_actual = forecast_7d_from_window(window_scaled)

    return {
        "district": payload.district,
        "forecast_24h": preds24_actual,
        "forecast_7d": preds7_actual,
        "forecast_source": "user_window"
    }