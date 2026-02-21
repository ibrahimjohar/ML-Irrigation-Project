# ML Irrigation Project

Forecast rainfall and predict crop yield and irrigation area at Pakistan ADM2 (district) level using a multivariate LSTM for time‑series and tree‑based models for tabular learning. Served via a FastAPI backend and explored with a Streamlit dashboard.

## Features
- Data pipeline for district‑level merging/cleaning and rainfall time‑series ingestion
- Multivariate LSTM (sequence length 60) for rainfall forecasting (24‑step horizon + recursive 7‑day)
- Tabular models (Random Forest, XGBoost, CatBoost) for `Crop_Yield` and `Irrigation_Area`
- FastAPI service exposing forecast and prediction endpoints
- Streamlit dashboard for interactive exploration and reports

## Repository Structure
```
.
├─ api/                     # FastAPI backend (models, data, Dockerfile)
│  ├─ data/                 # CSVs used by the API (district dataset, rainfall time series)
│  ├─ models/               # Inference artifacts (LSTM .keras/.pkl, joblib models expected)
│  ├─ main.py               # API entrypoint
│  ├─ requirements.txt
│  └─ Dockerfile
├─ dashboard/               # Streamlit dashboard
│  ├─ data/                 # Copy of district dataset for local runs
│  ├─ app.py
│  └─ requirements.txt
├─ notebooks/               # EDA, data merging, LSTM prep/training, tabular modeling
│  ├─ 04_lstm_preparation.ipynb
│  ├─ 05_global_multivariate_LSTM.ipynb
│  ├─ 06_merge_datasets.ipynb
│  ├─ 07_process_merged_data.ipynb
│  ├─ 08_add_lstm_features.ipynb
│  └─ 09_ml_modeling_pipeline.ipynb
├─ data/
│  ├─ geospatial/           # Pakistan shapefiles
│  └─ processed/            # Processed CSVs
├─ models/                  # LSTM model/scaler & training summary
├─ scripts/                 # Utilities (e.g., district mapping)
└─ requirements.txt         # Root environment for notebooks/dashboard
```

## Data
- District dataset (example columns): `District`, `Year`, `Avg_Rainfall`, `Avg_Temperature`, `Crop_Yield`, `Irrigation_Area`, `Rain_Next_1`, `Rain_Next_24`, `Rain_Next_7d`
- Rainfall time series: `date`, `ADM2_PCODE`, `n_pixels`, `rfh`, `rfh_avg`, `r1h`, `r3h` (and related aggregates)
- By default the API reads from `api/data/` and the dashboard looks for a copy at `dashboard/data/` or `../data/`

## Models
### Rainfall Forecasting (LSTM)
- Global multivariate LSTM trained across districts
- Sequence length: 60 timesteps; Forecast horizon: 24 (with recursive 7‑day helper)
- Artifacts used at inference time:
  - `api/models/global_multistep_lstm.keras`
  - `api/models/lstm_scaler.pkl`

### Tabular Predictors
- Targets: `Crop_Yield`, `Irrigation_Area`
- Candidate models: Random Forest, XGBoost, CatBoost (training results in `models/training_results.csv`)
- Runtime expectation: best model artifacts saved as `api/models/Crop_Yield_RandomForest.joblib` and `api/models/Irrigation_Area_CatBoost.joblib` (not committed by default due to `.gitignore` of `*.joblib`)

## API
Run the API locally or via Docker (see Quick Start). Endpoints:
- `GET /` — health/status
- `GET /predict?district=PK203&year=2018` — rainfall forecast + crop/irrigation predictions for a given district/year
- `GET /predict_now?district=PK203` — rainfall forecast from the latest available window (no year)
- `POST /predict_with_window` — forecast from a user‑provided 1..60×5 window of features (order: `rfh, rfh_avg, r1h, r3h, n_pixels`)

Example (PowerShell/CMD):
```bash
curl "http://localhost:8000/predict?district=PK203&year=2018"
```

## Dashboard
Streamlit app providing:
- Home: dataset KPIs, trends, distributions, and map
- Forecast (By Year): calls the API to show rainfall forecasts + ML predictions
- Real‑time: latest‑window forecast
- Custom Window: upload your own window for forecasting

## Quick Start
### Prerequisites
- Python 3.10 recommended

### Option A — Run API Locally
```bash
# at project root
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
# source venv/bin/activate

pip install -r api/requirements.txt
cd api
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
Notes:
- Ensure the following files exist: `api/models/global_multistep_lstm.keras`, `api/models/lstm_scaler.pkl`.
- For tabular predictions, place the trained `.joblib` files in `api/models/` as described above.

### Option B — Run API in Docker
```bash
cd api
docker build -t irrigation-api .
# Windows PowerShell volume mounts (adjust path as needed)
docker run --rm -p 8000:8000 ^
  -v "$PWD/models:/app/models" ^
  -v "$PWD/data:/app/data" ^
  irrigation-api
```

### Run the Dashboard
```bash
# New terminal at project root
python -m venv venv
venv\Scripts\activate
pip install -r dashboard/requirements.txt
streamlit run dashboard/app.py
```
Dashboard expects the API at `http://localhost:8000`. You can change the endpoints inside `dashboard/app.py` if needed.

## Troubleshooting
- API fails to start with a `FileNotFoundError` for `.joblib`: add the trained `Crop_Yield_*.joblib` and `Irrigation_Area_*.joblib` files under `api/models/`, or retrain/export them.
- Dataset not found in dashboard: put `final_cleaned_district_dataset.csv` in `dashboard/data/` or ensure the search paths in `dashboard/app.py` point to the right location.
- Port already in use: run `uvicorn`/Streamlit on a different port.

## Roadmap
- Consolidate training code into reusable modules (`preprocess/`, `features/`)
- Add docker‑compose for API + dashboard
- Expose model metrics and feature importances in the dashboard
- CI for linting/tests and basic API health checks

---

Maintainers: Ibrahim and collaborators.
