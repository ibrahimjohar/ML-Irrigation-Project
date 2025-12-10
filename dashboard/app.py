# dashboard/app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import altair as alt
import pydeck as pdk
import requests
import io
import base64
from datetime import datetime
from jinja2 import Template

# Use the full screen width to avoid cramped charts/cards
st.set_page_config(page_title="ML Irrigation Dashboard", page_icon="🌾", layout="wide")

# -------------------------
# CONFIG / PATHS / API
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATHS = [
    os.path.join(BASE_DIR, "..", "data", "final_cleaned_district_dataset.csv"),
    os.path.join(BASE_DIR, "data", "final_cleaned_district_dataset.csv"),
    "/app/data/final_cleaned_district_dataset.csv"
]

API_PREDICT = "http://localhost:8000/predict"
API_NOW = "http://localhost:8000/predict_now"
API_WINDOW = "http://localhost:8000/predict_with_window"

# -------------------------
# TRY TO LOAD DATASET (robust)
# -------------------------
df_codes = None
for p in DATA_PATHS:
    if os.path.exists(p):
        try:
            df_codes = pd.read_csv(p)
            df_codes.columns = df_codes.columns.str.strip()
            break
        except Exception as e:
            st.sidebar.error(f"Failed to read dataset at {p}: {e}")

if df_codes is None:
    st.sidebar.error("Dataset not found. Put final_cleaned_district_dataset.csv in ../data or /app/data")
    st.stop()

# -------------------------
# PREP DATA
# -------------------------
# minimal cleaning
if "Year" not in df_codes.columns and "year" in df_codes.columns:
    df_codes["Year"] = df_codes["year"].astype(int)
df_codes["Year"] = df_codes["Year"].astype(int)

DISTRICTS = sorted(df_codes["District"].unique())

# Mock lat/lon mapping (if you have a district-to-latlon mapping, load it)
# For demo: create small random coords — replace with real geometry for production
np.random.seed(0)
district_coords = {}
for i, d in enumerate(DISTRICTS):
    # keep coords inside Pakistan bounding box approx: lat 23-37, lon 60-77
    district_coords[d] = {
        "lat": 23 + np.random.rand() * 14,
        "lon": 60 + np.random.rand() * 17
    }

# -------------------------
# HELPERS
# -------------------------
def api_get(url, params):
    try:
        r = requests.get(url, params=params, timeout=10)
        return r.json()
    except Exception as e:
        return {"error": f"API error: {e}"}

def labeled_chart(values, title, ylabel, height=280):
    df = pd.DataFrame({"Time Step": list(range(len(values))), ylabel: values})
    chart = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X("Time Step", title="Time Step"),
            y=alt.Y(ylabel, title=ylabel),
            tooltip=[alt.Tooltip("Time Step"), alt.Tooltip(ylabel)]
        )
        .properties(title=title, height=height, padding={"left": 16, "right": 12, "top": 10, "bottom": 16})
    )
    st.altair_chart(chart, use_container_width=True)

def sparkline(values, color="#66c2a5", height=60, width=200):
    df = pd.DataFrame({"x": list(range(len(values))), "y": values})
    chart = (
        alt.Chart(df)
        .mark_area(opacity=0.3, color=color)
        .encode(x="x:Q", y="y:Q")
        .properties(height=height, width=width)
    )
    return chart

def mini_spark(values, color="#66c2a5", height=160):
    """Compact sparkline without axes for KPI-style cards."""
    df = pd.DataFrame({"x": list(range(len(values))), "y": values})
    base = alt.Chart(df)
    area = base.mark_area(opacity=0.15, color=color)
    line = base.mark_line(color=color, strokeWidth=2)
    return (
        (area + line)
        .encode(
            x=alt.X(
                "x:Q",
                axis=alt.Axis(
                    title="Years (tail)",
                    labels=True,
                    ticks=True,
                    tickCount=5,
                    labelOpacity=0.7,
                    titleFontSize=10,
                    labelFontSize=10,
                ),
            ),
            y=alt.Y(
                "y:Q",
                axis=alt.Axis(
                    title=None,
                    labels=True,
                    labelOpacity=0.75,
                    labelColor="#d6d6d6",
                    labelFontSize=10,
                    grid=True,
                ),
            ),
            tooltip=[alt.Tooltip("y:Q", title="Value")]
        )
        .properties(height=height, padding={"left": 18, "right": 12, "top": 6, "bottom": 14})
    )

# PDF generation helper (simple HTML -> PDF using WeasyPrint if available)
def render_pdf_from_html(html_str):
    try:
        # prefer weasyprint if installed
        from weasyprint import HTML
        pdf_bytes = HTML(string=html_str).write_pdf()
        return pdf_bytes
    except Exception as e:
        # fallback: return None — user must install weasyprint/wkhtmltopdf
        st.warning("WeasyPrint not available; install it to enable PDF export.")
        return None

def csv_download_button(df, filename="export.csv", label="Download CSV"):
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    b64 = base64.b64encode(csv_bytes).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{label}</a>'
    st.markdown(href, unsafe_allow_html=True)

# -------------------------
# THEME CSS (optional tweaks)
# -------------------------
st.markdown(
    """
    <style>
    .stApp { background-color: #0f1720; color: #fff; }
    .stMetric { color: #fff; }
    .kpi-card { background:#0b1220; padding:12px; border-radius:10px; }
    .section-card { background:#0b1220; padding:16px; border-radius:12px; }
    .chart-card { background:#0b1220; padding:12px; border-radius:10px; }
    /* Sidebar button tweaks */
    .stButton > button {
        border-radius: 10px;
        border: 1px solid #223448;
        background: linear-gradient(90deg, #1f2a3c, #182232);
        color: #e6eef7;
    }
    .stButton > button:hover {
        border-color: #4b9fff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------
# SIDEBAR NAVIGATION (button-based, no radio)
# -------------------------
NAV_ITEMS = [
    ("Home", "🌐 Home Dashboard"),
    ("Forecast (By Year)", "📌 Forecast by Year"),
    ("Real-time", "⏳ Real-time"),
    ("Custom Window", "🧪 Custom Window"),
    ("Model Info", "🧠 Model Info"),
    ("About", "ℹ️ About"),
]

if "nav" not in st.session_state:
    st.session_state.nav = "Home"

st.sidebar.markdown("### Navigation")
for nav_key, nav_label in NAV_ITEMS:
    is_active = st.session_state.nav == nav_key
    btn_type = "primary" if is_active else "secondary"
    if st.sidebar.button(nav_label, key=f"nav-{nav_key}", use_container_width=True, type=btn_type):
        st.session_state.nav = nav_key

menu = st.session_state.nav

# -------------------------
# HOME DASHBOARD (cards + mini charts)
# -------------------------
if menu == "Home":
    st.header("🌾 ML Irrigation — Home Dashboard")

    # KPIs row
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Districts", len(DISTRICTS))
    k2.metric("Years", f"{df_codes['Year'].min()} - {df_codes['Year'].max()}")
    k3.metric("Avg Rainfall (mm)", f"{df_codes['Avg_Rainfall'].mean():.2f}")
    k4.metric("Avg Irrigation Area", f"{df_codes['Irrigation_Area'].mean():.2f}")

    st.divider()

    # Trends block
    trend_left, trend_right = st.columns([2.6, 2.4])
    with trend_left:
        st.markdown("### 📈 Sample District Trend")
        sample_d = DISTRICTS[0] if DISTRICTS else None
        if sample_d:
            sample_df = df_codes[df_codes["District"] == sample_d].sort_values("Year")
            c = (
                alt.Chart(sample_df)
                .mark_line(point=True)
                .encode(
                    x=alt.X("Year:O", title="Year"),
                    y=alt.Y("Avg_Rainfall:Q", title="Avg Rainfall"),
                    tooltip=["Year", "Avg_Rainfall"]
                )
                .properties(height=420, padding={"left": 20, "right": 14, "top": 12, "bottom": 36})
            )
            st.altair_chart(c, use_container_width=True)
            st.caption(f"Example district: {sample_d}")

            # Fill vertical space with quick stats
            st.markdown("**District summary (last 30 yrs)**")
            tail = sample_df.tail(30)
            st.write(
                f"- Avg rainfall: {tail['Avg_Rainfall'].mean():.2f} | Min: {tail['Avg_Rainfall'].min():.2f} | Max: {tail['Avg_Rainfall'].max():.2f} | Δ: {tail['Avg_Rainfall'].iloc[-1] - tail['Avg_Rainfall'].iloc[0]:+.2f}\n"
                f"- Crop yield: {tail['Crop_Yield'].mean():.2f} | Irrigation area: {tail['Irrigation_Area'].mean():.2f}"
            )
        else:
            st.info("No district data available.")

    with trend_right:
        st.markdown("### Key Trends")
        kt_container = st.container(border=True)
        with kt_container:
            top_left, top_right = st.columns(2)
            with top_left:
                st.markdown("**Avg Rainfall (last 30 yrs)**")
                vals = df_codes.groupby("Year")["Avg_Rainfall"].mean().tail(30).values
                st.altair_chart(mini_spark(vals, color="#2e8b8b", height=180), use_container_width=True)
                st.caption(f"Mean: {np.mean(vals):.2f} mm  |  Min: {np.min(vals):.2f}  |  Max: {np.max(vals):.2f}  |  Δ: {vals[-1]-vals[0]:+.2f}")
            with top_right:
                st.markdown("**Crop Yield (last 30 yrs)**")
                vals = df_codes.groupby("Year")["Crop_Yield"].mean().tail(30).values
                st.altair_chart(mini_spark(vals, color="#c58b1b", height=180), use_container_width=True)
                st.caption(f"Mean: {np.mean(vals):.2f}  |  Min: {np.min(vals):.2f}  |  Max: {np.max(vals):.2f}  |  Δ: {vals[-1]-vals[0]:+.2f}")

            st.markdown("---")
            st.markdown("**Irrigation Area (last 30 yrs)**")
            vals = df_codes.groupby("Year")["Irrigation_Area"].mean().tail(30).values
            st.altair_chart(mini_spark(vals, color="#b23c4f", height=240), use_container_width=True)
            st.caption(f"Mean: {np.mean(vals):.2f}  |  Min: {np.min(vals):.2f}  |  Max: {np.max(vals):.2f}  |  Δ: {vals[-1]-vals[0]:+.2f}")

    st.divider()

    # Map + Distribution panels
    mcol1, mcol2 = st.columns([2.2, 1.1], gap="large")
    with mcol1:
        st.markdown("### 🗺️ District Map")
        map_df = pd.DataFrame(
            [{"district": d, "lat": district_coords[d]["lat"], "lon": district_coords[d]["lon"]} for d in DISTRICTS]
        )
        st.pydeck_chart(
            pdk.Deck(
                map_style="dark",
                initial_view_state=pdk.ViewState(latitude=30, longitude=68, zoom=4),
                layers=[
                    pdk.Layer(
                        "ScatterplotLayer",
                        data=map_df,
                        get_position='[lon, lat]',
                        get_radius=30000,
                        get_fill_color=[255, 140, 0],
                        pickable=True
                    )
                ],
                tooltip={"text": "{district}"}
            )
        )
        st.info("Tip: click a point on the map to see the district name in tooltip.")

    with mcol2:
        st.markdown("### 📊 Distributions")
        dist_container = st.container(border=True)
        with dist_container:
            st.caption("Crop Yield")
            st.altair_chart(
                alt.Chart(df_codes)
                .mark_bar(color="#6ab6ff", opacity=0.9)
                .encode(
                    alt.X("Crop_Yield:Q", bin=True, title="Crop Yield"),
                    alt.Y("count()", title="Count")
                )
                .properties(height=180, padding={"left": 10, "right": 10, "top": 10, "bottom": 10}),
                use_container_width=True
            )
            st.markdown("---")
            st.caption("Irrigation Area")
            st.altair_chart(
                alt.Chart(df_codes)
                .mark_bar(color="#9ad5a0", opacity=0.9)
                .encode(
                    alt.X("Irrigation_Area:Q", bin=True, title="Irrigation Area"),
                    alt.Y("count()", title="Count")
                )
                .properties(height=180, padding={"left": 10, "right": 10, "top": 10, "bottom": 10}),
                use_container_width=True
            )

    st.divider()

    # quick export
    st.markdown("### Export Data")
    csv_download_button(df_codes.head(1000), filename="district_preview.csv", label="Download preview CSV (1k rows)")

# -------------------------
# FORECAST BY YEAR
# -------------------------
elif menu == "Forecast (By Year)":
    st.header("📌 Forecast by District & Year")

    if not DISTRICTS:
        st.error("No district list.")
        st.stop()

    left, right = st.columns([1, 2])
    with left:
        district = st.selectbox("Select District", DISTRICTS)
        year = st.number_input("Year", min_value=int(df_codes["Year"].min()), max_value=int(df_codes["Year"].max())+5, value=int(df_codes["Year"].median()))
        run = st.button("Run Prediction")

    if run:
        data = api_get(API_PREDICT, {"district": district, "year": int(year)})
        if data.get("error") or data.get("detail"):
            st.error(data.get("error") or data.get("detail"))
        else:
            rain = data["rainfall"]
            st.subheader("Rainfall")
            st.metric("Next 1h", f"{rain['next_1h']:.2f} mm")
            st.metric("Next 24h avg", f"{rain['next_24h_avg']:.2f} mm")
            labeled_chart(rain["last_60h"], "Past 60h", "Rainfall (mm)")
            labeled_chart(rain["pred_24h"], "Next 24h", "Predicted Rainfall (mm)")
            st.subheader("ML Predictions")
            st.metric("Crop Yield", f"{data['crop_yield_prediction']:.2f}")
            st.metric("Irrigation Area", f"{data['irrigation_area_prediction']:.2f} ha")

            # Build a small HTML report and allow PDF download
            html = f"""
            <h1>Forecast Report</h1>
            <p>District: {district} — Year: {year}</p>
            <p>Crop Yield: {data['crop_yield_prediction']:.3f}</p>
            <p>Irrigation Area: {data['irrigation_area_prediction']:.3f}</p>
            <p>Next 1h Rain: {rain['next_1h']:.3f} mm</p>
            """
            pdf_bytes = render_pdf_from_html(html)
            if pdf_bytes:
                b64 = base64.b64encode(pdf_bytes).decode()
                href = f'<a href="data:application/pdf;base64,{b64}" download="forecast_report.pdf">Download PDF report</a>'
                st.markdown(href, unsafe_allow_html=True)
            # CSV download of the small payload:
            out_df = pd.DataFrame([{
                "district": district, "year": year,
                "crop_yield": data['crop_yield_prediction'],
                "irr_area": data['irrigation_area_prediction'],
                "next1": rain['next_1h']
            }])
            csv_download_button(out_df, filename="forecast_payload.csv", label="Download CSV")

# -------------------------
# REAL-TIME
# -------------------------
elif menu == "Real-time":
    st.header("⏳ Real-time (Latest Window)")

    district_now = st.selectbox("District", DISTRICTS, key="rt_district")
    if st.button("Get Now"):
        data = api_get(API_NOW, {"district": district_now})
        if data.get("detail") or data.get("error"):
            st.error(data.get("detail") or data.get("error"))
        else:
            st.metric("Next 1h", f"{data['forecast_24h'][0]:.2f} mm")
            labeled_chart(data['last_60h_rfh'], "Past 60h", "Rainfall (mm)")
            labeled_chart(data['forecast_24h'], "Next 24h", "Predicted Rainfall (mm)")

# -------------------------
# CUSTOM WINDOW
# -------------------------
elif menu == "Custom Window":
    st.header("🧪 Custom Window Forecast")
    st.write("Upload CSV with columns: rfh, rfh_avg, r1h, r3h, n_pixels (1..60 rows)")
    uploaded = st.file_uploader("CSV", type=["csv"])
    if uploaded:
        user_df = pd.read_csv(uploaded)
        st.write(user_df.head())
        if st.button("Run custom forecast"):
            payload = {"district": None, "year": None, "window": user_df.values.tolist()}
            res = requests.post(API_WINDOW, json=payload).json()
            if res.get("detail") or res.get("error"):
                st.error(res.get("detail") or res.get("error"))
            else:
                st.metric("Next 1h", f"{res['forecast_24h'][0]:.2f} mm")
                labeled_chart(res['forecast_24h'], "Next 24h (custom)", "Rainfall (mm)")
                labeled_chart(res['forecast_7d'], "Next 7d (custom)", "Rainfall (mm)")

# -------------------------
# MODEL INFO
# -------------------------
elif menu == "Model Info":
    st.header("🧠 Model & Training Info")
    st.markdown("- LSTM: global_multistep_lstm (multivariate, seq=60, future=24)")
    st.markdown("- Crop model: RandomForest (saved joblib)")
    st.markdown("- Irrigation model: CatBoost (saved joblib)")
    st.markdown("You can extend this section to show model metrics, training plots, hyperparams, and feature importances.")
    if st.button("Show feature importances (sample)"):
        # sample: read a saved CSV of feature importances if available
        st.info("Add code to fetch real feature importances from saved artifacts.")

# -------------------------
# ABOUT
# -------------------------
elif menu == "About":
    st.header("ℹ️ About This Project")
    st.write("""
    This dashboard visualizes rainfall, crop yield and irrigation area forecasts.
    - API endpoints: /predict, /predict_now, /predict_with_window
    - Built with Streamlit, FastAPI back-end and multivariate LSTM + tree-based models.
    """)
    st.write("Made by: You. Hostable in Docker.")
