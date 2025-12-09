
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title="AquaML Dashboard", layout="wide")

# ---------- Custom CSS Styling ----------
st.markdown("""
<style>

:root {
    --card-bg: #ffffff;
    --card-text: #000000;
    --card-shadow: 0 4px 12px rgba(0,0,0,0.06);
}

/* Dark mode overrides */
[data-theme="dark"] {
    --card-bg: #1e1e1e;
    --card-text: #f0f0f0;
    --card-shadow: 0 4px 12px rgba(255,255,255,0.06);
}

.card {
    padding: 20px;
    border-radius: 18px;
    background: var(--card-bg);
    color: var(--card-text);
    box-shadow: var(--card-shadow);
    height: 150px;
}

.card-green {
    background: linear-gradient(135deg, #0c8e70, #4cb070);
    color: white;
}

.card-title {
    font-size: 18px;
    font-weight: 600;
}

.card-value {
    font-size: 32px;
    font-weight: bold;
}

.subtext {
    font-size: 13px;
    opacity: 0.8;
}

.section-title {
    font-size: 22px;
    font-weight: 600;
    margin-top: 10px;
    margin-bottom: -10px;
}

</style>
""", unsafe_allow_html=True)



# IRRIGATION FORECAST BOX
with colB:
    st.markdown("""
        <div class="card" style="height:230px;">
            <div class="section-title">Irrigation Forecast – Next Hour</div>
            <br>
            Now: <b>12L</b> • 68%  
            <hr>
            +15 min: <b>8L</b> • 70%  
            <hr>
            +30 min: <b>5L</b> • 72%  
            <hr>
            +45 min: <b>0L</b> • 73%  
        </div>
    """, unsafe_allow_html=True)


# ---------- FORECAST SECTION WITH MODEL OUTPUT ----------
st.subheader("📊 ML Rainfall Forecasts")

colx, coly, colz = st.columns(3)

# Dummy placeholders — replace these with your model outputs
next_hour = np.random.uniform(0, 5)
next_day = np.random.uniform(0, 20)
next_week = np.random.uniform(0, 80)

with colx:
    st.metric("Rainfall Next Hour", f"{next_hour:.2f} mm")

with coly:
    st.metric("Rainfall Next 24 Hours", f"{next_day:.2f} mm")

with colz:
    st.metric("Rainfall Next 7 Days", f"{next_week:.2f} mm")


# ---------- GRAPH PLACEHOLDER ----------
st.subheader("📈 Irrigation & Rainfall Trends")
st.line_chart(np.random.randn(20, 3))
