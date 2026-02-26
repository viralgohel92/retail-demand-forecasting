import sys
from pathlib import Path

# allow app to access src folder
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import streamlit as st
import pandas as pd
import joblib

from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.feature_engineering import create_time_feature, create_lag_feature
from src.future_forecast import forecast_next_week
from src.train import feature_column
from src.business import generate_business_insights


st.set_page_config(page_title="Retail Demand Forecasting", layout="wide")

st.title("Retail Demand Forecasting System")

# ---------------- LOAD MODEL ----------------
model = joblib.load("models/xgboost_model.pkl")

# ---------------- LOAD DATA ----------------
df = load_data()
df = preprocess_data(df)
df = create_time_feature(df)
df = create_lag_feature(df)
df = df.dropna()

# ---------------- STORE SELECT ----------------
stores = sorted(df['Store'].unique())
store_id = st.selectbox("Select Store", stores)

store_df = df[df['Store'] == store_id].copy()

# ---------------- FORECAST ----------------
future_date, future_prediction = forecast_next_week(store_df, model, feature_column)

st.subheader("Next Week Sales Forecast")

col1, col2 = st.columns(2)

with col1:
    st.write(f"**Store:** {store_id}")
    st.write(f"**Forecast Week:** {future_date.strftime('%d %B %Y')}")

with col2:
    st.metric("Expected Sales", f"{int(future_prediction):,}")

# ---------------- BUSINESS INSIGHTS ----------------
st.subheader("Inventory Decision Recommendation")

# take last available row
last_row = store_df.sort_values("Date").iloc[-1:].copy()

# attach prediction
last_row["Predicted"] = future_prediction

# run business logic
insight_df = generate_business_insights(last_row)

decision = insight_df.iloc[0]["Stock_risk"]
change_percent = insight_df.iloc[0]["Demand_change_%"]

# decision color
if "Increase" in decision:
    st.error(f"⚠ {decision}")
elif "Decrease" in decision:
    st.warning(f"⬇ {decision}")
else:
    st.success(f"✔ {decision}")

st.write(f"Expected Demand Change: **{change_percent:.2f}%**")

# ---------------- SALES TREND ----------------
st.subheader("Recent Sales Trend (Last 20 Weeks)")

recent_data = store_df.sort_values("Date").tail(20)
st.line_chart(recent_data.set_index("Date")["Weekly_Sales"])