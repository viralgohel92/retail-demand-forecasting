import sys
from pathlib import Path

# add project root to python path
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


st.title("Retail Demand Forecasting System")

# load trained model
model = joblib.load("models/xgboost_model.pkl")

# load and prepare data
df = load_data()
df = preprocess_data(df)
df = create_time_feature(df)
df = create_lag_feature(df)
df = df.dropna()

# store selection
stores = sorted(df['Store'].unique())
store_id = st.selectbox("Select Store", stores)

store_df = df[df['Store'] == store_id].copy()


# -------- FUTURE FORECAST ONLY --------
future_date, future_prediction = forecast_next_week(store_df, model, feature_column)

st.subheader("Next Week Demand Forecast")

st.write("Store:", store_id)
st.write("Forecast Week:", future_date.date())
st.write("Expected Sales:", int(future_prediction))

# show last 20 weeks history for context
st.subheader("Recent Sales Trend")

recent_data = store_df.sort_values("Date").tail(20)
st.line_chart(recent_data.set_index("Date")["Weekly_Sales"])