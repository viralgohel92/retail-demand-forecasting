# Retail Demand Forecasting System

## Overview

This project is an end-to-end Machine Learning system that forecasts weekly retail sales for Walmart stores using historical data.
The system trains forecasting models, evaluates prediction accuracy, generates business decisions (inventory actions), and provides a live prediction interface through a Streamlit web application.

The model predicts **next week's sales demand** for a selected store using past sales behavior and external economic factors.

---

## Problem Statement

Retail stores must decide inventory levels before demand actually occurs.
If inventory is too high → storage cost & wastage.
If inventory is too low → stockouts & revenue loss.

This system helps a store manager answer:

> “How much demand should I expect next week and should I increase or reduce inventory?”

---

## Dataset

Walmart Store Sales Dataset

The dataset contains weekly store sales along with external influencing factors:

* Store ID
* Date
* Weekly Sales (Target Variable)
* Holiday Flag
* Temperature
* Fuel Price
* CPI
* Unemployment Rate

Data file location:

```
data/raw/Walmart.csv
```

Loaded using:
`src/data_loader.py`

---

## Machine Learning Approach

This is a **time-series regression forecasting** problem.

Instead of predicting randomly, the model learns patterns from past weeks using lag features (historical memory).

### Lag Features (Key Idea)

The model remembers previous sales behavior:

| Feature | Meaning           |
| ------- | ----------------- |
| lag_1   | Last week sales   |
| lag_2   | Sales 2 weeks ago |
| lag_4   | Monthly pattern   |
| lag_12  | Quarterly pattern |

These are created in:
`src/feature_engineering.py`

The model learns:

Sales(t) = f(previous sales + time seasonality + economic indicators)

---

## Feature Engineering

Time based features created:

* Year
* Month
* Week Number
* Quarter
* Day of Week

Lag features:

* lag_1
* lag_2
* lag_4
* lag_12

Files:

```
src/preprocessing.py
src/feature_engineering.py
```

---

## Train-Test Strategy (Important)

This project uses **Time Series Split**, NOT random split.

Why?
Random splitting leaks future data into training and produces fake accuracy.

Instead:

* Past data → Training
* Future data → Testing

Implemented in:
`src/train.py`

---

## Models Used

### 1. Baseline Model

Linear Regression
Used as a benchmark model.

### 2. Advanced Model

XGBoost Regressor
Chosen because it performs well on tabular forecasting data.

Training file:
`src/train.py`

Saved Models:

```
models/linear_regression_model.pkl
models/xgboost_model.pkl
```

---

## Evaluation Metrics

The models are evaluated using:

* MAE (Mean Absolute Error)
* RMSE (Root Mean Squared Error)

Implemented in:
`src/evaluate.py`

---

## Business Decision Engine

Predictions are converted into real business actions.

Logic:

| Demand Change | Action |
| ------------- | ------ |

> +5% | Increase Inventory |
> -5% | Decrease Inventory |
> Otherwise | Stable

Generated in:
`src/business.py`

Output example:

```
Store | Date | Actual | Predicted | Demand_change_% | Stock_risk
```

Reports saved:

```
reports/store_forecast_report.csv
reports/store_forecast_report_xgboost.csv
```

---

## Future Forecasting (Next Week Prediction)

The system performs **recursive forecasting**.

Steps:

1. Take the latest available store data
2. Shift lag features forward
3. Predict next week sales
4. Display result

Implemented in:
`src/future_forecast.py`

---

## Visualization

Actual vs predicted sales graph:
`src/visualize.py`

Feature importance plot:
`src/interpret.py`

---

## Streamlit Web Application

Interactive UI to forecast store demand.

Features:

* Select Store
* Predict next week sales
* View last 20 weeks trend

File:

```
app/app.py
```

Run the web app:

```
streamlit run app/app.py
```

---

## How to Run the Project

### 1) Create Virtual Environment

```
python -m venv venv
```

Activate:

```
venv\Scripts\activate
```

### 2) Install Dependencies

```
pip install -r requirements.txt
```

### 3) Train Models

```
python main.py
```

### 4) Launch Forecast Web App

```
streamlit run app/app.py
```

---

## Project Structure

```
retail_demand_forecasting/
│
├── app/                → Streamlit UI
├── src/                → ML pipeline modules
├── data/
│   ├── raw/
│   └── processed/
├── models/             → Saved ML models
├── reports/            → Business reports
├── main.py             → Training pipeline
└── requirements.txt
```

---

## Technologies Used

* Python
* Pandas
* Scikit-Learn
* XGBoost
* Matplotlib
* Joblib
* Streamlit

---

## What This Project Demonstrates

* Time Series Forecasting
* Feature Engineering
* Model Evaluation
* Model Persistence
* Business Decision Automation
* ML Deployment (Streamlit)

---

## Future Improvements

* Multi-week forecasting
* Holiday season modeling
* Hyperparameter tuning
* API deployment (FastAPI)
* Cloud deployment

---


## Author
Viral Gohel  
Computer Science Student | Machine Learning Enthusiast

