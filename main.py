import pandas as pd

# ---- Import all modules from our ML pipeline ----
from src.data_loader import load_data
from src.preprocessing import preprocess_data,handle_missing_lags
from src.feature_engineering import create_time_feature,create_lag_feature
from src.train import time_series_split,preppare_data,train_baseline_model,train_xgboost_model
from src.evaluate import evaluate_model
from src.visualize import plot_predictions
from src.business import generate_business_insights
from src.predict import load_model


# show more rows when printing dataframe (for debugging / inspection)
pd.set_option("display.max_rows",100)


# ============================================================
# STEP 1 — LOAD DATA
# Reads raw CSV file from data folder
# ============================================================
df = load_data()


# ============================================================
# STEP 2 — DATA PREPROCESSING
# - convert Date column to datetime
# - sort data store-wise and time-wise
# ============================================================
df = preprocess_data(df)


# ============================================================
# STEP 3 — FEATURE ENGINEERING
# Create time based features like:
# year, month, week, quarter, day_of_week
# ============================================================
df = create_time_feature(df)


# ============================================================
# STEP 4 — LAG FEATURES (IMPORTANT FOR FORECASTING)
# Adds memory to model:
# lag_1 = last week sales
# lag_2 = 2 weeks ago
# lag_4 = 1 month ago
# lag_12 = 3 months ago
# ============================================================
df = create_lag_feature(df)


# remove rows where lag values do not exist yet
df = handle_missing_lags(df)

df.to_csv("data/processed/training_dataset.csv", index=False)


# inspect prepared dataset
print(df.head(5))
print(df.info())


# ============================================================
# STEP 5 — TIME SERIES TRAIN/TEST SPLIT
# IMPORTANT:
# Not random split.
# Past = train
# Future = test
# ============================================================
train , test  = time_series_split(df)

print("Train rows:", len(train))
print("Test rows:", len(test))

# confirm no data leakage
print("Train last date:", train['Date'].max())
print("Test first date:", test['Date'].min())


# ============================================================
# STEP 6 — PREPARE FEATURES AND TARGET
# X = input features
# y = Weekly_Sales (target)
# ============================================================
X_train,X_test,y_train,y_test = preppare_data(train,test)


# ============================================================
# STEP 7 — TRAIN BASELINE MODEL (LINEAR REGRESSION)
# Used as benchmark model
# ============================================================
modoel = train_baseline_model(X_train,y_train)


# evaluate baseline model
prediction = evaluate_model(modoel,X_test,y_test)


# optional visualization (currently commented)
# plot_predictions(test,prediction,store_id=1)


# ============================================================
# STEP 8 — GENERATE BUSINESS INSIGHTS
# Converts predictions into business decisions:
# increase inventory / reduce inventory / stable
# ============================================================
test_df = test.copy()
test_df['Predicted'] = prediction

final_output = []

for store in test_df['Store'].unique():
    store_df = test_df[test_df['Store'] == store]
    store_df = generate_business_insights(store_df)
    final_output.append(store_df)

final_output = pd.concat(final_output)

print(final_output[['Store','Date','Weekly_Sales','Predicted','Demand_change_%','Stock_risk']].head(20))


# ============================================================
# STEP 9 — LOAD SAVED MODEL (MODEL PERSISTENCE CHECK)
# Demonstrates that saved model can be reused without retraining
# ============================================================
model = load_model()
prediction = model.predict(X_test)

print("Model loaded and Predictions generated.")


# ============================================================
# STEP 10 — TRAIN ADVANCED MODEL (XGBOOST)
# More powerful than linear regression for tabular forecasting
# ============================================================
print("\nTraining XGboost Model...")
xgb_model = train_xgboost_model(X_train,y_train)


# evaluate XGBoost
print("\nEvaluating XGBoost..")
xgb_predictions = evaluate_model(xgb_model,X_test,y_test)


# ============================================================
# STEP 11 — BUSINESS INSIGHTS USING XGBOOST PREDICTIONS
# ============================================================
test_df_xb = test.copy()
test_df_xb['Predicted'] = xgb_predictions

final_output_xb = []

for store in test_df_xb['Store'].unique():
    store_df_xb = test_df_xb[test_df_xb['Store'] == store]
    store_df_xb = generate_business_insights(store_df_xb)
    final_output_xb.append(store_df_xb)

final_output_xb = pd.concat(final_output_xb)



# ============================================================
# STEP 12 — MODEL INTERPRETATION
# Shows which features most influence predictions
# ============================================================
from src.interpret import plot_feature_importance
from src.train import feature_column

plot_feature_importance(xgb_model, feature_column)


# ============================================================
# STEP 13 — EXPORT REPORTS
# Saves forecast decision report as CSV files
# ============================================================
final_output.to_csv("reports/store_forecast_report.csv", index=False)
final_output.to_csv("reports/store_forecast_report_xgboost.csv", index=False)

print("Reports saved")