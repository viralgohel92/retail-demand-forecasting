import pandas as pd 
from sklearn.linear_model import LinearRegression
import joblib

def time_series_split(df):
    
    #last date in dataset
    last_date = df['Date'].max()
    
    # take last 20% of period as test
    split_date = df['Date'].quantile(0.80)

    train = df[df['Date'] <= split_date]
    test = df[df['Date'] > split_date]

    return train,test

feature_column = ['Holiday_Flag',
                  'Temperature',
                  'Fuel_Price',
                  'CPI', 
                  'Unemployment',
                  'year',
                  'month', 
                  'week',
                  'day_of_week',
                  'quarter', 
                  'lag_1', 
                  'lag_2',
                  'lag_4', 
                  'lag_12']

target_column = ['Weekly_Sales']

def preppare_data(train,test):

    X_train = train[feature_column]
    y_train = train[target_column]

    X_test = test[feature_column] 
    y_test = test[target_column]

    return X_train,X_test,y_train,y_test

from pathlib import Path

model_path = Path("models/linear_regression_model.pkl")

def train_baseline_model(X_train,y_train):

    model = LinearRegression()
    model.fit(X_train,y_train)

    # save model 
    model_path.parent.mkdir(parents = True , exist_ok = True)
    joblib.dump(model,model_path)

    print(f"model saved at {model_path}")

    return model


from xgboost import XGBRegressor

xgb_model_path = Path("models/xgboost_model.pkl")

def train_xgboost_model(X_train,y_train):

    model = XGBRegressor(n_estimators = 300,
                     learning_rate = 0.05,
                     max_depth = 6,
                     subsample = 0.8,
                     colsample_bytree=0.8,
                     random_state=42)
    
    model.fit(X_train,y_train)

    xgb_model_path.parent.mkdir(parents=True , exist_ok=True)
    joblib.dump(model,xgb_model_path)

    print(f"XGBoost model saved at {xgb_model_path}")


    return model