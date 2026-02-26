import pandas as pd

def forecast_next_week(store_df, model, feature_cols):

    store_df = store_df.sort_values('Date')

    last_row = store_df.iloc[-1].copy()

    # create next week row
    next_week = last_row.copy()
    next_week['Date'] = last_row['Date'] + pd.Timedelta(days=7)

    # update time features
    next_week['year'] = next_week['Date'].year
    next_week['month'] = next_week['Date'].month
    next_week['week'] = next_week['Date'].isocalendar().week
    next_week['quarter'] = next_week['Date'].quarter
    next_week['day_of_week'] = next_week['Date'].dayofweek

    # shift lags
    next_week['lag_12'] = last_row['lag_4']
    next_week['lag_4']  = last_row['lag_2']
    next_week['lag_2']  = last_row['lag_1']
    next_week['lag_1']  = last_row['Weekly_Sales']

    X = pd.DataFrame([next_week[feature_cols]])

    prediction = model.predict(X)[0]

    return next_week['Date'], prediction