import pandas as pd 

def create_time_feature(df):

    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['week'] = df['Date'].dt.isocalendar().week.astype(int)
    df['day_of_week'] = df['Date'].dt.dayofweek 
    df['quarter'] = df['Date'].dt.quarter 

    return df 


def create_lag_feature(df):
    
    #lag = past data 
    #last week sales 
    df['lag_1'] = df.groupby('Store')['Weekly_Sales'].shift(1)

    # 2 week ago 
    df['lag_2'] = df.groupby('Store')['Weekly_Sales'].shift(2)

    # 4 week ago (monthly pattern)
    df['lag_4'] = df.groupby('Store')['Weekly_Sales'].shift(4)

    # 12 week ago (quarterly pattern)
    df['lag_12'] = df.groupby('Store')['Weekly_Sales'].shift(12)

    return df 