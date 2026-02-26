import pandas as pd 

def preprocess_data(df):
    df["Date"] = pd.to_datetime(df["Date"],dayfirst=True)
    df = df.sort_values(by = ['Store','Date'])

    return df 


def handle_missing_lags(df):

    # remove rows where lag features don't exist yet
    df = df.dropna(subset = ['lag_1','lag_2','lag_4','lag_12'])

    return df 