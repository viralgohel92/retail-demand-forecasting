import numpy as np

def generate_business_insights(store_df):
    store_df = store_df.copy()

    #calculate demand changes percentage
    store_df['Demand_change_%'] = ((store_df['Predicted']-store_df['lag_1'])/store_df['lag_1'])*100

    # risk detection 

    store_df["Stock_risk"] = np.where(
        store_df['Demand_change_%'] > 5 , "Hight Demand - Increase Inventory",
        np.where(store_df['Demand_change_%']< -5 , "Low Demand - Decrease Inventory","Stable")
    )

    return store_df