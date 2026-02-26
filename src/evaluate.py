import pandas as pd 
from sklearn.metrics import mean_squared_error,mean_absolute_error
import numpy as np

def evaluate_model(model,X_test,y_test):
    
    prediction = model.predict(X_test)

    mae = mean_absolute_error(y_test,prediction)
    rmse = np.sqrt(mean_squared_error(y_test,prediction))

    print("MAE :",mae)
    print("RMSE :",rmse)

    return prediction