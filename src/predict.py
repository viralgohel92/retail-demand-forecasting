import joblib 
from pathlib import Path

model_path  = Path("models/linear_regression_model.pkl")

def load_model():
    model = joblib.load(model_path)
    return model