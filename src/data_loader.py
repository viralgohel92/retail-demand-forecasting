import pandas as pd 
from pathlib import Path

Data_path = Path("data/raw/Walmart.csv")

def load_data():
    df = pd.read_csv(Data_path)
    return df 

