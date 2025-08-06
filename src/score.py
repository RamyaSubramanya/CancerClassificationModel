import joblib
import pandas as pd
from src.modelling import model_and_evaluate

def init():
    global model
    model = joblib.load("model.pkl")  # or load it from local if passed with deployment

def run(raw_data):
    try:
        data = pd.DataFrame(raw_data)
        predictions = model.predict(data)
        return predictions.tolist()
    except Exception as e:
        return f"Error: {str(e)}"
