import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline import read_and_split
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')
import joblib
from azureml.core import Run
import json
import argparse
from azureml.core import Run

def model_and_evaluate():
    """
    Build Logistic Regression model and evaluate the model against the error metrics

    Returns:
        prints the accuracy of the model
    """
    X_train, X_test, y_train, y_test = read_and_split()

    model = LogisticRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("Predictions have been made.")
    print(f'Model that you selected is {model}')
    
    #check performance of the model
    accuracy = round(accuracy_score(y_test, predictions),2)*100
    
    run = Run.get_context()
    run.log("accuracy", accuracy)
    with open("outputs/metrics.json", "w") as f:
        json.dump({"accuracy": accuracy}, f)

    return accuracy, predictions, model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, default='outputs/model.pkl')
    args = parser.parse_args()
    accuracy, predictions, model = model_and_evaluate()
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    joblib.dump(model, args.output_path)
    
    # ✅ Upload model file to Azure ML output
    run = Run.get_context()
    run.upload_file(name='model_output/model.pkl', path_or_stream=args.output_path)
    
    print(f"Model Accuracy: {accuracy}")