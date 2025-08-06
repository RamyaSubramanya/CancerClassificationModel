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


def model_and_evaluate(model_name):
    """
    Build model and evaluate the model against the error metrics

    Args:
        model_name (str): Runs the algorithm based on the model name
    
    Returns:
        prints the accuracy of the model
    """
    X_train, X_test, y_train, y_test = read_and_split()

    #choose the model (LogisticRegression, RandomForest, GradientBoosting)
    if model_name=='LogisticRegression':
        model = LogisticRegression()
    elif model_name=='RandomForest':
        model = RandomForestClassifier()
    elif model_name=='GradientBoosting':
        model = GradientBoostingClassifier()
    
    print(f'Model that you selected is {model_name}')
    #fit the model on train data
    model.fit(X_train, y_train)

    #predictions on test data 
    predictions = model.predict(X_test)
    print("Predictions have been made.")
    
    #check performance of the model
    accuracy = round(accuracy_score(y_test, predictions),2)*100
    # print(f'Accuracy is:{accuracy}')
    return accuracy, predictions
