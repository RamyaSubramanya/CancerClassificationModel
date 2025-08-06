import sys
import os

# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline import read_and_preprocess, split_train_test
from src.modelling import model_and_evaluate
import pandas as pd

def test_model():
    X, y = read_and_preprocess()
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    accuracy, predictions = model_and_evaluate(model_name='LogisticRegression')

    # Step 3: Simple functional checks
    assert len(predictions)>0, "Predictions should not be empty"
    assert isinstance(predictions, list), "Predictions should be a list."
    assert len(predictions) == len(y_test), "Predictions and test data should have same length"
    assert isinstance(accuracy, float), "Accuracy sould be a float"
    print("✅ All basic model checks passed!")