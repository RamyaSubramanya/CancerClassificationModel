import sys
import os

# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline import read_and_split
from src.modelling import model_and_evaluate
import pandas as pd
import numpy as np

def test_model():
    X_train, X_test, y_train, y_test = read_and_split()
    accuracy, predictions = model_and_evaluate(model_name='LogisticRegression')

    # Step 3: Simple functional checks
    assert len(predictions)>0, "Predictions should not be empty"
    assert isinstance(predictions, np.ndarray), "Predictions should be a NumPy array."
    assert predictions.shape[0] == y_test.shape[0], "Predictions and y_test should have same number of samples"
    assert isinstance(accuracy, float), "Accuracy sould be a float"
    print("✅ All basic model checks passed!")