import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from src.pipeline import read_and_split
from src.modelling import model_and_evaluate

def main():
    X_train, X_test, y_train, y_test = read_and_split()
    model_name = input('Choose model name from one of the three options:\n'
                       '1. LogisticRegression\n'
                       '2. RandomForestClassifier\n'
                       '3. GradientBoostingClassifier\n'
                        'Enter the model name as displayed:')
    
    valid_models = ['LogisticRegression', 'RandomForestClassifier', 'GradientBoostingClassifier']
    if model_name in valid_models:
        accuracy, _ = model_and_evaluate(model_name)
    else: 
        print("Invalid input.")
        return None
    return accuracy  

if __name__=='__main__':
    print(f'Executing from {__name__}')
    result = main()
    if result is not None:
        print(f'Accuracy is:{result}')