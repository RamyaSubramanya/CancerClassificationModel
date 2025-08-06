import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

def read_and_split():
    """
    Read the data, split the data into train and test.
    
    Returns: 
        dataframes: Independent and target variables.
        
    """
    base_dir = os.getcwd()
    data_path = os.path.join(base_dir, 'data', 'Breast_cancer_data.csv')
    data = pd.read_csv(data_path)
    print("Data has been received.")
    
    X = data.drop(columns='diagnosis')
    y = data['diagnosis']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
    print("Data has been split into train and test")
    
    return X_train, X_test, y_train, y_test