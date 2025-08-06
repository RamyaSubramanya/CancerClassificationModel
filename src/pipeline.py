import pandas as pd
from sklearn.model_selection import train_test_split

def read_and_preprocess():
    """
    Read the data, split the data into Independent and target variables.
    
    Returns: 
        dataframes: Independent and target variables.
        
    """
    data = pd.read_csv(r'D:\Data Science\Machine Learning & Deep Learning ANN (Regression & Classification)\Classification Practicals\BreastCancerPrediction\data\Breast_cancer_data.csv')
    X = data.drop(columns='diagnosis')
    y = data['diagnosis']
    print("Data has been received.")
    return X, y

def split_train_test(X, y):
    """
    Splits independent and target variables into train, test datasets on 30:30 ratio
    
    Args:
        X (dataframe): Independent variable
        y (dataframe): Target variable

    Returns:
        dataframe: independent and target variables split into train and test datasets
    """

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
    print("Data has been split into train and test")
    
    return X_train, X_test, y_train, y_test