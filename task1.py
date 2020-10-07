import numpy as np
import pandas as pd
from pathlib import Path

xTrainPath = Path('X_train.csv')
yTrainPath = Path('y_train.csv')
xTestPath = Path('X_test.csv')

def main():
    x_train, y_train, x_test = dataframes()

    # impute missing data?
    
    # add more features for non-linear relationships?
    

def dataframes():
    assert xTrainPath.exists() or yTrainPath.exists() or xTestPath.exists(), 'Wrong path.'
    
    x_train = pd.read_csv(xTrainPath, index_col=0)
    y_train = pd.read_csv(yTrainPath, index_col=0)
    x_test = pd.read_csv(xTestPath, index_col=0)
    
    return x_train, y_train, x_test

if __name__ == "__main__":
	main()
