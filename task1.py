import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from pathlib import Path

xTrainPath = Path('X_train.csv')
yTrainPath = Path('y_train.csv')
xTestPath = Path('X_test.csv')

def main():
    x_train, y_train, x_test = dataframes()
    
    # NaN's are evenly distributed
#    nanInfo(x_train, 'x_train')
#    nanInfo(x_test, 'x_test')
    
    x_train = impute(x_train)
    x_test = impute(x_test)
    y_train.index = y_train.index.astype(int)
    
    nanInfo(x_train, 'x_train')
    nanInfo(x_test, 'x_test')
    
    print(x_train)
    print(y_train)
    print(x_test)


def impute(df):
    imputer = SimpleImputer(strategy='mean')	
    imputed = imputer.fit_transform(df)
    imputed_df = pd.DataFrame(data=imputed)
    return imputed_df


def nanInfo(df, name):
    print("number of NaN's for each column in %s:\n%s\n" % (name, df.isnull().sum(axis = 0)))
    df_size = df.size
    df_nan_size = df.isnull().sum().sum()
    if df_nan_size != 0:
        nan_percentage = df_size/df_nan_size
        print('percentage of missing data in %s:\t%s\n' % (name, int(nan_percentage)/100))
    else:
        print('percentage of missing data in %s:\t%s\n' % (name, str(0)))


def dataframes():
    assert xTrainPath.exists() or yTrainPath.exists() or xTestPath.exists(), 'Wrong path.'
    
    x_train = pd.read_csv(xTrainPath, index_col=0)
    y_train = pd.read_csv(yTrainPath, index_col=0)
    x_test = pd.read_csv(xTestPath, index_col=0)
    return x_train, y_train, x_test

if __name__ == "__main__":
	main()
