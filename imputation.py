import numpy as np
import pandas as pd

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer

def impute(df, strategy):
    if strategy == 'simple':
    	imputer = SimpleImputer(strategy='mean')
    elif strategy == 'knn':
        imputer = KNNImputer()
    elif strategy == 'iterative':
    	imputer = IterativeImputer(n_nearest_features=100, verbose=5, tol=1e-5)
    else:
        print('no valid strategy selected. Available imputation strategies: simple, knn or iterative')
    imputed = imputer.fit_transform(df)
    imputed_df = pd.DataFrame(data=imputed)
    return imputed_df
