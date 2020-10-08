import numpy as np
from sklearn.preprocessing import scale
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from load_data import dataframes, impute
from outlier_detection import detect_outliers

x_train, y_train, x_test = dataframes()
x_train = x_train.values
y_train = y_train.values
x_train = impute(x_train)
kf = KFold(n_splits=10)
x_train_outliers = detect_outliers(x_train)
x_train = x_train[x_train_outliers==1]
y_train = y_train[x_train_outliers==1]
x_train = scale(x_train)

for train_idx, test_idx in kf.split(x_train):
    reg = LinearRegression()
    reg.fit(x_train[train_idx], y_train[train_idx])
    y_pred = reg.predict(x_train[test_idx])
    print(r2_score(y_train[test_idx], y_pred))
