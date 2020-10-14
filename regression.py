import numpy as np
from argparse import ArgumentParser
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import mean_absolute_error, r2_score
from load_data import dataframes, impute
from outlier_detection import detect_outliers

n_splits=3

parser = ArgumentParser()
parser.add_argument('--test', dest='test', action='store_true', default=False)
args = parser.parse_args()

#DATAFRAME
x_train, y_train, x_test = dataframes()
x_train = x_train.values
x_test = x_test.values
y_train = y_train.values.reshape(-1)

#IMPUTE
x_train = impute(x_train)

#OUTLIER DETECTION
x_train, y_train = detect_outliers(x_train, y_train)

#FEATURE SELECTION
scaler = StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
k_best = SelectKBest(f_classif, k=50).fit(x_train, y_train)
x_train = k_best.transform(x_train)

#CROSS-VALIDATION
if args.test:
    x_test = impute(x_test)
    x_test = scaler.transform(x_test)
    x_test = k_best.transform(x_test)
    reg = SVR(kernel='rbf', C=10.0)
    reg.fit(x_train, y_train)
    y_test = reg.predict(x_test)
    y_test = np.concatenate((
        np.arange(y_test.size).reshape(-1, 1),
        y_test.reshape(-1, 1)), axis=1)
    np.savetxt(fname='y_test.csv', header='id,y', delimiter=',', X=y_test,
            fmt=['%d', '%.5f'], comments='')
else:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    total_score = 0
    for train_idx, test_idx in kf.split(x_train):
        #reg = LinearRegression()
        #reg = Ridge(alpha=100)
        #reg = DecisionTreeRegressor()
        reg = SVR(kernel='rbf', C=10.0)
        reg.fit(x_train[train_idx], y_train[train_idx])
        y_pred = reg.predict(x_train[test_idx])
        total_score += r2_score(y_train[test_idx], y_pred)

    print('average score:', total_score/n_splits)
