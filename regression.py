import numpy as np
from argparse import ArgumentParser
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest, f_classif, RFECV, VarianceThreshold
from sklearn.metrics import mean_absolute_error, r2_score
from load_data import dataframes
from imputation import impute
from outlier_detection import detect_outliers
import matplotlib.pyplot as plt

n_splits=3

parser = ArgumentParser()
parser.add_argument('--test', dest='test', action='store_true', default=False)
args = parser.parse_args()

x_train, y_train, x_test = dataframes()
x_train = x_train.values
x_test = x_test.values
y_train = y_train.values.reshape(-1)
#IMPUTE
x_train = impute(x_train, 'knn')
x_test = impute(x_test, 'knn')
#SCALING
scaler = StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
#FEATURE SELCTION
vt = VarianceThreshold(threshold=1)
vt.fit(x_train)
x_train = vt.transform(x_train)
x_test = vt.transform(x_test)
#x_train = np.concatenate((x_train, x_train**2), axis=1)
print("#Features after variance thresholding:", x_train.shape[1])
#k_best = SelectKBest(f_classif, k=50).fit(x_train, y_train)
#x_train = k_best.transform(x_train)
rfecv = RFECV(
    estimator=SVR(kernel='linear', C=10.0), step=0.2,
    cv=KFold(n_splits=n_splits, shuffle=True, random_state=42),
    scoring='r2', n_jobs=-1)
rfecv.fit(x_train, y_train)
print("Optimal number of features:", rfecv.n_features_)
print(rfecv.grid_scores_)
plt.figure()
plt.xlabel("Number of features")
plt.ylabel("Cross validation r^2 score")
plt.plot(range(1, len(rfecv.grid_scores_) +1), rfecv.grid_scores_)
plt.show()
x_train = rfecv.transform(x_train)
x_test = rfecv.transform(x_test)
#OUTLIER DETECTION
outliers_removed = x_train.shape[0]
x_train, y_train = detect_outliers(x_train, y_train)
outliers_removed -= x_train.shape[0];
print("Outlier removed:", outliers_removed)
scaler = StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)

if args.test:
    x_test = scaler.transform(x_test)
    #x_test = k_best.transform(x_test)
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
