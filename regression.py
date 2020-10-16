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
import matplotlib.pyplot as plt
from outlier_detection import isolation_forest, local_outlier

n_splits=3
model = SVR(kernel='linear', C=10.0)

parser = ArgumentParser()
parser.add_argument('--test', dest='test', action='store_true', default=False)
args = parser.parse_args()

x_train, y_train, x_test = dataframes()
x_train = x_train.values
x_test = x_test.values
y_train = y_train.values.reshape(-1)

#IMPUTE
x_train = impute(x_train, 'knn')
if args.test:
    x_test = impute(x_test, 'knn')

#FEATURE SELCTION
#x_train = np.concatenate((x_train, x_train**2), axis=1)
#fs = SelectKBest(f_classif, k=100).fit(x_train, y_train)
fs = RFECV(
    estimator=model, step=0.2, scoring='r2', n_jobs=-1,
    cv=KFold(n_splits=n_splits, shuffle=True, random_state=42))
fs.fit(StandardScaler().fit_transform(x_train), y_train)
print("Optimal number of features:", fs.n_features_)
print(fs.grid_scores_)
#plt.figure()
#plt.xlabel("Number of features")
#plt.ylabel("Cross validation r^2 score")
#plt.plot(range(1, len(rfecv.grid_scores_) +1), rfecv.grid_scores_)
#plt.show()
x_train = fs.transform(x_train)
if args.test:
    x_test = fs.transform(x_test)

#OUTLIER DETECTION
outliers_removed = x_train.shape[0]
#x_train, y_train = isolation_forest(x_train, y_train)
x_train, y_train = local_outlier(x_train, y_train, neighbors=30)
outliers_removed -= x_train.shape[0];
print("Outlier removed:", outliers_removed)

#SCALING
scaler = StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
if args.test:
    x_test = scaler.transform(x_test)

if args.test:
    model.fit(x_train, y_train)
    y_test = model.predict(x_test)
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
        model.fit(x_train[train_idx], y_train[train_idx])
        y_pred = model.predict(x_train[test_idx])
        total_score += r2_score(y_train[test_idx], y_pred)

    print('average score:', total_score/n_splits)
