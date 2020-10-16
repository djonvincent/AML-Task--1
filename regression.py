import numpy as np
from argparse import ArgumentParser
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import mean_absolute_error, r2_score
from load_data import dataframes
from imputation import impute
from outlier_detection import isolation_forest, local_outlier

n_splits=3

parser = ArgumentParser()
parser.add_argument('--test', dest='test', action='store_true', default=False)
parser.add_argument('--finetune', dest='finetune', action='store_true', default=False)
parser.add_argument('--k', type=int, default=100)
parser.add_argument('--knn', type=int, default=30)
args = parser.parse_args()

#DATAFRAME
x_train, y_train, x_test = dataframes()
x_train = x_train.values
x_test = x_test.values
y_train = y_train.values.reshape(-1)

def preprocess(x_raw, y_raw, x_test_raw, k_features, lo_neighbors):
    #IMPUTE
    x = impute(x_raw, 'knn')
    x_test = impute(x_test_raw, 'knn')

    #OUTLIER DETECTION
    #x_train, y_train = isolation_forest(x_train, y_train)
    x, y = local_outlier(x, y_raw, neighbors=lo_neighbors)

    #SCALING
    scaler = StandardScaler().fit(x)
    x = scaler.transform(x)
    x_test = scaler.transform(x_test)

    #FEATURE SELECTION
    k_best = SelectKBest(f_classif, k=k_features).fit(x, y)
    x = k_best.transform(x)
    x_test = k_best.transform(x_test)

    return x, y, x_test

def CV_score(x, y, model, k_features, lo_neighbors, n_splits=3):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    total_score = 0
    for train_idx, test_idx in kf.split(x):
        x_train_kf, y_train_kf, x_test_kf = preprocess(
            x[train_idx], y[train_idx], x[test_idx], k_features, lo_neighbors
        )
        model.fit(x_train_kf, y_train_kf)
        y_pred = model.predict(x_test_kf)
        total_score += r2_score(y[test_idx], y_pred)
    return total_score/n_splits;

#CROSS-VALIDATION
if args.test:
    x_train, y_train, x_test = preprocess(x_train, y_train, x_test)
    reg = SVR(kernel='rbf', C=10.0)
    reg.fit(x_train, y_train)
    y_test = reg.predict(x_test)
    y_test = np.concatenate((
        np.arange(y_test.size).reshape(-1, 1),
        y_test.reshape(-1, 1)), axis=1)
    np.savetxt(fname='y_test.csv', header='id,y', delimiter=',', X=y_test,
            fmt=['%d', '%.5f'], comments='')
elif args.finetune:
    scores = []
    for C in [8.0, 10.0, 15.0]:
        for k in [50, 75, 100, 150]:
            for knn in [10, 20, 30, 40, 50]:
                print(f'C={C}, k={k}, knn={knn}')
                model = SVR(kernel='rbf', C=C)
                score = CV_score(x_train, y_train, model, k, knn)
                print(score)
                scores.append([C, k, knn, score])
    np.savetxt(fname='finetuning.csv', header='C,k,knn', delimiter=',',
            X=scores, fmt=['%.1f', '%d', '%d', '%.5f'], comments='')
else:
    model = SVR(kernel='rbf', C=10.0)
    avg_score = CV_score(x_train, y_train, model, args.k, args.knn)
    print('average score:', avg_score)
