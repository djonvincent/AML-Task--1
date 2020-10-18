import numpy as np
from argparse import ArgumentParser
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_regression, f_regression
from sklearn.metrics import mean_absolute_error, r2_score
from load_data import dataframes
from imputation import impute
from outlier_detection import isolation_forest, local_outlier
from feature_selection import kBest

n_splits=3

parser = ArgumentParser()
parser.add_argument('--test', dest='test', action='store_true', default=False)
parser.add_argument('--finetune', dest='finetune', action='store_true', default=False)
parser.add_argument('--k', type=int, default=180)
parser.add_argument('--knn_out', type=int, default=25)
parser.add_argument('--knn_imp', type=int, default=15)
parser.add_argument('--C', type=int, default=20)
args = parser.parse_args()

#DATAFRAME
x_train, y_train, x_test = dataframes()
x_train = x_train.values
x_test = x_test.values
y_train = y_train.values.reshape(-1)

def preprocess(x_raw, y_raw, x_test_raw, k_features, lo_neighbors,
        imp_neighbors):
    #IMPUTE
    x = impute(x_raw, 'knn', neighbors=imp_neighbors)
    x_test = impute(x_test_raw, 'knn', neighbors=imp_neighbors)

    #OUTLIER DETECTION
    #x_train, y_train = isolation_forest(x_train, y_train)
    x, y = local_outlier(x, y_raw, neighbors=lo_neighbors)

    #SCALING
    scaler = StandardScaler().fit(x)
    x = scaler.transform(x)
    x_test = scaler.transform(x_test)

    #FEATURE SELECTION
    x, x_test = kBest(x, y, x_test, f_regression, k_features)
    #x, x_test = kBest(x, y, x_test, mutual_info_regression, k_features)

    return x, y, x_test

def CV_score(x, y, model, k_features, lo_neighbors, imp_neighbors, n_splits=3):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    total_score = 0
    for train_idx, test_idx in kf.split(x):
        x_train_kf, y_train_kf, x_test_kf = preprocess(
            x[train_idx], y[train_idx], x[test_idx], k_features, lo_neighbors,
            imp_neighbors
        )
        model.fit(x_train_kf, y_train_kf)
        y_pred = model.predict(x_test_kf)
        total_score += r2_score(y[test_idx], y_pred)
    return total_score/n_splits;

#CROSS-VALIDATION
if args.test:
    x_train, y_train, x_test = preprocess(
        x_train, y_train, x_test, args.k, args.knn_out, args.knn_imp
    )
    reg = SVR(kernel='rbf', C=args.C)
    reg.fit(x_train, y_train)
    y_test = reg.predict(x_test)
    y_test = np.concatenate((
        np.arange(y_test.size).reshape(-1, 1),
        y_test.reshape(-1, 1)), axis=1)
    np.savetxt(fname='y_test.csv', header='id,y', delimiter=',', X=y_test,
            fmt=['%d', '%.5f'], comments='')
elif args.finetune:
    scores = []
    for C in [20.0, 25.0, 30.0]:
        for k in [100, 140, 180, 220]:
            for knn_out in [20, 25, 30]:
                for knn_imp in [3, 5, 10, 15]:
                    print(f'C={C}, k={k}, knn_out={knn_out}, knn_imp={knn_imp}')
                    model = SVR(kernel='rbf', C=C)
                    score = CV_score(x_train, y_train, model, k, knn_out,
                            knn_imp)
                    print('Score:', score)
                    scores.append([C, k, knn_out, knn_imp, score])
    np.savetxt(fname='finetuning.csv', header='C,k,knn_out,knn_imp,r2', delimiter=',',
            X=scores, fmt=['%.1f', '%d', '%d', '%d', '%.5f'], comments='')
else:
    model = SVR(kernel='rbf', C=args.C)
    avg_score = CV_score(x_train, y_train, model, args.k, args.knn_out,
            args.knn_imp)
    print('average score:', avg_score)
