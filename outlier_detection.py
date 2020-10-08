import numpy as np
from sklearn.ensemble import IsolationForest
rng = np.random.RandomState(42)

def detect_outliers(data):
    clf = IsolationForest(random_state=rng)
    clf.fit(data)
    return clf.predict(data)

if __name__ == "__main__":
    from load_data import dataframes, impute
    from visualise import visualise

    x_train = impute(dataframes()[0])
    y_pred_train = detect_outliers(x_train)
    print("Inliers: ", (y_pred_train == 1).sum())
    print("Outliers: ", (y_pred_train == -1).sum())
    visualise(x_train, y_pred_train)
