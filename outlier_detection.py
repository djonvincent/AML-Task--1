import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


def isolation_forest(df, *args, **kwargs):
    forest = True
    rng = np.random.RandomState(42)
    sensitivity="auto"
    if "sensitivity" in kwargs:
        sensitivity = kwargs["sensitivity"]
    clf = IsolationForest(random_state=rng, contamination=sensitivity)
    clf.fit(df)
    df_outliers = clf.predict(df)
    print("Inliers: ", (df_outliers == 1).sum())
    print("Outliers: ", (df_outliers == -1).sum())
    df = df[df_outliers==1]
    if len(args) == 1:
        y = args[0]
        y = y[df_outliers==1]
        return df, y
    else:
        return df

def local_outlier(df, *args, **kwargs):
    rng = np.random.RandomState(42)
    neighbors=30
    if "neighbors" in kwargs:
        neighbors = kwargs["neighbors"]
    clf = LocalOutlierFactor(n_neighbors=neighbors)
    df_outliers = clf.fit_predict(df)
    print("Inliers: ", (df_outliers == 1).sum())
    print("Outliers: ", (df_outliers == -1).sum())
    df = df[df_outliers==1]
    if len(args) == 1:
        y = args[0]
        y = y[df_outliers==1]
        return df, y
    else:
        return df
    
    
if __name__ == "__main__":
    from load_data import dataframes
    from imputation import impute
    from visualise import visualise

    x_train = impute(dataframes()[0], 'knn')
    y_train = impute(dataframes()[1], 'knn')
#    x_train, y_train = detect_outliers(x_train, y_train, sensitivity='auto')


    
