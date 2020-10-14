def detect_outliers(df, *args, **kwargs):
    import numpy as np
    from sklearn.ensemble import IsolationForest
    
    rng = np.random.RandomState(42)
    sensitivity="auto"
    if "sensitivity" in kwargs:
        sensitivity = kwargs["sensitivity"]
    clf = IsolationForest(random_state=rng, contamination=sensitivity)
    clf.fit(df)
    df_outliers = clf.predict(df)
    df = df[df_outliers==1]
    print("Inliers: ", (df_outliers == 1).sum())
    print("Outliers: ", (df_outliers == -1).sum())
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
#    x_train = detect_outliers(x_train)
#    x_train, y_train = detect_outliers(x_train, y_train)
#    x_train = detect_outliers(x_train, sensitivity=0.05)
    x_train, y_train = detect_outliers(x_train, y_train, sensitivity=0.05)


    
