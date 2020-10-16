from sklearn.feature_selection import SelectKBest

def kBest(x, y, x_test, function, k_features):
	k_best = SelectKBest(function, k=k_features).fit(x, y)
	x = k_best.transform(x)
	x_test = k_best.transform(x_test)
	return x, x_test
	
