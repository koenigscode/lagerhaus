from .preprocessing import fill_na, std, one_hot_encode, skew, remove_correlated_features, remove_outliers

linear_regression = [fill_na(), remove_outliers(), skew(), std(), remove_correlated_features(), one_hot_encode(), ]
logistic_regression = linear_regression
lasso = [fill_na(), skew(), std(), one_hot_encode()]
ridge = [fill_na(), remove_outliers(), skew(), std(), one_hot_encode(), ]
decision_tree = []
svm = [fill_na(), skew(), std(), one_hot_encode(), ]
naive_bayes = [remove_outliers(), std(), remove_correlated_features(), ]
knn = [fill_na(), std(), one_hot_encode(), ]
nn = [fill_na(), remove_outliers(), std(), remove_correlated_features(), one_hot_encode(), ]

