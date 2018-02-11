import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from save_for_class import *

def univariate_statistics(full_path,
                           X, y, groups,
                           pipe,
                           param_grid, scoring,
                           compute_all_new, cv_out, cv_in):
    # Get features from data structure
    features = list(X.columns)

    # Univariate statistical analysis
    univ_stats = pd.concat([_univariate_statistics(X,
                                         y, feature)
                            for feature in features],
                            axis=1)

    # Save statistic
    univ_stats.to_hdf(full_path + '**statistics.h5',
                      key = 'test',
                      format ='t')

    return univ_stats

def _univariate_statistics(X, y, feature):
        # Compute statistical analysis
        X_feature = pd.DataFrame(X[feature])
        Y_feature = y
        X_feature['Y'] = Y_feature
        stats = __univariate_statistics(X_feature)
        return pd.DataFrame(stats)

def __univariate_statistics(univariate_X):
    return univariate_X.groupby('Y').describe()
