from sklearn.preprocessing import RobustScaler
import numpy as np

def scale(feature_list):
    sc = RobustScaler(quantile_range=(25, 75)).fit_transform(np.asarray(feature_list))
    return sc