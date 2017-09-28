from sklearn.preprocessing import *
import matplotlib.pyplot as plt
import numpy as np


def _scale_feature_time(feature_time_array):
    print feature_time_array
    scaled = (feature_time_array - feature_time_array.min(axis=0)) / (feature_time_array.max(axis=0) - feature_time_array.min(axis=0))
    return scaled

def scale_features(seizure_features):
    # seizure_features: ARRAY OF DIFFERENT feature_time_arrays

    # Scaling with respect to time
    # seizure_features = np.asarray(seizure_features)
    norm = np.apply_along_axis(_scale_feature_time, 1, seizure_features)
    print norm
    return norm


def scale(seizure_list):
    # scaler = StandardScaler().fit(feature_list)
    # norm_feature_list = scaler.transform(feature_list)

    # for feature in feature_list:
    #     [norm = (feature - feature.min(axis=0)) / (feature.max(axis=0) - feature.min(axis=0))
    #     # print X_std
    #     # stop
    #     # norm = X_std * (max - min) + min
    #     #     print np.mean(feature)
    #     #     print feature
    #     #     print np.std(feature)
    #     #     norm = (feature - np.mean(feature))/np.std(feature)
    #     plt.plot(norm)
    #     plt.show()

    # scaler = MinMaxScaler(copy=True, feature_range=(0, 1)).fit(feature_list)
    # stop
    print seizure_list
    scaled = [scale_features(seizure_features) for seizure_features in seizure_list]

    # stop
    # st = scaler.transform(feature_list)
    # # mean = np.mean(feature_list, axis = 1)11
    # # print len(mean)
    # # feature_list = map
    # # print feature_list
    # # stop

    return scaled