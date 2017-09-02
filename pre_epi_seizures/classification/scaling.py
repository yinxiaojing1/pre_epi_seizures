from sklearn.preprocessing import *
import matplotlib.pyplot as plt
import numpy as np

def scale(feature_list):
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

    norm_feature_list = [(feature - feature.min(axis=0)) / (feature.max(axis=0) - feature.min(axis=0)) for feature in feature_list]
    # norm_feature_list = scaler.transform(feature_list)
    # # mean = np.mean(feature_list, axis = 1)11
    # # print len(mean)
    # # feature_list = map
    # # print feature_list
    # # stop

    return np.asmatrix(norm_feature_list)