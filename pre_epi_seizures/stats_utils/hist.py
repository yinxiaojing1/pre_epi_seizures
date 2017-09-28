import numpy as np
import matplotlib.pyplot as plt

def histogram_feature_array_list(feature_array_list, labels_list, bins):
    hist_per_seizure = [histogram_feature_array(feature_array, labels_array, bins)
                        for feature_array, labels_array in zip(feature_array_list, labels_list)]
    hist_per_seizure = np.asarray(hist_per_seizure)
    hist_total = np.sum(hist_per_seizure, axis=0)
    return hist_total


def histogram_feature_array(feature_array, labels_array, bins):
    # print np.shape(feature_array)
    hist_array = np.apply_along_axis(func1d=_histogram_single_feature,
                                      axis=1, arr=feature_array,
                                      labels=labels_array, bins=bins)
    return hist_array


def _histogram_single_feature(single_feature, labels, bins):
    hist = np.histogram(single_feature[labels[0]:labels[1]], bins)[0]
    return hist


def unspecific_hist(labels, feature_array_list, bins):

    # hist = dict()
    hist = [histogram_feature_array_list(feature_array_list,
                                  labels[k][0], bins)
            for k in labels.keys()]

    return hist


def histogram(signal_arguments, labels, sampling_rate):
    feature_array_list = signal_arguments['feature_group_to_process']
    labels = labels['feature_group_to_process']
    bins = np.linspace(0, 1, 1000)
    hist = unspecific_hist(labels, feature_array_list, bins)
    mdata = [k for k in labels.keys()]
    return hist, mdata



# bins = np.linspace(0, 1, 1000)
#     # hist = scale(hist)
#     # print labels.keys()

#     labels = labels['feature_group_to_process']
#     # stop

#     print feature_groups_required

#     stop
#     for i, k in enumerate(labels.keys()):
#         plt.subplot(len(labels.keys()), 1, i+1)

#         # print hist[i][0]
#         # print bins
#         # print labels[k][1]
#         # stop
#         plt.bar(bins[0:-1] , hist[i][0], width = 0.01 , color = labels[k][1])
#         plt.xlim([0, 1])
#         plt.legend([k])

#     plt.savefig('test.png')
#     print histogram