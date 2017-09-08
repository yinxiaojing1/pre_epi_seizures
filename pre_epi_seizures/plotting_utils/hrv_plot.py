import pre_epi_seizures

from pre_epi_seizures.Preprocessing.pre_processing import *

from pre_epi_seizures.classification.labels import *

# from pre_epi_seizures.classification.labels import *


import numpy as np

def unspecific_hist(labels, sz, bins):

    hist = dict()
    for k in labels.keys():
        hist_per_seizure = np.asarray([np.histogram(sz[i][labels[k][0][i][0]:labels[k][0][i][1]], bins)[0]
                                  for i in xrange(len(sz))])
        hist_total = np.sum(hist_per_seizure, axis=0)
        hist[k] = (hist_total, labels[k][1])
    return hist


def plot(hist, bins):
    w = bins[0]/100
    for i, k in enumerate(hist.keys()):
        plt.subplot(len(hist.keys()), 1, i+1)
        plt.bar(bins , hist[k][0], width=w, color = hist[k][1])
        plt.legend([k])
    # plt.show()


def hist_per_seizure(data, nr_seizure, windows):

    windows = windows[nr_seizure]
    data = data[nr_seizure]
    labels = create_labels(1000,
                            windows_list=[windows],
                            inter_ictal = ((0, 5), 'g'),
                            pre_ictal = ((25,30), 'orange'),
                            post_ictal = ((35,40), 'y'),
                            ictal = ((30,35), 'r'))

    print labels

    bins = np.linspace(60, 120)

    hist = unspecific_hist(labels, [data], bins)

    plot(hist, bins[1:])

    # print hist

    # print len(hist['ictal'])
    # print len(bins)
    # print hist['ictal'][1]

    # stop

def plot_per_seizure(labels_seizure,
                     data_seizure_list, nr_seizure):
    plt.figure()
    data = np.asarray([data_seizure_list[nr_seizure]]).T

    labels = {k: ([labels_seizure[k][0][nr_seizure]], labels_seizure[k][1]) for k in labels_seizure.keys()}
    print labels
    plot_seizure_duration_per_feature(labels, data)


def plot_seizure_duration_per_feature(labels,
                                      data_single_feature):
    for k in labels.keys():
        print labels[k]
        up = labels[k][0][0][1]
        down = labels[k][0][0][0]
        print up
        print down
        print up - down
        n = np.linspace(down, up, up-down, dtype='int')
        print len(n)
        print len(data_single_feature)
        # print data_single_feature[
        # print data_single_feature[down:up,:]
        plt.plot(n, data_single_feature[down:up,:], color = labels[k][1])



#signal
sampling_rate = 1000
time_before_seizure = 30
time_after_seizure = 10
# path_to_load = '~/Desktop/phisionet_seizures_new.h5'
# sampling_rate = 1000
path_to_load = '~/Desktop/seizure_datasets_new.h5'
# name_list = [str(time_before_seizure*60) + '_' + str(time_after_seizure*60)]
# group_list_raw = ['raw']
# group_list_baseline_removal = ['medianFIR']
# group_list_noise_removal = ['FIR_lowpass_40hz']
# group_list_esksmooth = ['esksmooth']
dataset_name = str(
time_before_seizure*60) + '_' + str(time_after_seizure*60)
raw_name = 'raw'
baseline_removal_name = 'baseline_removal'
raw_dataset_name = dataset_name + '/' + raw_name
baseline_removal_dataset_name = raw_dataset_name + '/' + baseline_removal_name
decimated_dataset_name = baseline_removal_dataset_name + '/' + 'decimation'
eks_dataset_name = decimated_dataset_name + '/' + 'eks_smoothing'
interpolated_dataset_name = eks_dataset_name + '/' + 'interpolation'
# group_name_list = list_group_signals(path_to_load, group_list[0])['signals']
# compress(path_to_load, group_name_list)

# raw = load_feature(path_to_load, raw_name, files='existent', feature_group_to_process=dataset_name)

# baseline_removal = load_feature(path_to_load, baseline_removal_name, files='existent', feature_group_to_process=raw_dataset_name)

# decimated = load_feature(path_to_load, 'decimation', files='existent', feature_group_to_process=baseline_removal_dataset_name)
# rpeaks = load_feature(path_to_load, 'rpeak_detection', files='existent', feature_group_to_process=baseline_removal_dataset_name)

# hrv = load_feature(path_to_load, 'hrv_computation', files='existent', feature_group_to_process=baseline_removal_dataset_name + '/' + 'rpeak_detection')
# eks = load_feature(path_to_load, 'eks_smoothing', files='existent', feature_group_to_process=baseline_removal_dataset_name + '/' + 'decimation', rpeak_group_to_process=baseline_removal_dataset_name + '/' + 'decimation' + '/' + 'rpeak_detection')[0]
# # stop
# # time_array_to_interpolate = np.linspace(0, 40*60 - 1.0/500, 40*60*500)
# # print time_array_to_interpolate
# interpolated = load_feature(path_to_load, 'interpolation', sampling_rate=500, files='existent', feature_group_to_process=eks_dataset_name)[0]
rpeaks = load_feature(path_to_load, 'rpeak_detection', files='existent', feature_group_to_process=interpolated_dataset_name)[0]
hrv = load_feature(path_to_load, 'hrv_computation', files='existent', feature_group_to_process=interpolated_dataset_name, rpeak_group_to_process=interpolated_dataset_name + '/' + 'rpeak_detection')[0]
# beat = load_feature(path_to_load, 'beat_phase_segmentation', files='existent', feature_group_to_process=interpolated_dataset_name, rpeak_group_to_process=interpolated_dataset_name + '/' + 'rpeak_detection')[0]
# sameni = load_feature(path_to_load, 'sameni_evolution', files='existent', feature_group_to_process=interpolated_dataset_name + '/' + 'beat_phase_segmentation')[0]
# rqa = load_feature(path_to_load, 'rqa_computation', files='existent', feature_group_to_process=interpolated_dataset_name + '/' + 'beat_phase_segmentation')[0]








# Select features --------------------------------------------------

print len(hrv[0])
windows = [np.linspace(rpeak[1], rpeak[-1], len(hr)) for rpeak, hr in zip(rpeaks, hrv)]

labels = create_labels(1000,
                        windows_list=windows,
                        inter_ictal = ((0, 5), 'g'),
                        pre_ictal = ((5,10), 'orange'),
                        unspecified=((10,35), 'blue'),
                        post_ictal = ((35,40), 'y'),
                        ictal = ((30,35), 'r'))

nr_sz = 5

data = hrv
hist_per_seizure(data, nr_sz, windows)
plot_per_seizure(labels, data, nr_sz)

plt.show()
# stop
print hrv

print labels

bins = np.linspace(60, 120)

hist = unspecific_hist(labels, data, bins)

# print hist

# print len(hist['ictal'])
# print len(bins)
# print hist['ictal'][1]

# stop

plot(hist, bins[1:])