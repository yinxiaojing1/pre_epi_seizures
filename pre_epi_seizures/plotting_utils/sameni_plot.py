import pre_epi_seizures

from pre_epi_seizures.Preprocessing.pre_processing import *

from pre_epi_seizures.classification.labels import *

from pre_epi_seizures.classification.scaling import *

# from pre_epi_seizures.classification.labels import *


import numpy as np
# from sklearn.preprocessing import *

def plot_ECG_seizure(ecg_data, sz_nr, start, end):
    plt.figure()
    # plt.subplot(1,2,1)
    # plt.title('interpolated ECG')
    plt.plot(signal[sz_nr])
    # plt.plot(n[rpeaks[sz_nr]], signal[sz_nr][rpeaks[sz_nr]], 'o', color='g')
    # plt.xlim([start, end])
    # plt.xlabel('time[s]')
    # plt.subplot(1,2,2)
    # plt.title('Detrended and Denoised ECG')
    # plt.plot(signal_t[sz_nr])
    plt.xlim([start, end])
    # plt.xlabel('time[s]')
    plt.show()
    # stop


def unspecific_hist(labels, seizure_list, bins):
    hist = dict()
    for k in labels.keys():
        hist_list = [np.histogram(
                           seizure[labels[k][0][i][0]:
                           labels[k][0][i][1]], 
                           bins)[0]
                for i, seizure in enumerate(seizure_list)]

        hist_array_per_seizure = np.asarray(hist_list)
        hist_total = np.sum(hist_array_per_seizure,
                            axis=0)
        hist[k] = (hist_total, labels[k][1])

    return hist


def histogram(hist, bins):
    w = bins[0]/10
    for i, k in enumerate(hist.keys()):
        plt.subplot(len(hist.keys()), 1, i+1)
        plt.bar(bins , hist[k][0], width=w, color = hist[k][1])
        plt.legend([k])


def hist_per_seizure(labels_seizure, data_seizure_list, nr_seizure):

    data = [data_seizure_list[nr_seizure]]

    labels = {k: ([labels_seizure[k][0][nr_seizure]], labels_seizure[k][1]) for k in labels_seizure.keys()}

    bins = np.linspace(0, 1, 10)

    hist = unspecific_hist(labels, data, bins)
    # stop

    histogram(hist, bins[1:])


def plot_per_seizure(labels_seizure,
                     data_seizure_list, nr_seizure):
    plt.figure()
    data = data_seizure_list[nr_seizure]
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


def feature_allocation(data, feature_list):
    feature_subset_per_seizure = [seizure[:, feature_list] for seizure in data]
    return feature_subset_per_seizure


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
interpolated = load_feature(path_to_load, 'interpolation', sampling_rate=500, files='existent', feature_group_to_process=eks_dataset_name)[0]
rpeaks = load_feature(path_to_load, 'rpeak_detection', files='existent', feature_group_to_process=interpolated_dataset_name)[0]
# hrv = load_feature(path_to_load, 'hrv_computation', files='existent', feature_group_to_process=interpolated_dataset_name, rpeak_group_to_process=interpolated_dataset_name + '/' + 'rpeak_detection')[0]
# beat = load_feature(path_to_load, 'beat_phase_segmentation', files='existent', feature_group_to_process=interpolated_dataset_name, rpeak_group_to_process=interpolated_dataset_name + '/' + 'rpeak_detection')[0]
sameni = load_feature(path_to_load, 'sameni_evolution', files='existent', feature_group_to_process=interpolated_dataset_name + '/' + 'beat_phase_segmentation')[0]
# rqa = load_feature(path_to_load, 'rqa_computation', files='existent', feature_group_to_process=interpolated_dataset_name + '/' + 'beat_phase_segmentation')[0]
# pca = load_feature(path_to_load, 'pca_beat_amp_computation', files='existent', feature_group_to_process=interpolated_dataset_name + '/' + 'QRS_fixed_segmentation')[0]
# pca_corrected = load_feature(path_to_load, 'pca_beat_amp_computation', files='existent', feature_group_to_process=interpolated_dataset_name + '/' + 'beat_phase_segmentation')[0]

# print data[0]

#---------------------------------------
# print hrv[0]
# print beat[0]
# print pca[0]

# stop

# Data allocation 3D ------------------------
data = sameni


# Window allocation 2D-----------------------
windows_list = [rpeak[1:] for rpeak in rpeaks]


# Label allocation -----------------------------
labels = create_labels(1000,
                        windows_list=windows_list,
                        inter_ictal = ((0, 5), 'g'),
                        pre_ictal = ((5,10), 'orange'),
                        unspecified=((10,35), 'blue'),
                        post_ictal = ((35,40), 'y'),
                        ictal = ((30,35), 'r'))


# Features to plot
features = [0]

start = time_before_seizure * 60
end = start + 5

plt.plot()
features_to_analyse = feature_allocation(data, features)



features_to_analyse = scale(features_to_analyse)


# print features_to_analyse
# stop


sz_nr = 11


hist_per_seizure(labels, features_to_analyse, sz_nr)

# plt.show()

plot_per_seizure(labels, features_to_analyse, sz_nr)

plt.show()

