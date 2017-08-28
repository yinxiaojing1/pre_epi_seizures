from pre_epi_seizures.storage_utils.storage_utils_hdf5 import \
    load_signal, save_signal, delete_signal, list_group_signals

from pre_epi_seizures.storage_utils.data_handlers import *

from pre_epi_seizures.logging_utils.formatter_logging import logger as _logger

from filtering import baseline_removal, noise_removal,\
    create_filtered_dataset, eks_smoothing

from segmentation import rpeak_detection, create_heart_beat_dataset,\
    find_rpeaks, detect_rpeaks, hrv_computation, QRS_fixed_segmentation

from resampling import resample_rpeaks, interpolate_signal,\
    interpolation, decimation

from visual_inspection import visual_inspection

from morphology import *
# from Filtering.gaussian_fit import get_phase, mean_extraction,\
#     beat_fitter, ecg_model

from Filtering.filter_signal import filter_signal

from Filtering.eksmoothing import *

from biosppy.signals import ecg
from biosppy.clustering import centroid_templates, kmeans

from memory_profiler import profile

import time
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import logging
import sys
import copy
import memory_profiler
import functools

def compress(path_to_load, group_name_list):
    print 'before loading' 
    print group_name_list

    # Memory loop (one signal at a time)
    for i, group_name in enumerate(group_name_list):
        signal_structure = load_signal(path_to_load, [group_name])
        one_signal_structure = get_one_signal_structure(signal_structure, group_name)
        record = get_multiple_records(one_signal_structure)


def extract_feature(feature, arg_list):
    return locals[feature](arg_list)


def get_names(group_name_list):
    print group_name_list
    return [group_name[1] for group_name in group_name_list]


def load_feature(path_to_load, feature_to_load, files='just_new_data', sampling_rate=1000, **feature_groups_required):

    feature_group_to_process = feature_groups_required['feature_group_to_process']

    feature_group_extracted = [feature_group_to_process + '/' + feature_to_load]

    feature_group_aux = {k:feature_groups_required[k] for k in feature_groups_required.keys() if 'group' in k and 'process' not in k}

    auxiliary_inputs = {k:feature_groups_required[k] for k in feature_groups_required.keys() if 'group' not in k and 'process' not in k}

    if files=='all_new':
        names_to_save = get_names(list_group_signals(path_to_load, feature_group_to_process)['signals'])
        print feature_groups_required
        feature_group_to_process
        for k in feature_groups_required.keys():
            feature_groups_required[k] = list_group_signals(path_to_load, feature_groups_required[k])['signals']
        names_to_save = get_names(list_group_signals(path_to_load, feature_group_to_process)['signals'])

    if files=='just_new_data':
        names_already_processed = get_names(list_group_signals(path_to_load, feature_group_extracted[0])['signals'])
        names_to_save = get_names(list_group_signals(path_to_load, feature_group_to_process)['signals'])
        names_to_save = [name for name in names_to_save if name not in names_already_processed]
        feature_group_to_process = [(feature_group_to_process, name) for name in names_to_save]
        stop # Ver melhor
        print feature_group_to_process
        stop
        print feature_groups_required
        for k in feature_groups_required.keys():
            feature_groups_required[k] = list_group_signals(path_to_load, feature_groups_required[k])['signals']
        #*****************IMPORTANT CHANGE***************************
        names_to_save = get_names(list_group_signals(path_to_load, feature_group_to_process)['signals'])

    if files=='existent':
        feature_group_name_extracted = list_group_signals(path_to_load, feature_group_extracted[0])['signals']
        try:
            signal_structure = load_signal(path_to_load, feature_group_name_extracted)
            extracted_features = [get_multiple_records(get_one_signal_structure(signal_structure, group_name)) for group_name in feature_group_name_extracted]
            return extracted_features
        except Exception as e:
            _logger.debug(e)

    for k in feature_groups_required.keys():
        signal_structure = load_signal(path_to_load, feature_groups_required[k])
        feature_groups_required[k] = [get_multiple_records(get_one_signal_structure(signal_structure, group_name)) for group_name in feature_groups_required[k]]

    print feature_groups_required

    for i, name in enumerate(names_to_save):
        dict_to_process = {}
        for k in feature_groups_required.keys():
            dict_to_process[k] = [feature_groups_required[k][i]]
        print dict_to_process
        extracted_features, mdata_features = globals()[feature_to_load](dict_to_process, sampling_rate)
        # print extracted_features
        # print mdata_features
        delete_signal(path_to_load, [name], feature_group_extracted)
        save_signal(path_to_load, extracted_features, mdata_features, [name], feature_group_extracted)

    # # Memory Intensive
    # extracted_features, mdata_features = globals()[feature_to_load](feature_groups_required, sampling_rate)

    # delete_signal(path_to_load, names_to_save, feature_group_extracted)
    # save_signal(path_to_load, extracted_features, mdata_features, names_to_save, feature_group_extracted)




# @profile
def main():

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
    eks = load_feature(path_to_load, 'eks_smoothing', files='existent', feature_group_to_process=baseline_removal_dataset_name + '/' + 'decimation', rpeak_group_to_process=baseline_removal_dataset_name + '/' + 'decimation' + '/' + 'rpeak_detection')
    # stop
    # time_array_to_interpolate = np.linspace(0, 40*60 - 1.0/500, 40*60*500)
    # print time_array_to_interpolate
    interpolated = load_feature(path_to_load, 'interpolation', sampling_rate=500, files='existent', feature_group_to_process=eks_dataset_name)
    rpeaks = load_feature(path_to_load, 'rpeak_detection', files='existent', feature_group_to_process=interpolated_dataset_name)

    QRS = load_feature(path_to_load, 'QRS_fixed_segmentation', files='existent', feature_group_to_process=interpolated_dataset_name, rpeak_group_to_process=interpolated_dataset_name + '/' + 'rpeak_detection')

    sz_nr = 1
    data = QRS[sz_nr]
    rpeaks = rpeaks[sz_nr]
    # print QRS[0]
    # plt.plot(QRS[0][4])
    # plt.show()
    # stop

    # Time labels
    ti_inter_ictal = 0
    i_sample_inter_ictal = sampling_rate * ti_inter_ictal * 60
    tf_inter_ictal = 5
    f_sample_inter_ictal = sampling_rate * tf_inter_ictal * 60
    inter_ictal = np.arange(i_sample_inter_ictal, f_sample_inter_ictal)
    color_inter_ictal = 'green'

    ti_pre_ictal = tf_inter_ictal
    i_sample_pre_ictal = sampling_rate * ti_pre_ictal * 60
    tf_pre_ictal = time_before_seizure
    f_sample_pre_ictal = sampling_rate * tf_pre_ictal * 60
    pre_ictal = np.arange(i_sample_pre_ictal, f_sample_pre_ictal)
    color_pre_ictal = 'orange'

    ti_ictal = time_before_seizure
    i_sample_ictal = sampling_rate * ti_ictal * 60
    tf_ictal = time_before_seizure + 5
    f_sample_ictal = sampling_rate * tf_ictal * 60
    ictal = np.arange(i_sample_ictal, f_sample_ictal)
    color_ictal = 'red'

    ti_post_ictal = tf_ictal
    i_sample_post_ictal = sampling_rate * ti_post_ictal * 60
    tf_post_ictal = time_before_seizure + time_after_seizure
    f_sample_post_ictal = sampling_rate * tf_post_ictal * 60
    post_ictal = np.arange(f_sample_post_ictal, f_sample_post_ictal)
    color_post_ictal = 'blue'


    beat_inter_ictal = np.where(np.logical_and(rpeaks>=i_sample_inter_ictal, rpeaks<f_sample_inter_ictal))[0]
    beat_pre_ictal = np.where(np.logical_and(rpeaks>=i_sample_pre_ictal, rpeaks<f_sample_pre_ictal))[0]
    beat_ictal = np.where(np.logical_and(rpeaks>=i_sample_ictal, rpeaks<f_sample_ictal))[0]
    beat_post_ictal = np.where(np.logical_and(rpeaks>=i_sample_post_ictal, rpeaks<f_sample_post_ictal))[0]

    print beat_inter_ictal
    print beat_pre_ictal
    print beat_ictal
    print beat_post_ictal

    init = 300
    cluster_inter_ictal = beat_inter_ictal[init]
    cluster_pre_ictal = beat_pre_ictal[init]
    cluster_ictal = beat_ictal[init]
    cluster_post_ictal = beat_post_ictal[init]

    centroids = data[np.asarray([cluster_inter_ictal, cluster_pre_ictal, cluster_ictal, cluster_post_ictal])]

    beat_inter_ictal = np.where(np.logical_and(rpeaks>=i_sample_inter_ictal, rpeaks<f_sample_inter_ictal))[0][-1]
    beat_pre_ictal = np.where(np.logical_and(rpeaks>=i_sample_pre_ictal, rpeaks<f_sample_pre_ictal))[0][-1]
    beat_ictal = np.where(np.logical_and(rpeaks>=i_sample_ictal, rpeaks<f_sample_ictal))[0][-1]
    beat_post_ictal = np.where(np.logical_and(rpeaks>=i_sample_post_ictal, rpeaks<f_sample_post_ictal))[0][-1]

    # clusters = {'first':[cluster_inter_ictal, cluster_pre_ictal, cluster_ictal, cluster_post_ictal]}
    # template = centroid_templates(data, clusters, 4)[0]
    clusters = kmeans(data=data, k=4, init=centroids, max_iter=3000, n_init=10, tol=0.0001)
    centroids = clusters['clusters'].keys()
    cluster_0 = clusters['clusters'][centroids[1]]

    print len(cluster_0)
    # print np.shape(template)
    # print np.shape(data)

    hist0, bins = np.histogram(clusters['clusters'][centroids[0]], bins=[0, beat_inter_ictal, beat_pre_ictal, beat_ictal, beat_post_ictal])
    hist1, bins = np.histogram(clusters['clusters'][centroids[1]], bins=[0, beat_inter_ictal, beat_pre_ictal, beat_ictal, beat_post_ictal])
    hist2, bins = np.histogram(clusters['clusters'][centroids[2]], bins=[0, beat_inter_ictal, beat_pre_ictal, beat_ictal, beat_post_ictal])
    hist3, bins = np.histogram(clusters['clusters'][centroids[3]], bins=[0, beat_inter_ictal, beat_pre_ictal, beat_ictal, beat_post_ictal])

    print hist0
    print hist1
    print hist2
    print hist3

    plt.bar(range(len(hist0)),hist0,width=0.5, color = 'g')
    plt.bar(range(len(hist1)),hist1,width=0.5, color = 'orange')
    plt.bar(range(len(hist2)),hist2,width=0.5, color = 'r')
    plt.bar(range(len(hist3)),hist3,width=0.5, color = 'b')
    # plt.subplot(4, 1, 1)
    # plt.plot(template[0], color=color_inter_ictal)
    # plt.subplot(4, 1, 2)
    # plt.plot(template[1], color=color_pre_ictal)
    # plt.subplot(4, 1, 3)
    # plt.plot(template[2], color=color_ictal)
    # plt.subplot(4, 1, 4)
    # plt.plot(template[3], color=color_post_ictal)
    # plt.show()

    # plt.subplot(4, 1, 1)
    # plt.plot(data[clusters[0]], color=color_inter_ictal)
    # plt.subplot(4, 1, 2)
    # plt.plot(template[1], color=color_pre_ictal)
    # plt.subplot(4, 1, 3)
    # plt.plot(template[2], color=color_ictal)
    # plt.subplot(4, 1, 4)
    # plt.plot(template[3], color=color_post_ictal)
    # plt.hist(cluster_0, bins=[0, beat_inter_ictal, beat_pre_ictal, beat_ictal, beat_post_ictal], histtype='bar', ec='black')
    # plt.hist(cluster_0, bins='auto', histtype='bar', ec='black')
    plt.show()

    # # ploting
    # start = 280
    # end = 290
    # sz_nr = 1
    # signal = interpolated
    # signal_t = eks
    # n = np.linspace(0, (len(signal[sz_nr])-1)/1000, len(signal[sz_nr]))
    # n_t = np.linspace(0, (len(signal_t[sz_nr])-1)/1000, len(signal_t[sz_nr]))

    # plt.subplot(1,2,1)
    # plt.title('interpolated ECG')
    # plt.plot(n, signal[sz_nr])
    # plt.plot(n[rpeaks[sz_nr]], signal[sz_nr][rpeaks[sz_nr]], 'o', color='g')
    # plt.xlim([start, end])
    # plt.xlabel('time[s]')
    # plt.subplot(1,2,2)
    # plt.title('Detrended and Denoised ECG')
    # plt.plot(n_t, signal_t[sz_nr])
    # plt.xlim([start, end])
    # plt.xlabel('time[s]')
    # plt.show()

    # #phase 
    # signal = signal[sz_nr]
    # rpeaks = rpeaks[sz_nr]
    # phase = get_phase(signal, rpeaks)

    # idx_up = np.where(phase==np.pi)[0]
    # idx_down = np.where(phase==(-1*np.pi))[0]
    # # mean_extraction(signal, phase, bins=1000)
    # print phase
    # print idx_up
    # print idx_down
    # signal = signal[rpeaks[0]:rpeaks[-1]]
    # phase = phase[rpeaks[0]:rpeaks[-1]]
    # print len(signal)
    # print len(phase)
    # plt.plot(signal*0.05)
    # plt.plot(phase)
    # plt.xlim([start*1000, end*1000])
    # plt.show()

    # fig_phase = plt.figure()
    # phase = get_phase(interpolated[sz_nr], rpeaks[sz_nr])
    # print phase[0]


    return




if __name__ == '__main__':
    main()

# g1