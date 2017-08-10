from pre_epi_seizures.storage_utils.storage_utils_hdf5 import \
    load_signal, save_signal

from pre_epi_seizures.storage_utils.data_handlers import *

from pre_epi_seizures.logging_utils.formatter_logging import logger as _logger

from heart_rate_variability import inst_heart_rate

from morphology import compute_baseline_model

from segmentation import compute_fixed_beats

from pca import compute_PC, trace_evol_PC

from resampling import resample_rpeaks

# # from filtering import baseline_removal, create_filtered_dataset

# # from segmentation import create_rpeak_dataset, create_heart_beat_dataset,\
# #     compute_beats

# from Filtering.gaussian_fit import get_phase, mean_extraction,\
#     beat_fitter, ecg_model

# from Filtering.filter_signal import filter_signal

from biosppy.signals import ecg

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import logging
import sys
import copy
import memory_profiler

def main():

#signal
    #signal
    time_before_seizure = 10
    time_after_seizure = 10
    path_to_load = '~/Desktop/phisionet_seizures.h5'
    sampling_rate = 200
    # path_to_load = '~/Desktop/seizure_datasets.h5'
    name_list = [str(time_before_seizure*60) + '_' + str(time_after_seizure*60)]
    group_list_raw = ['raw']

    # Raw signal 
    signal_structure_raw = load_signal(path_to_load,zip(group_list_raw, name_list))
    # one_signal_structure_raw = get_one_signal_structure(signal_structure, zip(group_list, name_list)[0])
    # records_raw = get_multiple_records(one_signal_structure_raw)

    # stop # ========================================================

    # Baseline removal
    group_list_baseline_removal = ['medianFIR']
    try:
        signal_structure_baseline_removal = load_signal(path_to_load,zip(group_list_baseline_removal, name_list))
    except Exception as e:
        _logger.debug(e)
        baseline_removal(path_to_load, name_list[0], group_list_raw[0], sampling_rate)
        signal_structure_baseline_removal = load_signal(path_to_load,zip(group_list_baseline_removal, name_list))

    # stop # =========================================================

    # Noise removal 
    group_list_noise_removal = ['FIR_lowpass_40hz']
    try:
        signal_structure_noise_removal = load_signal(path_to_load,zip(group_list_noise_removal, name_list))
    except Exception as e:
        print e
        _logger.debug(e)
        noise_removal(path_to_load, name_list[0], group_list_raw[0], sampling_rate)
        signal_structure_noise_removal = load_signal(path_to_load,zip(group_list_noise_removal, name_list))

    # stop # =========================================================

    # Rpeak detection
    # create_rpeak_dataset(path_to_load, zip(group_list_noise_removal, name_list))
    names = lambda label: name_list[0] + '_rpeaks_'+label
    labels = map(str, range(0,5))
    rpeaks_names = map(names, labels)
    group_list = group_list_noise_removal * len(labels)

    try:
        make_new_rpeaks
        rpeaks_signal_structure = load_signal(path_to_load ,zip(group_list, rpeaks_names))
    except Exception as e:
        print e
        _logger.debug(e)
        create_rpeak_dataset(path_to_load, zip(group_list_baseline_removal, name_list), sampling_rate)
        rpeaks_signal_structure = load_signal(path_to_load ,zip(group_list, rpeaks_names))

    # Alocate data *****************************************
    nr_patient = 2
    seizure_nr = 1

    one_signal_structure = get_one_signal_structure(signal_structure_baseline_removal, zip(group_list_baseline_removal, name_list)[0])
    records = get_multiple_records(one_signal_structure)

    one_signal_structure = get_one_signal_structure(rpeaks_signal_structure, zip(group_list, rpeaks_names)[seizure_nr])
    rpeaks = [get_multiple_records(get_one_signal_structure(rpeaks_signal_structure, group_name)) for group_name in zip(group_list, rpeaks_names)]

    print records
    # ******************************************************
    Fs = 1000
    N = len(records[0,:])
    T = (N - 1) / Fs
    time = np.linspace(0, T, N, endpoint=False)
    print time

    rpeaks_resample = resample_rpeaks(np.diff(rpeaks[nr_patient]), rpeaks[nr_patient], time)
    # rpeaks_resample = filter_signal(signal=rpeaks_resample, ftype='FIR', band='lowpass',
    #               order=100, frequency=40,
    #               sampling_rate=Fs)
    plt.subplot(2,1,1)
    plt.plot(np.diff(rpeaks[nr_patient]))
    plt.subplot(2,1,2)
    plt.plot(rpeaks_resample)
    plt.show()


    signal = records[seizure_nr,:]
    rpeaks = rpeaks[seizure_nr]

    stop 

    beat_nr = 10

    start = 0 * 1000
    end = 4 * 60 * 1000


    beats = compute_fixed_beats(signal, rpeaks)
    print np.shape(beats[0:5])
    pca = compute_PC(beats[0:5])
    pca = np.dot(pca, beats[0:5] )
    # print pca

    # stop
    # pca = trace_evol_PC(beats[0:4]) 
    print np.shape(pca)
    plt.plot(pca[2,:])
    plt.show()
    stop
#     plt.subplot(5,1,1)
#     plt.plot(pca[:,4])
#     plt.subplot(5,1,2)
#     plt.plot(pca[:,3])
#     plt.subplot(5,1,3)
#     plt.plot(pca[:,2])
#     plt.subplot(5,1,4)
#     plt.plot(pca[:,1])
#     plt.subplot(5,1,5)
#     plt.plot(pca[:,0])

#     plt.show()
# #     # print(pca.explained_variance_ratio_)
#     # print(pca.components_) 

#     hist = np.histogram(pca[:,4].T, bins=100, range=None, normed=False, weights=None, density=None)
#     # stop
#     # model = compute_baseline_model(signal, rpeaks, start, end)

#     # print len(model)

#     plt.hist(pca[:,0], bins=1000)
#     plt.ylim(0,40)
#     plt.show()

#     stop
#     models = [compute_baseline_model(record, rpeak, start, end) for record,rpeak in zip(records, rpeaks)]

#     print np.shape(models)
main()