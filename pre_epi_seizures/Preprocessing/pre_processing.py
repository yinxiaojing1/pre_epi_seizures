from pre_epi_seizures.storage_utils.storage_utils_hdf5 import \
    load_signal, save_signal, delete_signal, list_group_signals

from pre_epi_seizures.storage_utils.data_handlers import *

from pre_epi_seizures.logging_utils.formatter_logging import logger as _logger

from filtering import baseline_removal, noise_removal, create_filtered_dataset

from segmentation import create_rpeak_dataset, create_heart_beat_dataset,\
    compute_beats, find_rpeaks, detect_rpeaks

from resampling import resample_rpeaks, interpolate_signal

from visual_inspection import visual_inspection

from morphology import *
# from Filtering.gaussian_fit import get_phase, mean_extraction,\
#     beat_fitter, ecg_model

from Filtering.filter_signal import filter_signal

from Filtering.eksmoothing import *

from biosppy.signals import ecg

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
    return [group_name[1] for group_name in group_name_list]

def load_feature(path_to_load, feature_to_load, files='just_new_data', sampling_rate=1000, **feature_groups_required):

    feature_group_to_process = feature_groups_required['feature_group_to_process']

    feature_group_extracted = [feature_group_to_process + '/' + feature_to_load]

    print feature_group_to_process
    print feature_group_extracted

    if files=='all_new':
        print feature_groups_required
        for k in feature_groups_required.keys():
            feature_groups_required[k] = list_group_signals(path_to_load, feature_groups_required[k])['signals']
        names_to_save = get_names(list_group_signals(path_to_load, feature_group_to_process)['signals'])

    if files=='just_new_data':
        print feature_groups_required
        for k in feature_groups_required.keys():
            feature_groups_required[k] = list_group_signals(path_to_load, feature_groups_required[k])['signals']
        #*****************IMPORTANT CHANGE***************************
        names_to_save = get_names(list_group_signals(path_to_load, feature_group_to_process)['signals'])

    if files=='existent':
        feature_group_name_extracted = list_group_signals(path_to_load, feature_groups_required[k])['signals']
        try:
            signal_structure = load_signal(path_to_load, feature_group_name_extracted)
            extracted_features = [get_multiple_records(get_one_signal_structure(signal_structure, group_name)) for group_name in feature_group_name_extracted]
            return extracted_features
        except Exception as e:
            _logger.debug(e)

    for k in feature_groups_required.keys():
        signal_structure = load_signal(path_to_load, feature_groups_required[k])
        feature_groups_required[k] = [get_multiple_records(get_one_signal_structure(signal_structure, group_name)) for group_name in feature_groups_required[k]]

    extracted_features, mdata_features = globals()[feature_to_load](feature_groups_required, sampling_rate)

    save_signal(path_to_load, extracted_features, mdata_features, feature_group_extracted, names_to_save)

    return extracted_features, mdata



# @profile
def main():

    #signal
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

    group_list = [str(
    time_before_seizure*60) + '_' + str(time_after_seizure*60) + '/raw']
    # group_name_list = list_group_signals(path_to_load, group_list[0])['signals']
    # compress(path_to_load, group_name_list)

    load_feature(path_to_load,'baseline_removal', feature_group_to_process=group_list[0])
    # print list_group_signals(path_to_load, group_list[0])
    return




if __name__ == '__main__':
    main()

# g1