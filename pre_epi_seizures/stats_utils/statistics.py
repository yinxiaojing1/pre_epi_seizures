from pre_epi_seizures.Preprocessing.pre_processing import load_feature

from pre_epi_seizures.classification.labels import create_labels

from pre_epi_seizures.classification.scaling import *


from pre_epi_seizures.storage_utils.storage_utils_hdf5 import \
    load_signal, save_signal, delete_signal, list_group_signals

from pre_epi_seizures.storage_utils.data_handlers import *

from hist import *

from pre_epi_seizures.storage_utils.patients_data_new import patients

from datetime import datetime

import numpy as np


# # def create_labels_for_feature

def create_save_structure_single_feature_group(set_structure,
                                               feature_group):
    return (feature_group, set_structure)


def create_set_from_disk(path_to_load, feature_group):
    list_signals = list_group_signals(path_to_load, feature_group)['signals']

    return list_signals


def get_record_dimension(feature_group):
    len_record_before_str = feature_group[0: feature_group.index('_')]
    len_record_after_str = feature_group[feature_group.index('_')+1:feature_group.index('/')]

    # print len_record_after_str
    # print len_record_before_str

    len_record = int(len_record_before_str) + int(len_record_after_str)

    return len_record


def get_before_after(feature_group):
    len_record_before_str = feature_group[0: feature_group.index('_')]
    len_record_after_str = feature_group[feature_group.index('_')+1:feature_group.index('/')]

    return (int(len_record_before_str), int(len_record_after_str))


def get_window_sample(feature_group):
    try:
        window_str = feature_group[feature_group.index('w'):feature_group.index('msec')]
    except Exception as e:
        print e
        window_sample = 1
    return window_sample


def segmentation_rpeak_feature(segmentation_signal_group_name):
    signal_group_name = segmentation_signal_group_name
    feature_group = signal_group_name[0]
    signal_name = signal_group_name[1]
    raw_feature_index = feature_group.rfind('/')
    raw_feature =  feature_group[0:raw_feature_index]
    rpeaks_group_name = raw_feature + '/' + 'rpeak_detection'
    return rpeaks_group_name


def create_window_segmentation_signal(path_to_load,
                                      segmentation_signal_group_name):
    signal_group_name = segmentation_signal_group_name

    # Feature group
    feature_group = signal_group_name[0]

    # Signal name
    signal_name = signal_group_name[1]

    rpeaks_group_name = segmentation_rpeak_feature(signal_group_name)
    rpeaks_structure = load_signal(path_to_load, [(rpeaks_group_name, signal_name)])
    rpeaks_array = get_multiple_records(get_one_signal_structure(rpeaks_structure, (rpeaks_group_name, signal_name)))
    windows = rpeaks_array[0]
    return windows


def _create_win(path_to_load, signal_group_name, sampling_rate):
    """ From structure with feature_group and signal name create
     appropriate time domain.
    
    Parameters
    ----------
    signal_group_name : Tuple
    Structure with feature_name and signal_name

    Returns
    -------
    win : numpy array
    Appropriate time window.
    """

    # Feature group
    feature_group = signal_group_name[0]

    # Signal name
    signal_name = signal_group_name[1]

    # Get duration of seizure
    len_record = get_record_dimension(feature_group)

    try:
        window_str = feature_group[feature_group.index('w'):feature_group.index('msec')]
    except Exception as e:
        print e

        if 'segmentation' in feature_group:
            windows = create_window_segmentation_signal(path_to_load,
                                            signal_group_name)

        else: 
            window_sample = 1
            windows = np.arange(0, len_record * sampling_rate, window_sample)


    return windows


def create_win(path_to_load, list_signals, sampling_rate):
    windows = [_create_win(path_to_load, signal_group_name, sampling_rate)
                for signal_group_name in list_signals]
    return windows


def make_pointers_patients(group_name):
    filename = group_name[1]
    patient_number = int(filename[0])
    seizure_number = int(filename[-1])

    return (patient_number, seizure_number)


def compute_ictal_duration(group_name):
    filename = group_name[1]
    patient_number = int(filename[0])
    seizure_number = int(filename[-1])


    ictal = datetime.combine(datetime(1, 1, 1),
                                  patients[str(patient_number)]
                                  ['ictal_on_time']
                                  [seizure_number]
                                  )
    post_ictal = datetime.combine(datetime(1, 1, 1),
                                  patients[str(patient_number)]
                                  ['post_ictal_time']
                                  [seizure_number]
                                  )

    ictal_duration_sec = (post_ictal - ictal).total_seconds()

    ictal_duration_min = np.true_divide(ictal_duration_sec, 60)

    return ictal_duration_min


def _ictal_limits(ictal_duration, before):
    post_ictal = np.true_divide(before, 60) + ictal_duration
    ictal = np.true_divide(before, 60)
    return (ictal, post_ictal)


def create_ictal_limits(list_signals, before_list):
    ictal_durations = [compute_ictal_duration(group_name)
                       for group_name in list_signals]

    ictal_limits = [_ictal_limits(ictal_duration, before)
                    for ictal_duration, before
                    in zip(ictal_durations, before_list)]

    return (ictal_limits, 'r')


def create_post_ictal_limits(ictal_limits, up):
    post_ictal_limits = [(limits[1], up)
                        for limits in ictal_limits[0]]

    return (post_ictal_limits, 'orange')


def create_pre_ictal_limits(ictal_limits, low):
    pre_ictal_limits = [(low, limits[0])
                        for limits in ictal_limits[0]]

    return (pre_ictal_limits, 'b')


def create_limits(low, up, list_signals, color):
    nr_limits = len(list_signals)
    limits = ([(low, up)] * nr_limits, color)
    return limits


def create_labels_from_list(path_to_load, list_signals,
                            pre_ictal_low, post_ictal_up, sampling_rate=1000):
    windows = create_win(path_to_load, list_signals, sampling_rate)

    before_after_list= [get_before_after(signal_group_name[0])
                        for signal_group_name in list_signals]

    before_list = [before_after[0]
                   for before_after in before_after_list]

    # Ictal limits based on signal list
    ictal_limits = create_ictal_limits(list_signals, before_list)

    post_ictal_limits = create_post_ictal_limits(ictal_limits, post_ictal_up)

    pre_ictal_limits = create_pre_ictal_limits(ictal_limits, pre_ictal_low)

    inter_ictal_limits = create_limits(0, pre_ictal_low, list_signals, 'g')

    inter_ictal_second_limits = create_limits(post_ictal_up, 70, list_signals, 'c')

    labels = create_labels(sampling_rate,
                            windows_list=windows,
                            inter_ictal=inter_ictal_limits,
                            pre_ictal=pre_ictal_limits,
                            post_ictal=post_ictal_limits,
                            ictal=ictal_limits,
                            inter_ictal_second=inter_ictal_second_limits)
    return labels


def records_intensive(path_to_load, signal_group_name_list):
    # print signal_group_name_list
    # stop
    signal_structure = load_signal(path_to_load,
                                           signal_group_name_list)
    feature_groups_extracted = [get_multiple_records(get_one_signal_structure(signal_structure, group_name))
                                for group_name in signal_group_name_list]

    return feature_groups_extracted


def compute_statistic(path_to_load, path_to_save, statistic_to_load,
                      group_to_save,
                      name_to_save,
                      pre_ictal_low,
                      post_ictal_up,
                      sampling_rate=1000,
                      **feature_groups_required):


    # *** MEM intensive ****************************************
    # Check signals to load and labels
    labels = dict()
    feature_groups_extracted = dict()

    param_str = '_preictallow:' + str(pre_ictal_low) + '_postictalup:' + str(post_ictal_up)

    # # create labels
    #     labels[k] = create_labels_feature(path_to_load, feature_groups_required[k],
    # #                                       feature_groups_to_load[k], sampling_rate)
    # load all files from group --Memory intensive

    for k in feature_groups_required.keys():
        feature_groups_extracted[k] = scale(records_intensive(path_to_load, feature_groups_required[k]))
        labels[k] = create_labels_from_list(path_to_load,
                                            feature_groups_required[k],
                                            pre_ictal_low,
                                            post_ictal_up,
                                            sampling_rate)
        path_name_to_save_list = feature_groups_required[k]

    stat, mdata = globals()[statistic_to_load](feature_groups_extracted, labels, sampling_rate)
    # print feature_groups_extracted

    # print stat, mdata

    # print str(labels['feature_group_to_process'])
    # # stop

    group_to_save = group_to_save + '/'  + statistic_to_load + '_' + param_str
    delete_signal(path_to_save, [name_to_save], [group_to_save])
    save_signal(path_to_save, [stat], [mdata], [name_to_save], [group_to_save])
    # print labels

    # **************************************************************



    # Memory intensive
    # stop







def main():

    #signal
    sampling_rate = 1000
    time_before_seizure = 50
    time_after_seizure = 20
    # path_to_load = '~/Desktop/phisionet_seizures_new.h5'
    # sampling_rate = 1000
    path_to_load = '/Volumes/ASSD/pre_epi_seizures/h5_files/processing_datasets/seizure_datasets_new.h5'

    path_to_save = '/Volumes/ASSD/pre_epi_seizures/h5_files/processing_datasets/satistic_datasets.h5'
    dataset_name = str(
    time_before_seizure*60) + '_' + str(time_after_seizure*60)
    raw_name = 'raw'
    baseline_removal_name = 'baseline_removal'
    raw_dataset_name = dataset_name + '/' + raw_name
    baseline_removal_dataset_name = raw_dataset_name + '/' + baseline_removal_name
    decimated_dataset_name = baseline_removal_dataset_name + '/' + 'decimation'
    eks_dataset_name = decimated_dataset_name + '/' + 'eks_smoothing'
    interpolated_dataset_name = eks_dataset_name + '/' + 'interpolation'
    Qrs_fixed_dataset_name = baseline_removal_dataset_name + '/' + 'QRS_fixed_segmentation'
    # dsfsdf
    # raw = load_feature(path_to_load, raw_name, files='existent', feature_group_to_process=dataset_name)[0]



    set_structure = create_set_from_disk(path_to_load, baseline_removal_dataset_name)
    # stop
    # save_structure = create_save_structure_single_feature_group([set_structure],
                                                                # baseline_removal_dataset_name)
    # stop
    compute_statistic(path_to_load, path_to_save, 'histogram',
                      group_to_save=baseline_removal_dataset_name,
                      name_to_save='all_data',
                      pre_ictal_low=10,
                      post_ictal_up=60,
                      sampling_rate=1000,
                      feature_group_to_process=set_structure)


    for signal_group_name in set_structure:
        compute_statistic(path_to_load, path_to_save, 'histogram',
                      group_to_save=baseline_removal_dataset_name,
                      name_to_save=signal_group_name[1],
                      pre_ictal_low=10,
                      post_ictal_up=60,
                      sampling_rate=1000,
                      feature_group_to_process=[signal_group_name])


    stop


main()