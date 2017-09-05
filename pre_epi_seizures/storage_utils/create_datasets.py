
from pre_epi_seizures.logging_utils.formatter_logging import\
    logger as _logger

from storage_utils_hdf5 import\
    load_signal, save_signal, list_group_signals

from data_handlers import\
    get_record, get_sampling_frequency, get_seizure_times_seconds,\
    get_one_signal_structure, parse_sample_seizure


import numpy as np


def fetch_group_seizures(patient_number):
    return '/PATIENT' + str(patient_number) + '/crysis'


def fetch_group_free(patient_number):
    return '/PATIENT' + str(patient_number) + '/free'


def list_seizures_files_patient(path_to_load, patient_number):
    list_group = list_group_signals(path=path_to_load,
                                    group=fetch_group_seizures(
                                        patient_number))
    return list_group['signals']


def list_free_files_patient(path_to_load, patient_number):
    list_group = list_group_signals(path=path_to_load,
                                    group=fetch_group_free(
                                        patient_number))
    return list_group['signals']


def list_all_files_patient(path_to_load, patient_number):
    list_seizures = list_seizures_files_patient(path_to_load, patient_number)
    list_free = list_free_files_patient(path_to_load, patient_number)
    list_all = list_seizures + list_free
    list_sorted = sorted([(item[1], item[0]) for item in list_all])
    return [(item[1], item[0]) for item in list_sorted]


def list_seizures_from_file(one_signal_structure):
    one_signal_structure


def get_record_dataset_seizure(path_to_load,
                               samples_before_seizure, samples_after_seizure,
                               sample_seizure, group_name_all_list,
                               group_name_seizure, total_record):

    diff_before_seizure = int(sample_seizure - samples_before_seizure)
    diff_after_seizure = int(sample_seizure + samples_after_seizure)
    samples_record = 2 * 3600 * 1000

    if diff_before_seizure < 0:

        group_name_previous_file = group_name_all_list[
            group_name_all_list.index(group_name_seizure) - 1]

        previous_record = get_record(
            one_signal_structure=get_one_signal_structure(
                signals_structure=load_signal(path=path_to_load,
                                              group_name_list=[group_name_previous_file]),
            group_name=group_name_previous_file))

        previous_record_to_append = previous_record[diff_before_seizure:]
        total_record = np.concatenate((previous_record_to_append,
                                       total_record))
        samples_added = abs(diff_before_seizure)
        diff_before_seizure = int(diff_before_seizure + samples_added)
        diff_after_seizure = int(diff_after_seizure + samples_added)
        samples_record = int(samples_record + samples_added)

    if diff_after_seizure > samples_record:

        group_name_next_file = group_name_all_list[
            group_name_all_list.index(group_name_seizure) + 1]

        next_record = get_record(
            one_signal_structure=load_signal(
                path=path_to_load,
                group_name_list=[group_name_next_file]))

        next_record_to_append = next_record[
            :diff_after_seizure - samples_record]

        total_record = np.concatenate((total_record, next_record_to_append))

    return total_record[diff_before_seizure:diff_after_seizure]


def get_record_dataset_seizure_file(path_to_load,
                                    time_before_seizure,
                                    time_after_seizure,
                                    group_name_all_list,
                                    group_name_file_seizure):

    # _logger.debug(group_name_file_seizure)
    signals_structure = load_signal(path_to_load, [group_name_file_seizure])
    one_signal_structure = get_one_signal_structure(signals_structure,
                                                    group_name_file_seizure)
    total_record = get_record(one_signal_structure)
    Fs = get_sampling_frequency(one_signal_structure)
    seizure_times = get_seizure_times_seconds(one_signal_structure)
    sample_seizures =map(parse_sample_seizure,
                          seizure_times)
    samples_before_seizure = int(time_before_seizure * Fs)
    samples_after_seizure = int(time_after_seizure * Fs)

    dataset_seizures_file = [
        get_record_dataset_seizure(
            path_to_load=path_to_load,
            samples_before_seizure=samples_before_seizure,
            samples_after_seizure=samples_after_seizure,
            sample_seizure=sample_seizure,
            group_name_all_list=group_name_all_list,
            group_name_seizure=group_name_file_seizure,
            total_record=total_record)
        for sample_seizure in sample_seizures]

    _logger.debug(np.shape(np.asarray(dataset_seizures_file)))

    # stop
    return dataset_seizures_file 


def create_seizure_dataset_patient(path_to_load, path_to_save,
                                   time_before_seizure,
                                   time_after_seizure, patient_number):
    _logger.debug('searching all files')
    group_name_all_list = list_all_files_patient(path_to_load=path_to_load,
                                                 patient_number=patient_number)
    _logger.debug(group_name_all_list)
    # this is stupid ... doesnt have much impact tough
    group_name_seizure_list = list_seizures_files_patient(
        path_to_load=path_to_load,
        patient_number=patient_number)

    _logger.debug(group_name_seizure_list)

    group_name_already_saved = list_group(path_to_load, )
    # stop
    # X = load_signal(path=path_to_load,
    #                 group_name_list=group_name_list)
    stop
    dataset_list =\
        [get_record_dataset_seizure_file(
            path_to_load,
            time_before_seizure=time_before_seizure,
            time_after_seizure=time_after_seizure,
            group_name_all_list=group_name_all_list,
            group_name_file_seizure=group_name)
            for group_name in group_name_seizure_list]

    dataset_list = [val for sublist in dataset_list for val in sublist]

    # mdata = group_name_seizure_list

    _logger.error('the dataset_list is the following: %s', dataset_list)
    return dataset_list


def create_seizure_dataset(path_to_load, path_to_save,
                           time_before_seizure,
                           time_after_seizure, *args):
    seizure_dataset =\
        [create_seizure_dataset_patient(path_to_load,
                                        path_to_save,
                                        time_before_seizure,
                                        time_after_seizure,
                                        arg)
         for arg in args]
    _logger.debug('Befor_saving')
    stop

    seizure_dataset = [val for sublist in seizure_dataset for val in sublist]
    group_list = ['raw']
    name_list = [str(time_before_seizure) + '_' + str(time_after_seizure)]
    mdata_list = ['']
    signal_list = [seizure_dataset]
    save_signal(path_to_save, signal_list, mdata_list, name_list, group_list)
    _logger.debug(seizure_dataset)

    # mdata = create_seizure_mdata(path_to_load, *args)

    # _logger.debug(mdata)

    return np.asmatrix(seizure_dataset)


# def create_seizure_mdata(path_to_load, *args):

#     mdata = [list_seizures_files_patient(
#         path_to_load=path_to_load,
#         patient_number=patient_number)
#         for patient_number in args]

#     mdata = [val for sublist in mdata for val in sublist]

#     return mdata


# def save_seizure_dataset(seizure_dataset, group_name_list)
#     []
#     mdata = group_name_list
#     singal = save_signal


_logger.setLevel(10)
path_to_load = '~/Desktop/HSM_data.h5'
path_to_save = '~/Desktop/seizure_datasets_tests.h5'
patient_number = 1

time_before_seizure = 30 * 60
time_after_seizure = 10 * 60

dataset = create_seizure_dataset(path_to_load, path_to_save,
                                 time_before_seizure,
                                 time_after_seizure, 2)

_logger.debug('the dataset is the following: %s', dataset)
_logger.debug(np.shape(dataset))
# a = list_all_blocks(path_to_load, patient_number)

# # b = sorted(a)
# _logger.debug('Existent_Signals: %s', a)

# # raw = load_signal(path_to_load, a)

# # _logger.debug(raw)
