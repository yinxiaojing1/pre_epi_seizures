from pre_epi_seizures.storage_utils.storage_utils_hdf5 import \
    load_signal, save_signal, delete_signal, list_group_signals

from pre_epi_seizures.Preprocessing.pre_processing import *

# from pre_epi_seizures.stats_utils.statistics import create_set_from_disk

import matplotlib.pyplot as plt
import seaborn as sns

import os

# Code needs improving
def load_all_features_from_disk(path_to_load, feature_group_name):
    feature_group_extracted = feature_group_name
    feature_group_name_extracted = list_group_signals(path_to_load, feature_group_extracted)['signals']
    feature_group_name_extracted= [group_name for group_name in feature_group_name_extracted if 'window_' not in group_name[1]]

    try:
        signal_structure = load_signal(path_to_load, feature_group_name_extracted)
        extracted_features = [get_multiple_records(get_one_signal_structure(signal_structure, group_name))
                              for group_name in feature_group_name_extracted
                              if 'window' not in group_name[1]]
        mdata = [get_mdata_dict(get_one_signal_structure(signal_structure, group_name))
                 for group_name in feature_group_name_extracted
                 if 'window' not in group_name[1]]

        return extracted_features, mdata
    except Exception as e:
        _logger.debug(e)

    return extracted_features, mdata


def load_all_feature_windows_from_disk(path_to_load, feature_group_name):
    feature_group_extracted = feature_group_name
    feature_group_name_extracted = list_group_signals(path_to_load, feature_group_extracted)['signals']
    feature_group_name_extracted= [group_name for group_name in feature_group_name_extracted if 'window_' in group_name[1]]

    try:
        signal_structure = load_signal(path_to_load, feature_group_name_extracted)
        extracted_features = [get_multiple_records(get_one_signal_structure(signal_structure, group_name))
                              for group_name in feature_group_name_extracted]
        mdata = [get_mdata_dict(get_one_signal_structure(signal_structure, group_name))
                 for group_name in feature_group_name_extracted]

        return extracted_features, mdata
    except Exception as e:
        _logger.debug(e)

    return extracted_features, mdata


def load_single_feature_from_disk(path_to_load, feature_group_name_record):
    feature_group_name_extracted = [feature_group_name_record]
    try:
        signal_structure = load_signal(path_to_load, feature_group_name_extracted)
        extracted_features = [get_multiple_records(get_one_signal_structure(signal_structure, group_name))
                              for group_name in feature_group_name_extracted]
        mdata = [get_mdata_dict(get_one_signal_structure(signal_structure, group_name))
                 for group_name in feature_group_name_extracted]

        return extracted_features, mdata

    except Exception as e:
        _logger.debug(e)


def load_feature_from_input_list(path_to_load, feature_group_name_record_list):
    feature_group_name_extracted = feature_group_name_record_list

    signal_structure = load_signal(path_to_load, feature_group_name_extracted)
    try:
        print 'starting to load signals...'
        signal_structure = load_signal(path_to_load, feature_group_name_extracted)
        
        print 'here are the signals?....'
        print signal_structure
        
        extracted_features = [get_multiple_records(get_one_signal_structure(signal_structure, group_name))
                              for group_name in feature_group_name_extracted]
           
        print 'here are the signals?....'
        print extracted_features
       
        
        mdata = [get_mdata_dict(get_one_signal_structure(signal_structure, group_name))
                 for group_name in feature_group_name_extracted]

        return extracted_features, mdata

    except Exception as e:
        print signal_structure
        print e
        stop

        
def get_patient_seizure_from_input_list(path_to_load, feature_group_name_record_list):
    patient_seizure_list = [{'patient_nr': group_name[0], 'seizure': group_name[-1]}
                            for group_name in feature_group_name_record_list]
    return patient_seizure_list


def load_feature_window_from_input_list(path_to_load, feature_group_name_record_list):
    feature_group_name_extracted = [(feature_group_name_record[0],
                                     'window_' + feature_group_name_record[1])
                                    for feature_group_name_record \
                                        in feature_group_name_record_list]
    try:
        signal_structure = load_signal(path_to_load, feature_group_name_extracted)
        extracted_features = [get_multiple_records(get_one_signal_structure(signal_structure, group_name))
                              for group_name in feature_group_name_extracted]
        mdata = [get_mdata_dict(get_one_signal_structure(signal_structure, group_name))
                 for group_name in feature_group_name_extracted]

        return extracted_features, mdata

    except Exception as e:
        _logger.debug(e)




def get_patient_feature_records(path_to_load,
                                feature_name,
                                patient_nr):

    # Select the baseline records from the input patient number
    list_all = list_group_signals(path_to_load,
                                  feature_name)['signals']


    list_patient = [group_name
                    for group_name in list_all
                    if group_name[1][0] == str(patient_nr)]

    return list_patient


def get_patient_feature_lead_records(path_to_load,
                                feature_name,
                                patient_list, 
                                lead_list):

    # Select the baseline records from the input patient number
    
    list_all = list_group_signals(path_to_load,
                                  feature_name)['signals']


    list_patient = [group_name
                    for group_name in list_all
                    for patient_nr in patient_list
                    for lead in lead_list
                    if group_name[1][0] == str(patient_nr)
                    and lead in group_name[1]]
    

    return list_patient







        
    



# def load_all_features_from_set(path_to_load, set_feature_group_name):

#         try:
#         print feature_group_extracted
#         signal_structure = load_signal(path_to_load, feature_group_name_extracted)
#         extracted_features = [get_multiple_records(get_one_signal_structure(signal_structure, group_name))
#                               for group_name in feature_group_name_extracted]
#         mdata = [get_mdata_dict(get_one_signal_structure(signal_structure, group_name))
#                  for group_name in feature_group_name_extracted]

#         return extracted_features, mdata
#     except Exception as e:
#         _logger.debug(e)


# path_to_load = '/Volumes/ASSD/pre_epi_seizures/h5_files/processing_datasets/seizure_datasets_new.h5'
# path_to_map= '/Volumes/ASSD/pre_epi_seizures/h5_files/processing_datasets/seizure_datasets_new_map.txt'
# path_to_save = '/Volumes/ASSD/pre_epi_seizures/plotting_utils/teste/'

# feature_group_name = 'hrv_time_features'


# feature_groups = get_feature_group_name_list(path_to_map,
#                                              feature_group_name + '#')

# print feature_groups