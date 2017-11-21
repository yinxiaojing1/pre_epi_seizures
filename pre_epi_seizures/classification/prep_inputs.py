
# --------------------------------
from load_for_class import *
from labels import *
from supervised_training import *


# -------------------------------
import pandas as pd



# Baseline records path
path_to_load_seizure = '/Volumes/ASSD/pre_epi_seizures/h5_files/processing_datasets/seizure_datasets_new.h5'
path_to_load_seizure_map = '/Volumes/ASSD/pre_epi_seizures/h5_files/processing_datasets/seizure_datasets_new_map.txt'

# Baseline records path
path_to_load_baseline = '/Volumes/ASSD/pre_epi_seizures/h5_files/processing_datasets/baseline_datasets_new.h5'
path_to_load_baseline_map = '/Volumes/ASSD/pre_epi_seizures/h5_files/processing_datasets/baseline_datasets_new_map.txt'

# label_structure
label_structure = {'baseline':(1, [], 'g'), 'pre_seizure': (2, [], 'y')}


# DUMP
# def create_label_structure_for_all_records(len_data_struct,
#                                            len_data_struct,
#                                            window):
#     label_struct = [{'baseline': (1, [(0, 50 * scale)], 'g')}] * len_baseline_data_struct
#     scale = 60
#     baseline_label_struct = [{'baseline': (1, [(0, 50 * scale)], 'g')}] * len_baseline_data_struct
#     seizure_label_struct = [{'pre_seizure': (2, [(0, pre_seizure_window * scale)], 'r')}] * len_seizure_data_struct
#     return baseline_label_struct, seizure_label_struct


def flatten(list):
    flat_list = [item for sublist in l for item in sublist]
    return flat_list


def get_labels_list(data_struct,
                    data_window_struct,
                    label_struct):

    labels = [create_label_from_feature_record(feature_record,
                                               feature_record_window,
                                               data_label,
                                               1000)
                       for feature_record, feature_record_window, data_label\
                       in zip(data_struct,
                              data_window_struct,
                              label_struct)]

    return labels


def itemgetter(ix_list,list_):
    
    return[list_[i] for i in xrange(0,len(list_)) if i in ix_list ]


def get_set_ix(ix,
               data_struct,
               feature_names,
               data_window_struct,
               label_struct):
    if type(ix) != list:
      ix = [ix]

    # stop
    data_struct = itemgetter(ix, data_struct)
    feature_names = itemgetter(ix, feature_names)
    data_window_struct = itemgetter(ix, data_window_struct)
    label_struct = itemgetter(ix, label_struct)

    return data_struct, data_window_struct,\
           feature_names, label_struct



def get_training_ix(test_ix, data_struct):
    ix_data_struct = range(0, len(data_struct))
    return[ix
           for ix in ix_data_struct
           if ix != test_ix]


def load_feature_groups_baseline_seizure_per_patient(patient_nr):

    # # Patient to analyse
    # patient_nr = 3

    # Feature group to analyse -- point of entry
    feature_name = get_feature_group_name_list(path_to_load_baseline_map,
                                               'hrv_time_features#')[-1]

    # Load baseline data headers based on patient < --- Different Feature Groups
    baseline_feature_name_list = get_patient_feature_records(path_to_load_baseline,
                                                   feature_name,
                                                   patient_nr)

    # Feature group to analyse -- point of entry < --- Different Feature Groups
    feature_name = get_feature_group_name_list(path_to_load_seizure_map,
                                               'hrv_time_features#')[-1]


    # Load seiure data headers based on patient
    seizure_feature_name_list = get_patient_feature_records(path_to_load_seizure,
                                                  feature_name,
                                                  patient_nr)

    baseline_data_struct = load_feature_from_input_list(path_to_load_baseline,
                                                         baseline_feature_name_list)

    baseline_data_window_struct = load_feature_window_from_input_list(path_to_load_baseline,
                                                         baseline_feature_name_list)

    seizure_data_struct = load_feature_from_input_list(path_to_load_seizure,
                                                         seizure_feature_name_list)

    seizure_data_window_struct = load_feature_window_from_input_list(path_to_load_seizure,
                                                         seizure_feature_name_list)

    data_struct = (baseline_data_struct, baseline_data_window_struct, seizure_data_struct, seizure_data_window_struct)
    baseline_data_struct = data_struct[0][0]
    baseline_feature_names = data_struct[0][1]
    baseline_data_window_struct = data_struct[1][0]

    # Get seizure dataset
    seizure_data_struct = data_struct[2][0]
    seizure_feature_names = data_struct[2][1]
    seizure_data_window_struct = data_struct[3][0]

    return baseline_data_struct, baseline_feature_names, baseline_data_window_struct,\
           seizure_data_struct, seizure_feature_names, seizure_data_window_struct




def create_supervised_set_baseline_seizure(data_struct,
                                           prediction_window_min):
    # Get baseline dataset
    baseline_data_struct = data_struct[0][0]
    baseline_feature_names = data_struct[0][1]
    baseline_data_window_struct = data_struct[1][0]

    # Get seizure dataset
    seizure_data_struct = data_struct[2][0]
    seizure_feature_names = data_struct[2][1]
    seizure_data_window_struct = data_struct[3][0]

    # Label structure for analysis * < -- create_label_structure
    baseline_label_struct, seizure_label_struct =\
                         create_label_structure(
                                len(baseline_data_struct),
                                len(seizure_data_struct),
                                prediction_window_min)


    stop

    # Get_labels for classification
    baseline_label, seizure_labelget_labels_list(baseline_label_struct, seizure_label_struct,
                    baseline_data_struct, seizure_data_struct,
                    baseline_data_window_struct,
                    seizure_data_window_struct)



    # Create training set

    # trainig on criterion * <-- create_training_criterion
    training_seizure_ix = len(seizure_data_struct)-1
    training_baseline_ix = 0
    training_seizure_data_struct = seizure_data_struct[0:training_seizure_ix]
    training_seizure_data_window_struct = seizure_data_window_struct[0:training_seizure_ix]
    training_baseline_data_struct = [baseline_data_struct[training_baseline_ix]]
    training_baseline_data_window_struct = [baseline_data_window_struct[training_baseline_ix]]


    # training Dataframes
    training_seizure = create_training(training_seizure_data_struct,
                                       training_seizure_data_window_struct,
                                       labels_seizure,
                                       seizure_feature_names)


    training_baseline = create_training(training_baseline_data_struct,
                                        training_baseline_data_window_struct,
                                        labels_baseline,
                                        baseline_feature_names)

    # Initial discard of data
    training_seizure = training_seizure.loc[training_seizure['labels'] != -1]

    # Colors
    training_seizure['colors'] = 'r'
    training_baseline['colors'] = 'g'
    print len(training_seizure)
    print len(training_baseline)

    return training_baseline, training_seizure


