
# --------------------------------
from load_for_class import *
from labels import *
from supervised_training import *


# -------------------------------
import pandas as pd
import numpy as np



# Baseline records path
path_to_load_seizure = '/Volumes/ASSD/pre_epi_seizures/h5_files/processing_datasets/seizure_datasets_new.h5'
path_to_load_seizure_map = '/Volumes/ASSD/pre_epi_seizures/h5_files/processing_datasets/seizure_datasets_new_map.txt'

# Baseline records path
path_to_load_baseline = '/Volumes/ASSD/pre_epi_seizures/h5_files/processing_datasets/baseline_datasets_new.h5'
path_to_load_baseline_map = '/Volumes/ASSD/pre_epi_seizures/h5_files/processing_datasets/baseline_datasets_new_map.txt'

    
def load_feature_groups_baseline_seizure_per_patient(patient_list, feature_slot):

    # Feature group to analyse -- point of entry
    feature_name = get_feature_group_name_list(path_to_load_baseline_map,
                                               feature_slot)[0]

    # Load baseline data headers based on patient < --- Different Feature Groups
    # baseline_feature_name_list = get_patient_feature_records(path_to_load_baseline,
    #                                                feature_name,
    #                                                patient_nr)

    baseline_feature_name_list = get_patient_feature_lead_records(path_to_load_baseline,
                                                   feature_name,
                                                   patient_list,
                                                   'ECG')
       
    # Feature group to analyse -- point of entry < --- Different Feature Groups
    feature_name = get_feature_group_name_list(path_to_load_seizure_map,
                                               feature_slot)[0]

    # Load seiure data headers based on patient
    # seizure_feature_name_list = get_patient_feature_records(path_to_load_seizure,
    #                                               feature_name,
    #                                               patient_nr)

    seizure_feature_name_list = get_patient_feature_lead_records(path_to_load_seizure,
                                                  feature_name,
                                                  patient_list,
                                                  'ECG')
    
    baseline_data_struct = load_feature_from_input_list(path_to_load_baseline,
                                                         baseline_feature_name_list)

    baseline_data_window_struct = load_feature_window_from_input_list(path_to_load_baseline,
                                                         baseline_feature_name_list)

    seizure_data_struct = load_feature_from_input_list(path_to_load_seizure,
                                                         seizure_feature_name_list)

    seizure_data_window_struct = load_feature_window_from_input_list(path_to_load_seizure,
                                                         seizure_feature_name_list)
    data_struct = (baseline_data_struct, 
                   baseline_data_window_struct,
                   seizure_data_struct,
                   seizure_data_window_struct)
    
    baseline_data_struct = data_struct[0][0]
    baseline_feature_names = data_struct[0][1]
    baseline_data_window_struct = data_struct[1][0]

    # Get seizure dataset
    seizure_data_struct = data_struct[2][0]
    seizure_feature_names = data_struct[2][1]
    seizure_data_window_struct = data_struct[3][0]
    
    return baseline_data_struct, baseline_feature_names, baseline_data_window_struct,\
           seizure_data_struct, seizure_feature_names, seizure_data_window_struct


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


def select_one_for_all_baseline(ix, data_struct):

    baseline_data_struct = data_struct[0]
    baseline_feature_names = data_struct[1]
    baseline_data_window_struct = data_struct[2]

    seizure_data_struct = data_struct[3]

    return [baseline_data_struct[ix]] * len(seizure_data_struct),\
           [baseline_feature_names[ix]]* len(seizure_data_struct),\
           [baseline_data_window_struct[ix]]* len(seizure_data_struct)


def unite_data_struct(data_struct, feature_names, data_window_struct):

    # Only works in the same feature group
    data_struct = np.concatenate(data_struct, axis=1)
    feature_names = feature_names[0]
    data_window_struct = np.concatenate(data_window_struct, axis=0)

    return data_struct, feature_names, data_window_struct


def select_equal_baseline_data(data_struct):

    baseline_struct = unite_data_struct(data_struct[0],
                                        data_struct[1],
                                        data_struct[2])

    seizure_data_struct = data_struct[3]

    baseline_data_struct = np.array_split(baseline_struct[0], len(seizure_data_struct), axis=1)
    baseline_data_window_struct = np.array_split(baseline_struct[2], len(seizure_data_struct), axis=0)
    baseline_feature_names = [baseline_struct[1]] * len(seizure_data_struct)

    return baseline_data_struct, baseline_feature_names, baseline_data_window_struct


def random_drawing_data(data_struct, data_ix, nr_data_points):
    data_ix = np.random.choice(a=data_ix, size=nr_data_points)
    print data_struct
    random_data_struct = data_struct[:, data_ix]

    return random_data_struct


def random_drawing_data_window(data_window_struct, data_ix, nr_data_points):
    data_ix = np.random.choice(a=data_ix, size=nr_data_points)
    random_data_window_struct = data_window_struct[data_ix]

    return random_data_window_struct



def select_equal_baseline_data_random(data_struct):

    baseline_struct = unite_data_struct(data_struct[0],
                                    data_struct[1],
                                    data_struct[2])

    seizure_data_struct = data_struct[3]

    baseline_data_points = len(baseline_struct[0][0])
    baseline_ix = range(0, baseline_data_points, 1)
    baseline_data_points_per_seizure = baseline_data_points/len(seizure_data_struct)

    # Balanced
    baseline_data_struct = [random_drawing_data(baseline_struct[0],
                                                baseline_ix,
                                                baseline_data_points)
                            for seizure in seizure_data_struct]


    baseline_feature_names = [baseline_struct[1]] * len(seizure_data_struct)


    baseline_data_window_struct = [random_drawing_data_window(baseline_struct[2],
                                           baseline_ix,
                                           baseline_data_points)
                                   for seizure in seizure_data_struct]


    return baseline_data_struct, baseline_feature_names, baseline_data_window_struct


def select_all_seizure(data_struct):
    seizure_data_struct = data_struct[3]
    seizure_feature_names = data_struct[4]
    seizure_data_window_struct = data_struct[5]

    return seizure_data_struct,\
           seizure_feature_names,\
           seizure_data_window_struct

def select_all_baseline(data_struct):
    seizure_data_struct = data_struct[0]
    seizure_feature_names = data_struct[1]
    seizure_data_window_struct = data_struct[2]

    return seizure_data_struct,\
           seizure_feature_names,\
           seizure_data_window_struct


def create_dataframe_baseline_seizure(baseline_struct, seizure_struct):

    seizure_dataframe = create_training(seizure_struct[0],
                                              seizure_struct[1],
                                              seizure_struct[2],
                                              seizure_struct[3])

    seizure_dataframe = seizure_dataframe\
                            .loc[seizure_dataframe['labels'] != -1]

    baseline_dataframe = create_training(baseline_struct[0],
                                          baseline_struct[1],
                                          baseline_struct[2],
                                          baseline_struct[3])

    dataframe = pd.concat((seizure_dataframe, baseline_dataframe))

    return dataframe


def create_baseline_seizure_labels(pre_seizure_window_intrevals,
                                    baseline_data_struct,
                                    baseline_feature_names,
                                    baseline_data_window_struct,
                                    seizure_data_struct,
                                    seizure_feature_names,
                                    seizure_data_window_struct):
    # create labels based on pre_seizure_window_values
    baseline_label_struct = [{'baseline': (-1, [(0, 2 * 60 * 60)], 'g')}] * len(baseline_data_struct)

    seizure_label_struct = [{'pre_seizure': (1, pre_seizure_window_intrevals, 'r'),
                             'seizure':(2, [50 * 60, 70 * 60], 'y')}] * len(seizure_data_struct)

    # get_label_list from the structures
    baseline_labels = get_labels_list( 
                                      baseline_data_struct,
                                      baseline_data_window_struct,
                                      baseline_label_struct)
    seizure_labels = get_labels_list( 
                                      seizure_data_struct,
                                      seizure_data_window_struct,
                                      seizure_label_struct)

    return (baseline_data_struct,\
            baseline_feature_names,\
            baseline_data_window_struct,\
            baseline_labels), \
           (seizure_data_struct,\
            seizure_feature_names,\
            seizure_data_window_struct,
            seizure_labels)


def draw_random_data_points(df, nr_data_points):
    return df.sample(n=nr_data_points)


def _random_balanced_pair_seizure_baseline(seizure_df_single_group,
                                           baseline_df):
    # get nr of data points for this seizure
    nr_seizure_data_points = len(seizure_df_single_group[1]['group'])
    print nr_seizure_data_points

    # get baseline_data_points
    baseline_df = draw_random_data_points(baseline_df,
                                          nr_seizure_data_points)

    baseline_df['group'] = seizure_df_single_group[0]

    return baseline_df


def random_balanced_pair_seizure_baseline(df):
    # Get seizure related data points
    seizure_df = df.loc[df['labels'] >= 1]

    # Get baseline related data points
    baseline_df = df.loc[df['labels'] <= -1]

    # Get balanced nr of baseline data points
    baseline_df_list = [_random_balanced_pair_seizure_baseline(seizure_df_single_group,
                                                baseline_df)
                        for seizure_df_single_group in seizure_df.groupby('group')]

    # convert to dataframe
    baseline_df = pd.concat(baseline_df_list)
    dataframe = pd.concat([seizure_df, baseline_df])

    return dataframe





def prep_input_supervised_baseline_seizure(data_struct, pre_seizure_window_intrevals):

    # Baseline records selection
    baseline_struct = select_equal_baseline_data(data_struct)
    seizure_struct = select_all_seizure(data_struct)

    baseline_struct, seizure_struct = create_baseline_seizure_labels(pre_seizure_window_intrevals,
                                      baseline_struct[0],
                                      baseline_struct[1],
                                      baseline_struct[2],
                                      seizure_struct[0],
                                      seizure_struct[1],
                                      seizure_struct[2])

    dataframe = create_dataframe_baseline_seizure(baseline_struct, seizure_struct)

    print dataframe
    #dataframe = random_balanced_pair_seizure_baseline(dataframe)

    return dataframe




# # Select group of features
# feature_namea = 'hrv_time_features#'

# # Select Patient to analyze
# pt = 3
# data_struct = load_feature_groups_baseline_seizure_per_patient(
#                                                 patient_nr=pt
#                                                 )

# # stop

# prep_input_supervised_baseline_seizure(data_struct,[(0, 10*60)])