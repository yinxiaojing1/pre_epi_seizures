from load_for_class import *
from labels import *
from supervised_training import *


import pandas as pd
import numpy as np


def load_records(path_to_load, path_to_map, patient_list, feature_slot, lead_list):
    # Feature group to analyse -- point of entry
    feature_name = get_feature_group_name_list(path_to_map,
                                               feature_slot)[0]
    
    # Load records as lists, numpy storage method
    records_list = get_patient_feature_lead_records(path_to_load,
                                                    feature_name,
                                                    patient_list,
                                                    lead_list)
    data_struct = load_feature_from_input_list(path_to_load,
                                               records_list)
    window_data_struct = load_feature_window_from_input_list(path_to_load,
                                                         records_list)
    
    # convert to pandas
    records_data = data_struct[0]
    records_sample = window_data_struct[0]    
    records_mdata = data_struct[1]
    records_patient =[record_name[1][0] for record_name in records_list]
    records_seizure =[record_name[1][-1] for record_name in records_list]
    
    return convert_records_to_pd_dataframe(records_data, records_sample, records_mdata,
                                          records_patient, records_seizure)
                                           

def convert_records_to_pd_dataframe(records_data, records_sample, records_mdata,
                                    records_patient, records_seizure):
    # Get a list of dataframes for each record
    records_pd_list = [_convert_record_to_pd_dataframe(data, sample, mdata,
                                                      patient, seizure)
                       for data, sample, mdata, patient, seizure in zip(
                                              records_data, records_sample, records_mdata,
                                              records_patient, records_seizure)]
    
    # Concatenate the records dataframe into a single matrixof data
    final_data_struct_pd = pd.concat(records_pd_list)
    
    return final_data_struct_pd
                                          

def _convert_record_to_pd_dataframe(data, sample, mdata, patient, seizure):
    # First, convert data
    record = pd.DataFrame(data.T, columns=mdata['feature_legend'])
    
    # Add time sample
    record['time_sample'] = sample
    
    # Add patient number
    record['patient_nr'] = patient
    
    # Add seizure number
    record['seizure_nr'] = seizure
    
    return record   


def apply_label_structure(records, records_labels_struct):
    records['label'] = np.empty(len(records)) * np.nan
    records['color'] = ['m'] * len(records)

    for time_period, time_period_label_struct in records_labels_struct.iteritems():
        label = time_period_label_struct['label']
        color = time_period_label_struct['color']
        intervals_samples = time_period_label_struct['intervals_samples']
        for interval in intervals_samples:
            set_label_from_interval_sample(records,
                                           interval,
                                           label,
                                           color)

            
def set_label_from_interval_sample(data,
                                   interval_sample,
                                   label,
                                   color):
    # Get bounds from interval and appropriate label
    lower = interval_sample[0]
    upper = interval_sample[1]

    # Set label accordingly
    sample_ix = data['time_sample'].between(lower, upper)
    data.loc[sample_ix, 'label'] = label
    data.loc[sample_ix, 'color'] = color
    

def convert_record_to_pd_dataframe(data, sample, mdata):
    # First, convert data
    record = pd.DataFrame(data.T, columns=mdata['feature_legend'])
    
    # Add time sample
    record['time_sample'] = sample
    
    
def convert_to_pandas(path_to_load, path_to_map,
                      patient_list, feature_slot,
                      lead_list, label_struct):
    # Load the data from disk and conver to pandas dataframe
    data = load_records(path_to_load, path_to_map,
                        patient_list, feature_slot,
                        lead_list)
    
    # Apply labeling strategy
    apply_label_structure(data, label_struct)
    
    return data
    