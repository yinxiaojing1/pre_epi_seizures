from classification.load_for_class import *
#from labels import *
#from supervised_training import *

from storage_utils.patients_data_new import *
import pandas as pd
import numpy as np


def load_records(path_to_load, path_to_map, patient_list, feature_name, lead_list):

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
    records_patient =[int(record_name[1].split('_')[0]) for record_name in records_list]
    
    try:
        records_seizure =[int(record_name[1].split('_')[-1]) for record_name in records_list]      
    except:
        records_seizure =[0 for record_name in records_list]
        
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
    
    # Remove corrupted files
    records_pd_list= [record_pd
                     for record_pd in records_pd_list
                     if record_pd is not None]

    if records_pd_list:
        # Concatenate the records dataframe into a single matrixof data
        final_data_struct_pd = pd.concat(records_pd_list)

        return final_data_struct_pd


def _convert_record_to_pd_dataframe(data, sample, mdata, patient, seizure):
    print 'data'
    print patient
    print seizure
    
    try:
        # First, convert data
        record = pd.DataFrame(data.T, columns=mdata['feature_legend'])

        # Add time sample
        record['time_sample'] = sample

        # Add patient number
        record['patient_nr'] = str(patient)

        # Add seizure number
        record['seizure_nr'] = str(seizure)

    except Exception as e:
        print e
        record = None

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
    
    if data is not None:

        # Apply labeling strategy
        apply_label_structure(data, label_struct)

        return data
    
    
def add_seizure_types(data,
                      patient_id,
                      seizure_nr_id,
                      type_id,
                      localization_id
                     ):
    
    
    g = data.groupby([patient_id, seizure_nr_id])
    
    patients_seizures_in_data = g.groups.keys()
    
    # Set new collumns
    data[type_id] = np.nan
    data[localization_id] = np.nan
    
    for patient, seizure in patients_seizures_in_data:
        patient_dict = patients[str(patient)]
        type_seizure = patient_dict[type_id][int(seizure)]
        localization = patient_dict[localization_id][int(seizure)]
        data[type_id].loc[(data[patient_id]==patient) & (data[seizure_nr_id]==seizure)] = type_seizure
        data[localization_id].loc[(data[patient_id]==patient) & (data[seizure_nr_id]==seizure)] = localization
        
    return data