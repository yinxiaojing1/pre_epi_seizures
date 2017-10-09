import matplotlib.pyplot as plt

from pre_epi_seizures.storage_utils.storage_utils_hdf5 import \
    load_signal, save_signal, delete_signal, list_group_signals


from pre_epi_seizures.Preprocessing.pre_processing import *



path_to_load = '/Volumes/ASSD/pre_epi_seizures/h5_files/processing_datasets/seizure_datasets_new.h5'
path_to_map= '/Volumes/ASSD/pre_epi_seizures/h5_files/processing_datasets/seizure_datasets_new_map.txt'
raw_groups = get_feature_group_name_list(path_to_map,
                                         'raw#')


print raw_groups


stop