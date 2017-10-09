import matplotlib.pyplot as plt

from pre_epi_seizures.storage_utils.storage_utils_hdf5 import \
    load_signal, save_signal, delete_signal, list_group_signals


from pre_epi_seizures.Preprocessing.pre_processing import *




def load_all_features_from_disk(path_to_load, feature_group_name):
    feature_group_extracted = feature_group_name

    feature_group_name_extracted = list_group_signals(path_to_load, feature_group_extracted)['signals']
    print feature_group_name_extracted
    # print feature_group_name_extracted
    try:
        signal_structure = load_signal(path_to_load, feature_group_name_extracted)
        extracted_features = [get_multiple_records(get_one_signal_structure(signal_structure, group_name)) for group_name in feature_group_name_extracted]
        mdata = [get_mdata_dict(get_one_signal_structure(signal_structure, group_name)) for group_name in feature_group_name_extracted]
        return extracted_features, mdata
    except Exception as e:
        _logger.debug(e)

    return extracted_features, mdata

path_to_load = '/Volumes/ASSD/pre_epi_seizures/h5_files/processing_datasets/seizure_datasets_new.h5'
path_to_map= '/Volumes/ASSD/pre_epi_seizures/h5_files/processing_datasets/seizure_datasets_new_map.txt'
raw_groups = get_feature_group_name_list(path_to_map,
                                         'raw#')

features, mdata = load_all_features_from_disk(path_to_load, raw_groups[0])

print features 

for 
feature_example = features[0]


plt.plot(feature_example)
plt.savefig("out.png")