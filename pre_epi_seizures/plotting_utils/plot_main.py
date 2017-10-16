import matplotlib.pyplot as plt


from pre_epi_seizures.storage_utils.storage_utils_hdf5 import \
    load_signal, save_signal, delete_signal, list_group_signals


from pre_epi_seizures.Preprocessing.pre_processing import *


import os




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



# path_to_load = '/Volumes/ASSD/pre_epi_seizures/h5_files/processing_datasets/satistic_datasets.h5'
path_to_load = '/Volumes/ASSD/pre_epi_seizures/h5_files/processing_datasets/seizure_datasets_new.h5'
path_to_map= '/Volumes/ASSD/pre_epi_seizures/h5_files/processing_datasets/seizure_datasets_new_map.txt'


if not os.path.exists(directory):
#     os.makedirs(directory)

feature_name = 'baseline_removal'

feature_groups = get_feature_group_name_list(path_to_map,
                                         feature_name + '#')

# rpeak_name = 'rpeak_detection'

# feature_groups = get_feature_group_name_list(path_to_map,
#                                          rpeak_name + '#')


features_list, mdata = load_all_features_from_disk(path_to_load, feature_groups[0])
print features_list

plt.plot(features_list[2][0])
plt.savefig("baseline_removal.png")
# print mdata
# # stop

# directory = '/Volumes/ASSD/pre_epi_seizures/plotting_utils/features/' + feature_groups[0] + '/'

# if not os.path.exists(directory):
#     os.makedirs(directory)

# for i,feature_array in enumerate(features_list):
#     print i

#     for feature in feature_array:
#         for xlim in xrange(10, len(feature), 10):
#             print xlim
#             plt.figure()
#             plt.plot(feature)
#             plt.xlim([(xlim - 10) * 1000, xlim * 1000])
#             plt.ylim([-1000, 1000])
#             plt.savefig(directory + str(i) + feature_name + str(xlim))
#             plt.close()

# stop

# 45
hrv_groups = get_feature_group_name_list(path_to_map,
                                         'hrv_computation#')
features, mdata = load_all_features_from_disk(path_to_load, hrv_groups[0])

print features[0]

plt.plot(features[2][0])
plt.savefig("hrv.png")