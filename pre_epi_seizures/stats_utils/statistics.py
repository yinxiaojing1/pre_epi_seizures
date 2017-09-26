from pre_epi_seizures.Preprocessing.pre_processing import load_feature

from pre_epi_seizures.classification.labels import create_labels

from pre_epi_seizures.storage_utils.storage_utils_hdf5 import \
    load_signal, save_signal, delete_signal, list_group_signals


from hist import *



import numpy as np
# # def load_statistic(path_to_load, statistic_to_load,
# #                     files='just_new_data', sampling_rate=1000,
# #                     **feature_groups_required):

# # feature_group_to_process = feature_groups_required['feature_group_to_process']


# # def create_labels_for_feature


def create_set_from_disk(path_to_load, feature_group):
    list_signals = list_group_signals(path_to_load, feature_group)['signals']

    return list_signals


def create_labels_feature(path_to_load, feature_group, list_signals):
    try:
        window_str = feature_group[feature_group.index('w'):feature_group.index('msec')]
    except Exception as e:
        print e
        window_sample = 1

    len_record_before_str = feature_group[0: feature_group.index('_')]
    len_record_after_str = feature_group[feature_group.index('_'):feature_group.index('/')]
    len_record = int(len_record_before_str) + int(len_record_after_str)

    print len_record
    stop
    windows = np.arange(0, len(window_sample), window_sample)
    list_signals = list_group_signals(path_to_load, feature_group)['signals']


def main():

    #signal
    sampling_rate = 1000
    time_before_seizure = 50
    time_after_seizure = 20
    # path_to_load = '~/Desktop/phisionet_seizures_new.h5'
    # sampling_rate = 1000
    path_to_load = '/Volumes/ASSD/pre_epi_seizures/h5_files/processing_datasets/seizure_datasets_new.h5'

    dataset_name = str(
    time_before_seizure*60) + '_' + str(time_after_seizure*60)
    raw_name = 'raw'
    baseline_removal_name = 'baseline_removal'
    raw_dataset_name = dataset_name + '/' + raw_name
    baseline_removal_dataset_name = raw_dataset_name + '/' + baseline_removal_name
    decimated_dataset_name = baseline_removal_dataset_name + '/' + 'decimation'
    eks_dataset_name = decimated_dataset_name + '/' + 'eks_smoothing'
    interpolated_dataset_name = eks_dataset_name + '/' + 'interpolation'

    # dsfsdf
    raw = load_feature(path_to_load, raw_name, files='existent', feature_group_to_process=dataset_name)[0]

    set_structure = create_set_from_disk(path_to_load, raw_dataset_name)

    # stop
    labels = create_labels_feature(path_to_load, raw_dataset_name, set_structure)

    windows = [np.linspace(0, len(sz)-1, len(sz)) for sz in raw]
    labels = create_labels(1000,
                        windows_list=windows,
                        inter_ictal = ((0, 5), 'g'),
                        pre_ictal = ((5,10), 'orange'),
                        unspecified=((10,35), 'blue'),
                        post_ictal = ((35,40), 'y'),
                        ictal = ((30,35), 'r'))

    print labels

    stop
    hist = load_feature(path_to_load, 'histogram')

main()