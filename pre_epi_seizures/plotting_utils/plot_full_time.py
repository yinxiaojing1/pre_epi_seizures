from pre_epi_seizures.storage_utils.storage_utils_hdf5 import \
    load_signal, save_signal, delete_signal, list_group_signals

from pre_epi_seizures.Preprocessing.pre_processing import *

from pre_epi_seizures.stats_utils.statistics import create_set_from_disk

from plot_storage import *

import matplotlib.pyplot as plt
import seaborn as sns


import os


def fetch_directory(directory):
        # ** Create directory
    if not os.path.exists(directory):
        print 'MAKING'
        os.makedirs(directory)


def __plot_full_time(feature, feature_window):
    f1 = plt.figure()
    ax1 = f1.add_subplot(1,1,1)
    l1 = ax1.plot(feature_window, feature)
    l1 = ax1.axvline(50*60*1000, color='k', linestyle='--')
    return f1


def _plot_full_time(argument):
    return np.apply_along_axis(__plot_full_time, 1, argument[0], argument[1])


def _save_fig(figure_array, directory, feature_names, seizure_name):
    directory += '/' + seizure_name
    fetch_directory(directory)
    [figure.savefig(directory + '/' + feature_name,
               transparent=False)
    for figure, feature_name in zip(figure_array, feature_names)]

    print 'DOOOOOOOOOOOONNNNNNNNNNNNNEEEEEEEEEEEEEEEEE'


def plot_full_time(path_to_save, feature_group,
                   feature_names, feature_array_list,
                   feature_window_list, seizure_name_list):

    argument = zip(feature_array_list, feature_window_list)
    fig_list = map(_plot_full_time, argument)

    directory = path_to_save + feature_group[1:]

    [_save_fig(figure_array, directory, feature_names, seizure_name[1])
     for figure_array, seizure_name in zip(fig_list, seizure_name_list)]



def plot_all_seizures(path_to_load, path_to_save, feature_group):

    feature_array_list,\
        feature_mdata = load_all_features_from_disk(
                                                    path_to_load,
                                                    feature_group)

    feature_window_list,\
        features_windows_mdata = load_all_feature_windows_from_disk(
                                                     path_to_load,
                                                     feature_group)

    set_feature_group_name = create_set_from_disk(path_to_load, feature_group)
    feature_names = feature_mdata[0]['feature_legend']


    plot_full_time(path_to_save, feature_group,
                   feature_names, feature_array_list,
                   feature_window_list, set_feature_group_name)


print 'hello'
sns.set()


path_to_load = '/Volumes/ASSD/pre_epi_seizures/h5_files/processing_datasets/seizure_datasets_new.h5'
path_to_map= '/Volumes/ASSD/pre_epi_seizures/h5_files/processing_datasets/seizure_datasets_new_map.txt'
path_to_save = '/Volumes/ASSD/pre_epi_seizures/plotting_utils/hrv_features/'

feature_group = 'hrv_time_features'



feature_groups = get_feature_group_name_list(path_to_map,
                                             feature_group+ '#')

print feature_groups
# stop
for feature_group in feature_groups:
    plot_all_seizures(path_to_load, path_to_save, feature_group)
