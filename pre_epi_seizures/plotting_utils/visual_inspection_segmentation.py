
from pre_epi_seizures.storage_utils.storage_utils_hdf5 import \
    load_signal, save_signal, delete_signal, list_group_signals

from pre_epi_seizures.Preprocessing.pre_processing import *

from pre_epi_seizures.stats_utils.statistics import create_set_from_disk

import matplotlib.pyplot as plt
import seaborn as sns

import os


sns.set()

# def plot_time_no_label(feature_array_new, window_array_new, xlim_down, xlim_up, y_lim_down, ylim_up, xlabel, ylabel):
#     plt.figure()
#     for feature, window in zip(feature_array_new, window_array_new):
      


# def _plot_time_no_class_label(feature_array_new, window_array_new,
#                               xlim_down, xlim_up, y_lim_down, ylim_up,
#                               xlabel, ylabel, ax = None):
#     if ax is None:
#         fig, ax = plt.subplots()

#     for feature, window in zip(feature_array_new, window_array_new):
#         plt.plot(window, feature)

#     plt.xlim([xlim_down, xlim_up])
#     plt.ylim([ylim_up, y_lim_down])
#     plt.xlabel[xlabel]
#     plt.ylabel['ylabel']


# -----------------------------------------------------*****************************-------------------------------------
def load_all_features_from_disk(path_to_load, feature_group_name):
    feature_group_extracted = feature_group_name
    feature_group_name_extracted = list_group_signals(path_to_load, feature_group_extracted)['signals']
    feature_group_name_extracted= [group_name for group_name in feature_group_name_extracted if 'window_' not in group_name[1]]

    try:
        print feature_group_extracted
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
        print feature_group_extracted
        signal_structure = load_signal(path_to_load, feature_group_name_extracted)
        extracted_features = [get_multiple_records(get_one_signal_structure(signal_structure, group_name))
                              for group_name in feature_group_name_extracted]
        mdata = [get_mdata_dict(get_one_signal_structure(signal_structure, group_name))
                 for group_name in feature_group_name_extracted]

        return extracted_features, mdata
    except Exception as e:
        _logger.debug(e)

    return extracted_features, mdata


def load_segmented_feature_group(segmentation_feature_group):
    all_features_str = segmentation_feature_group.split('/')
    print all_features_str
    segmented_feature_group = segmentation_feature_group[:segmentation_feature_group.index(all_features_str[-1])-1]

    return segmented_feature_group

def _plot_visual_inspection_xlim_var(feature_seg, feature_window_seg, segmentation_seg):
    f1 = plt.figure()
    ax1 = f1.add_subplot(1,1,1) 
    print feature_seg
    print feature_window_seg
    print segmentation_seg


    feature_line = ax1.plot(feature_window_seg, feature_seg, 'b-')
    print 'feature done!!'
    seg_line = ax1.plot(feature_window_seg[segmentation_seg], feature_seg[segmentation_seg], 'yo')
    print 'segmented done!!'

    return f1, ax1, feature_line, seg_line

def _axis_label(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax


def _axis_set_limits(ax, xlim_down, xlim_up):
    ax.set_xlim(xlim_down, xlim_up)
    ax.set_ylim(-1000, 1000)
    return ax

def _axis_set_grid(ax):
    ax.grid(color='tab:gray', linestyle='-', linewidth=0.5)
    return ax

def _axis_set_legend(ax, lines_list, legend_list):
    ax.legend(legend_list)
    return ax


def _plot_visual_inspection(path_to_save, feature,
                            segmentation, feature_window,
                            segmentation_window,
                            feature_name, segmentation_name, xlim_var=10*1000):
    # feature_array_new = [feature, feature[segmentation]]
    # window_array_new = [window, window[segmentation]]

    

    xlabel = 'samples'
    ylabel = 'digitized units'
    for xlim in xrange(xlim_var, len(feature), xlim_var):
        x_down = xlim - xlim_var
        x_up = xlim
        # stopp
        feature_seg = feature[x_down:x_up]
        feature_window_seg = feature_window[x_down:x_up]

        ind_segmentation_seg = np.where(np.logical_and(segmentation<x_up, segmentation>x_down))[0]
        segmentation_seg = segmentation[ind_segmentation_seg] - x_down

        f1, ax1, feature_line, seg_line, = _plot_visual_inspection_xlim_var(feature_seg, feature_window_seg, segmentation_seg)
        ax1 = _axis_label(ax1, xlabel, ylabel)
        ax1 = _axis_set_legend(ax1, [feature_line, seg_line], [feature_name, segmentation_name])
        ax1 = _axis_set_grid(ax1)
        figure_directory = path_to_save
        fetch_directory(figure_directory)
        f1.savefig(figure_directory + str(xlim), transparent=False)
        plt.close()



def plot_visual_inspection(path_to_save, feature_group_name, feature_array,
                           feature_window_array, segmentation_array,
                           segmentation_window_array,
                           feature_legend, segmentation_legend):

    name_record = frature_group_name[1]
    path_to_save = path_to_save + name_record + '/'

    print 'here'
    print feature_array
    print segmentation_array
    print segmentation_legend
    print feature_legend
    xlim_var = 10*1000
    for feature, segmentation, feature_window,\
        segmentation_window, feature_name, segmentation_name\
        in  zip(feature_array, feature_window_array,
                         segmentation_array, segmentation_window_array,
                         feature_legend, segmentation_legend):
        _plot_visual_inspection(path_to_save, feature, feature_window,
                                segmentation, segmentation_window, 
                                feature_name, segmentation_name)


def fetch_directory(directory):
        # ** Create directory
    if not os.path.exists(directory):
        print 'MAKING'
        os.makedirs(directory)





path_to_load = '/Volumes/ASSD/pre_epi_seizures/h5_files/processing_datasets/seizure_datasets_new.h5'
path_to_map= '/Volumes/ASSD/pre_epi_seizures/h5_files/processing_datasets/seizure_datasets_new_map.txt'
path_to_save = '/Volumes/ASSD/pre_epi_seizures/plotting_utils/teste'



rpeak_name = 'rpeak_detection'

feature_groups = get_feature_group_name_list(path_to_map,
                                         rpeak_name + '#')

segmented_feature_group = load_segmented_feature_group(feature_groups[0])

segmentation_feature_group = feature_groups[0]


segmentation_features,\
    segmentation_features_mdata = load_all_features_from_disk(
                                                path_to_load,
                                                segmentation_feature_group)

segmented_features,\
    segmented_features_mdata = load_all_features_from_disk(
                                                path_to_load,
                                                segmented_feature_group)

segmentation_features_windows,\
    segmentation_features_windows_mdata = load_all_feature_windows_from_disk(
                                                 path_to_load,
                                                 segmentation_feature_group)

segmented_features_windows,\
    segmented_features_windows_mdata = load_all_feature_windows_from_disk(
                                                 path_to_load,
                                                 segmented_feature_group)

set_feature_group_name = create_set_from_disk(segmentation_feature_group)

# print 'SEGMENTED'
# print segmented_features
# print ''
# print 'SEGMENTATION'
# print segmentation_features


# ---------------------- TESTING------------------------------
sz_nr = 0

# feature_arrays
feature_array = segmented_features[sz_nr]
segmentation_array = segmentation_features[sz_nr]

feature_window_array = segmented_features_windows[sz_nr]
segmentation_window_array = segmentation_features_windows[sz_nr]

# set_feaure_groups
feature_group_name = set_feature_group_name[sz_nr]

# legend
feature_legend =  segmented_features_mdata[sz_nr]['feature_legend']
segmentation_legend =  segmentation_features_mdata[sz_nr]['feature_legend']

path_to_save = '/Volumes/ASSD/pre_epi_seizures/plotting_utils/teste'

plot_visual_inspection(path_to_save, feature_group_name,
                           feature_array,
                           feature_window_array, segmentation_array,
                           segmentation_window_array,
                           feature_legend, segmentation_legend)



stop