from pre_epi_seizures.storage_utils.storage_utils_hdf5 import \
    load_signal, save_signal

from pre_epi_seizures.storage_utils.data_handlers import *

from pre_epi_seizures.logging_utils.formatter_logging import logger as _logger

from pre_epi_seizures.feature_extraction.heart_rate_variability import insta_heart_rate

# # from filtering import baseline_removal, create_filtered_dataset

# # from segmentation import create_rpeak_dataset, create_heart_beat_dataset,\
# #     compute_beats

# from Filtering.gaussian_fit import get_phase, mean_extraction,\
#     beat_fitter, ecg_model

# from Filtering.filter_signal import filter_signal

from biosppy.signals import ecg

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import logging
import sys
import copy
import memory_profiler



#signal
time_before_seizure = 30
time_after_seizure = 10
path_to_load = '~/Desktop/seizure_datasets.h5'
name_list = [str(time_before_seizure*60) + '_' + str(time_after_seizure*60)]
group_list_baseline_removal = ['medianFIR']

f_loss = lambda label:'rpeaks_'+label

labels = map(str, range(0,6))

print labels
rpeaks_names = map(f_loss, labels)
print rpeaks_names

group_list = group_list_baseline_removal * len(labels)
print group_list

rpeaks = load_signal(path_to_load ,zip(group_list, rpeaks_names))

print rpeaks