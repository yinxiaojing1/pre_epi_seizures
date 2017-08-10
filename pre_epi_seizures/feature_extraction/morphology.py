from pre_epi_seizures.storage_utils.storage_utils_hdf5 import \
    load_signal, save_signal

from pre_epi_seizures.storage_utils.data_handlers import *

from pre_epi_seizures.logging_utils.formatter_logging import logger as _logger

# from filtering import baseline_removal, noise_removal, create_filtered_dataset

# from segmentation import create_rpeak_dataset, create_heart_beat_dataset,\
#     compute_beats

from eksmoothing import get_phase, mean_extraction,\
    beat_fitter, ecg_model

# from biosppy.signals import ecg

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import logging
import sys
import copy
import memory_profiler



def compute_baseline_model(signal, rpeaks, start, end):
    phase = get_phase(signal, rpeaks)
    mnx, sdx, mnphase = mean_extraction(signal[start:end],
                                        phase[start:end],
                                        bins = 500)
    values = beat_fitter(mnx,mnphase)
    model = ecg_model(values, mnphase)

    return model 



