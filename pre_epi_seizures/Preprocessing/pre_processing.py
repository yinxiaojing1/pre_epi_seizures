from pre_epi_seizures.storage_utils.storage_utils_hdf5 import  load_signal, save_signal

from pre_epi_seizures.logging_utils.formatter_logging import logger as _logger

from filtering import baseline_removal, create_filtered_dataset

from segmentation import create_rpeak_dataset, create_heart_beat_dataset

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import logging
import sys
import copy
import memory_profiler


def plot_models(models, names, colors):
    plt.figure()
    for ii, (model, name) in enumerate(zip(models, names), 1):
        plt.subplot(len(models), 1, ii)
        plt.title(name)
        for sig, color in zip(model.T, colors):
            _logger.debug(sig)
            plt.plot(sig, color=color)
            # if scatter not None:
            #     plt.scatter(scat, color=scatter_color)

    plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.46)
    plt.show()


def shape_array(array):
    return np.array([array]).T


# @profile
def main(arg):

    # _logger.debug("Starting Gaussian Fit...")
    # _logger.debug("vlfsn")

# #*****************phisionet data*******************************
#     path = '~/Desktop/phisionet_dataset.h5'
#     name = ['sz_'+str(arg)]
#     group = ['medianFIR', 'raw']

#     # # baseline_removal(path, name, 'raw')
#     # # create_rpeak_dataset(path, name, group)
#     X = load_signal(path, name, group)

#     # path = '~/Desktop/dummy.h5'
#     # # rpeaks = load_signal(path,'rpeaks_'+name, group)

#     _logger.debug(X[0]['signal'])
#     _logger.debug(X[0]['mdata'])

    # heart_beat, rpeak = create_heart_beat_dataset(path=path,
    #                                               name=name,
    #                                               group=group,
    #                                               save_dfile=None)

    # _logger.debug(np.asmatrix(rpeaks['signal'].T))
    # models = [X['signal'][40*200:60*200], X_raw['signal'][40*200:60*200]]
    # names = ['Filtered', 'Raw' ]
    # colors = ['red']
    # plot_models(models, names, colors)

#****************************************************************
    path = '~/Desktop/HSM_data.h5'
    name = ['FA77748S']
    group = ['/PATIENT1/crysis']

    raw = load_signal(path, name, group)

    begin_seizure_seconds = raw[0]['mdata']['crysis_time_seconds']
    begin_seizure_sample = int(1000*begin_seizure_seconds[0])

    sampling_rate_hertz = 1000

    signal = raw[0]['signal']
    no_seizure_ecg_raw = signal[0:begin_seizure_sample,0]

    ecg_10min_5min_raw = signal[begin_seizure_sample
                                -sampling_rate_hertz*10*60:
                                begin_seizure_sample
                                + sampling_rate_hertz*5*60]

    _logger.debug(begin_seizure_sample)

    _logger.debug(raw[0]['mdata'])

    # decimated = sp.signal.decimate(raw[0]['signal'], 2)
    # stop
    path_to_save = '~/Desktop/seizures_datasets.h5'
    save_signal(path=path_to_save, signal=ecg_10min_5min_raw, mdata={}, name = '10_15', group='raw')
    baseline_removal(path_to_save, ['10_15'], ['raw'])

    # group = ['medianFIR']

    # # X = load_signal(path, name, group)
    # # _logger.debug(X[0]['signal'])

    # start = 250*1000
    # end = 300*1000
    models = [ecg_10min_5min_raw]
    names = ['raw']
    colors = ['red']
    plot_models(models, names, colors)

main(3)